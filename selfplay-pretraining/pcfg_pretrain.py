"""
Random-PCFG pre-pretraining baseline (paper appendix H), FLOP-budgeted like selfplay.py so it can be
dropped into compute_matched.sh (WARMUP=pcfg). Saves a checkpoint train.py can warm-start from.

Grammar sampling (appendix H):
  terminals  : n_Sigma ~ U{2..16} distinct bytes from {1..255} (0 is padding, never emitted)
  nonterms   : |V| ~ U{1..8}, start symbol V0; each gets U{1..4} productions
  probs      : i.i.d. U(0,1) + 1e-6, normalized
  rhs        : U{1..4} symbols, each a terminal w.p. 0.5 (uniform over Sigma) else a uniform nonterminal
  repair     : a nonterminal with no terminal-only production has one uniformly chosen production's rhs
               replaced by a fresh terminal-only string (probability unchanged)
Derivation   : leftmost expansion with an explicit stack; stops when the stack empties, the word reaches
               64 bytes, or 10^4 expansions; if nothing was emitted, emit the start symbol's stored
               terminal-only yield. Each training row uses ONE fresh grammar; words are concatenated until
               the row is full (last word truncated).

    python pcfg_pretrain.py config/selfplay_1m.py --block_size=1024 --max_flops=5e13 --out_dir=out-pcfg
"""
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
out_dir = "out-pcfg"
task = "pcfg"  # pcfg (appendix H) | dyck | induction | zipf | mix  (see synth.py)
n_layer = 4
n_head = 4
n_embd = 128
mlp_hidden = 0
block_size = 4096  # paper: 4095-byte rows at a 4096 context
batch_rows = 8
learning_rate = 3e-3
warmup_steps = 20
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
max_flops = 0.0  # stop once this much compute is spent (0 = max_steps)
max_steps = 100000
log_interval = 50
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
# accepted-but-ignored self-play keys, so the same MODEL/SP_ARGS strings work for both warm-ups
eval_interval = 0
max_rounds = 0
ckpt_interval = 0
pool_size = 0
machine = ""
exec_threads = 0
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurator.py")).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

OUT_PREFIX = ord("O")
MAX_WORD = 64
MAX_EXPANSIONS = 10_000


def sample_grammar(rng):
    sigma = rng.sample(range(1, 256), rng.randint(2, 16))
    nV = rng.randint(1, 8)
    rules = []  # per nonterminal: (list of rhs, cumulative probs); terminal b encoded as -b, nonterminal as index
    yields = []
    for _ in range(nV):
        k = rng.randint(1, 4)
        rhss = [[-rng.choice(sigma) if rng.random() < 0.5 else rng.randrange(nV) for _ in range(rng.randint(1, 4))]
                for _ in range(k)]
        if not any(all(s < 0 for s in r) for r in rhss):  # productivity repair
            rhss[rng.randrange(k)] = [-rng.choice(sigma) for _ in range(rng.randint(1, 4))]
        w = [rng.random() + 1e-6 for _ in range(k)]
        tot = sum(w)
        cum = list(np.cumsum([x / tot for x in w]))
        rules.append((rhss, cum))
        yields.append([-s for s in next(r for r in rhss if all(s < 0 for s in r))])
    return rules, yields


def derive(rules, yields, rng):
    stack, out, n = [0], [], 0
    while stack and len(out) < MAX_WORD and n < MAX_EXPANSIONS:
        s = stack.pop()
        if s < 0:
            out.append(-s)
            continue
        rhss, cum = rules[s]
        u = rng.random()
        j = next((i for i, c in enumerate(cum) if u <= c), len(cum) - 1)
        stack.extend(reversed(rhss[j]))
        n += 1
    return out or list(yields[0])


def sample_row(n, rng):
    rules, yields = sample_grammar(rng)
    row = []
    while len(row) < n:
        row.extend(derive(rules, yields, rng))
    return row[:n]


if __name__ == "__main__":
    import synth
    gen = sample_row if task == "pcfg" else synth.GENERATORS[task]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type != "cuda" or dtype == "float32" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    model = GPT(GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden)).to(device)
    opt = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    flops_per_step = batch_rows * block_size * model.flops_per_token(block_size)
    print(f"params {model.get_num_params() / 1e6:.2f}M, {flops_per_step:.3e} FLOPs/step")

    flops, t0, step = 0.0, time.time(), 0
    while step < max_steps and not (max_flops > 0 and flops >= max_flops):
        rows = torch.tensor([gen(block_size, rng) for _ in range(batch_rows)], dtype=torch.long)
        y = rows.to(device)
        x = torch.cat([torch.full((batch_rows, 1), OUT_PREFIX, device=device), y[:, :-1]], 1)
        lr = learning_rate * min(1.0, (step + 1) / warmup_steps)  # short warmup, then constant (as self-play)
        for g in opt.param_groups:
            g["lr"] = lr
        with ctx:
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, 256), y.view(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        flops += flops_per_step
        if step % log_interval == 0:
            print(f"step {step} | {flops:.2e} FLOPs | loss {loss.item() / math.log(2):.4f} bpb | {time.time() - t0:.0f}s")
        step += 1

    torch.save({
        "learner": model.state_dict(), "round": step, "flops": flops, "config": config,
        "model_args": dict(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden),
    }, os.path.join(out_dir, "ckpt.pt"))
    print(f"done: {step} steps, {flops:.3e} FLOPs, final loss {loss.item() / math.log(2):.4f} bpb")

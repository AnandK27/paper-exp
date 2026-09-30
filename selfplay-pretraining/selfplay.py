"""
Self-Play Pretraining with Zero Data (Cowsik et al., arXiv:2609.30063), nanoGPT-style.

Two Llama-style transformers co-evolve from random init:
  generator g_phi : proposes Brainf*ck(+macros) programs  (RL + reward-weighted SFT)
  learner  pi_theta: next-byte prediction on the programs' outputs (plain cross-entropy)

Each round e:
  1. build pool B_e = fresh (sampled from g_phi) U mutations (MAP-Elites archive) U replay (bank)
  2. execute every program on U with a fresh random input tape -> y_i (T bytes)
  3. reward r_i = | < grad L(y_i; theta_e), P_e * (theta_{floor(e/2)} - theta_e) > |      (eq. 2)
     computed with forward-mode AD (one JVP per micro-batch, no per-sample gradients)
  4. learner: one AdamW step on mean_i L(y_i)                                             (eq. 1)
  5. generator: GRPO-style policy gradient with KL-to-uniform-prior in the advantage and
     sequence-level importance ratios for replayed programs (eq. 3-4), plus expert iteration (eq. 5)

Zero-shot eval: bits/byte on held-out natural byte data (never trained on).

    python selfplay.py config/selfplay_1m.py
"""
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import importlib
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# I/O
out_dir = "out-selfplay"
eval_interval = 128  # rounds
log_interval = 1
ckpt_interval = 512
sample_interval = 256  # dump example programs/outputs
eval_files = "data/eval/*.bin"  # glob of raw byte files for zero-shot bits/byte
eval_records = 256  # held-out sequences per dataset (paper: 256)
eval_len = 256  # record length incl. prefix byte (paper: 256)
seed = 1337
# models (learner and generator share the architecture, as in the paper)
n_layer = 4
n_head = 4
n_embd = 128
mlp_hidden = 0  # 0 -> ~8/3 * n_embd
block_size = 4096  # learner context; programs emit up to this many bytes
# machine U
machine = "bf"  # bf (paper's Brainf*ck tape) | stack (stackm.c)
max_prog_len = 64  # max body tokens (excluding the terminating F)
tape_len = 256
max_steps = 1 << 17
mask_padding = True  # learner loss over emitted bytes only (not the zero padding after halt)
# pool / round structure
pool_size = 1024  # M_e programs per round (1024 x 4096 bytes = 4.19M tokens/round)
mut_frac = 0.25  # fraction of the fresh slots filled by MAP-Elites mutations
replay_frac = 0.25  # fraction of the pool replayed from the bank
bank_size = 1 << 20
archive_per_niche = 8
archive_decay = 0.97
max_rounds = 8192  # 8192 x 4.19M = 34.36B tokens (the paper's max budget)
# learner optimizer (short warmup, then constant LR -> every checkpoint is a valid early stop)
learning_rate = 3e-3
warmup_rounds = 50
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
tokens_per_micro = 65536  # learner micro-batch token budget
# generator
gen_lr_ratio = 0.1  # generator lr = gen_lr_ratio * learning_rate
kl_beta = 0.02
lambda_ei = 1.0
gen_grad_clip = 1.0
# reward
reward_mode = "canonical"  # canonical | signed | shuffle | last_step | uniform (Table 5 ablations)
reward_attn = "manual"  # attention impl under forward-mode AD: manual | sdpa
reward_tokens_per_micro = 32768
snap_every = 8  # store learner snapshots every k rounds; theta_past = latest snapshot <= floor(e/2)
adam_eps = 1e-8
# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
compile = False
exec_threads = 0  # 0 -> min(32, cpu_count)
# learner implementation: llama (model.py) | nanochat (karpathy/nanochat GPT, see nanochat_learner.py)
learner_impl = "llama"
nanochat_dir = "~/nanochat"
nanochat_depth = 8
nanochat_seq_len = 2048
max_flops = 0.0  # stop self-play once this much compute is spent (0 = run max_rounds); for compute-matched tests
# which FLOPs count against max_flops (and are recorded as the checkpoint's "flops"):
#   all     = learner training + JVP reward + generator sampling + generator training
#   learner = learner training only (the paper's "effective compute" K x params x tokens)
flops_budget = "all"
check_reward = False  # at round 2, verify the JVP reward against explicit autograd gradients
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurator.py")).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type != "cuda" or dtype == "float32" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

M = importlib.import_module({"bf": "bf", "stack": "stackm"}[machine])
A = len(M.ALPHABET)
ALLOWED = torch.tensor(M.ALPHABET_BYTES.astype(np.int64), device=device)  # generator logit restriction
END_IDX = M.ALPHABET.index(M.END)
LOG_A = math.log(A)

# models ----------------------------------------------------------------------
if learner_impl == "nanochat":
    from nanochat_learner import NanochatLearner

    learner = NanochatLearner(nanochat_dir, nanochat_depth, nanochat_seq_len, device)
    n_layer, n_embd = learner.n_layer, learner.n_embd  # generator mirrors the learner's shape
    n_head = max(1, n_embd // 128)
    TOK, PREFIX = learner.byte_to_token, learner.bos_id  # bytes -> nanochat single-byte tokens, <|bos|> prefix
else:
    learner = GPT(GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden)).to(device)
    TOK, PREFIX = None, M.OUT_PREFIX
generator = GPT(GPTConfig(block_size=max_prog_len + 2, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden)).to(device)
# initialize the generator near the uniform prior g0: zero output head -> uniform over the alphabet
torch.nn.init.zeros_(generator.lm_head.weight)
print(f"learner params: {learner.get_num_params() / 1e6:.2f}M, generator params: {generator.get_num_params() / 1e6:.2f}M")

opt_l = learner.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
opt_g = generator.configure_optimizers(0.0, learning_rate * gen_lr_ratio, (beta1, beta2), device_type)
for g in opt_l.param_groups:
    g["eps"] = adam_eps
learner_fwd = torch.compile(learner) if compile else learner

# eval data -------------------------------------------------------------------
import glob

eval_sets = {}
for path in sorted(glob.glob(eval_files)):
    data = np.fromfile(path, dtype=np.uint8)
    n = eval_len - 1
    if len(data) < n + 1:
        continue
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(data) - n, size=eval_records)
    recs = np.stack([data[s : s + n] for s in starts]).astype(np.int64)
    eval_sets[os.path.splitext(os.path.basename(path))[0]] = torch.from_numpy(recs)
print(f"zero-shot eval sets: {list(eval_sets) or 'none (run data/prepare_eval.py)'}")


# helpers ---------------------------------------------------------------------
def seq_losses(model, inp, tgt, mask):
    """Per-sequence mean cross-entropy over content tokens. inp/tgt/mask: (B, T)."""
    logits = model(inp)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def micro_batches(out, out_len, budget):
    """Group programs by output length (short outputs are trimmed: attention is causal, so this is exact)."""
    n = len(out_len)
    order = np.argsort(-out_len, kind="stable")
    i = 0
    while i < n:
        L = int(out_len[order[i]]) if mask_padding else block_size
        if L == 0:
            break  # everything after this emitted nothing -> zero loss, zero reward
        T = min(block_size, 64 * math.ceil(L / 64))
        bs = max(1, budget // T)
        idx = order[i : i + bs]
        i += bs
        y = torch.from_numpy(out[idx, :T].astype(np.int64))
        inp = torch.cat([torch.full((len(idx), 1), PREFIX, dtype=torch.long), y[:, :-1]], 1)
        if mask_padding:
            mask = (torch.arange(T)[None, :] < torch.from_numpy(out_len[idx])[:, None].long()).float()
        else:
            mask = torch.ones(len(idx), T)
        inp, y = inp.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if TOK is not None:
            inp = torch.cat([inp[:, :1], TOK[inp[:, 1:]]], 1)
            y = TOK[y]
        yield idx, inp, y, mask.to(device, non_blocking=True)


FLOPS = {"learner": 0.0, "reward": 0.0, "gen_sample": 0.0, "gen_train": 0.0}


def counted_flops():
    return FLOPS["learner"] if flops_budget == "learner" else sum(FLOPS.values())


def add_flops(kind, model, n_seq, T, backward=True):
    FLOPS[kind] += n_seq * T * model.flops_per_token(T, backward)


def flat_params(model):
    return [p.detach().clone() for p in model.parameters()]


def adam_precond(opt, model):
    """P_e = lr / (sqrt(v_hat) + eps), the diagonal AdamW step operator (from the optimizer state)."""
    out = []
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            if "exp_avg_sq" not in st:
                out.append(torch.zeros_like(p))
                continue
            step = float(st["step"])
            vhat = st["exp_avg_sq"] / (1 - group["betas"][1] ** step)
            out.append(group["lr"] / (vhat.sqrt() + group["eps"]))
    # optimizer param order -> model param order
    order = {id(p): i for i, p in enumerate(p for g in opt.param_groups for p in g["params"])}
    return [out[order[id(p)]] for p in model.parameters()]


@torch.no_grad()
def compute_rewards(out, out_len, theta_past):
    """r_i = <grad_theta L(y_i; theta_e), P_e * (theta_past - theta_e)> via one JVP per micro-batch."""
    r = np.zeros(len(out_len), dtype=np.float64)
    if theta_past is None:
        return r
    names = [n for n, _ in learner.named_parameters()]
    params = {n: p.detach() for n, p in learner.named_parameters()}
    P = adam_precond(opt_l, learner)
    tangents = {n: (P[i] * (theta_past[i].to(device) - params[n])).to(params[n].dtype) for i, n in enumerate(names)}
    learner.set_attn_impl(reward_attn)
    for idx, inp, tgt, mask in micro_batches(out, out_len, reward_tokens_per_micro):
        add_flops("reward", learner, *inp.shape)  # JVP ~ primal + tangent ~ 3x forward ~ fwd+bwd
        with ctx:
            _, d = jvp(lambda p: seq_losses(lambda x: functional_call(learner, p, (x,)), inp, tgt, mask), (params,), (tangents,))
        r[idx] = d.float().cpu().double().numpy()
    learner.set_attn_impl("sdpa")
    return r


def verify_rewards(out, out_len, theta_past, r_jvp, k=4):
    """Debug: reward for the k longest outputs via autograd, <grad L(y_i), P * (theta_past - theta)>."""
    P = adam_precond(opt_l, learner)
    tang = [P[i] * (theta_past[i].to(device) - p.detach()) for i, p in enumerate(learner.parameters())]
    for idx, inp, tgt, mask in micro_batches(out, out_len, block_size):
        for j in range(min(k, len(idx))):
            learner.zero_grad(set_to_none=True)
            with ctx:
                seq_losses(learner, inp[j : j + 1], tgt[j : j + 1], mask[j : j + 1]).sum().backward()
            ref = sum((p.grad.float() * t.float()).sum() for p, t in zip(learner.parameters(), tang)).item()
            print(f"[check_reward] prog {idx[j]}: jvp {r_jvp[idx[j]]:+.6e} autograd {ref:+.6e}")
        break
    learner.zero_grad(set_to_none=True)


def learner_step(out, out_len, lr):
    for g in opt_l.param_groups:
        g["lr"] = lr
    M = len(out_len)
    total = 0.0
    for idx, inp, tgt, mask in micro_batches(out, out_len, tokens_per_micro):
        add_flops("learner", learner, *inp.shape)
        with ctx:
            losses = seq_losses(learner_fwd, inp, tgt, mask)
        loss = losses.sum() / M  # eq. 1: mean over the whole pool
        loss.backward()
        total += loss.item()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip)
    opt_l.step()
    opt_l.zero_grad(set_to_none=True)
    return total


def encode_programs(bodies):
    """Generator teacher-forcing tensors: input bytes, target alphabet indices, token mask, and log g0."""
    n = len(bodies)
    L = max_prog_len + 1
    inp = torch.full((n, L), M.PROG_PREFIX, dtype=torch.long)
    tgt = torch.full((n, L), END_IDX, dtype=torch.long)
    mask = torch.zeros(n, L)
    for i, b in enumerate(bodies):
        toks = [M.ALPHABET.index(c) for c in b] + [END_IDX]
        full = len(b) >= max_prog_len  # F is forced at max length -> not a policy decision
        k = len(toks) - (1 if full else 0)
        tgt[i, : len(toks)] = torch.tensor(toks)
        inp[i, 1 : len(toks)] = torch.tensor([M.ALPHABET_BYTES[t] for t in toks[:-1]], dtype=torch.long)
        mask[i, :k] = 1.0
    logg0 = -mask.sum(1) * LOG_A  # g0(x) = |A|^{-l(x)}
    return inp.to(device), tgt.to(device), mask.to(device), logg0.to(device)


def gen_logprob(inp, tgt, mask):
    with ctx:
        logits = generator(inp)[:, :, ALLOWED].float()
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None]).squeeze(-1)
    return (lp * mask).sum(1)


@torch.no_grad()
def sample_programs(n):
    """Autoregressively sample n program bodies from g_phi (restricted to the 19-token alphabet)."""
    seq = torch.full((n, 1), M.PROG_PREFIX, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    toks = []
    for t in range(max_prog_len + 1):
        add_flops("gen_sample", generator, n, seq.size(1), backward=False)  # no KV cache: full recompute
        with ctx:
            logits = generator(seq)[:, -1, ALLOWED].float()
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        if t == max_prog_len:
            nxt = torch.full_like(nxt, END_IDX)
        nxt = torch.where(done, torch.full_like(nxt, END_IDX), nxt)
        toks.append(nxt)
        done |= nxt == END_IDX
        seq = torch.cat([seq, ALLOWED[nxt][:, None]], 1)
        if done.all():
            break
    toks = torch.stack(toks, 1).cpu().numpy()
    bodies = []
    for row in toks:
        end = int(np.argmax(row == END_IDX))
        bodies.append("".join(M.ALPHABET[t] for t in row[:end]))
    return bodies


def sample_uniform(n):
    """i.i.d. tokens from the uniform prior g0 until F (the 'uniform' ablation / Solomonoff-prior baseline)."""
    bodies = []
    for _ in range(n):
        b = []
        while len(b) < max_prog_len:
            t = random.randrange(A)
            if t == END_IDX:
                break
            b.append(M.ALPHABET[t])
        bodies.append("".join(b))
    return bodies


# MAP-Elites archive + replay bank ------------------------------------------------
def niche(body, depth):
    L = len(body)
    return (min(int(depth), 8), 0 if L <= 8 else 1 if L <= 16 else 2 if L <= 32 else 3)


archive = {}  # niche -> {body: stored reward}


def archive_update(bodies, rewards, depths):
    for k in archive:
        for b in archive[k]:
            archive[k][b] *= archive_decay
    for b, r, d in zip(bodies, rewards, depths):
        if r <= 0:
            continue
        cell = archive.setdefault(niche(b, d), {})
        cell[b] = max(r, cell.get(b, 0.0))
        if len(cell) > archive_per_niche:
            del cell[min(cell, key=cell.get)]


def mutate(body):
    body = list(body)
    op = random.choice(("sub", "ins", "del")) if len(body) > 1 else random.choice(("sub", "ins"))
    if op == "ins" and len(body) >= max_prog_len:
        op = "sub"
    pos = random.randrange(len(body)) if body else 0
    if op == "sub" and body:
        body[pos] = random.choice(M.BODY_TOKENS)
    elif op == "del":
        del body[pos]
    else:
        body.insert(random.randrange(len(body) + 1), random.choice(M.BODY_TOKENS))
    return "".join(body)


def sample_mutations(n):
    cells = [c for c in archive.values() if c]
    if not cells:
        return []
    return [mutate(random.choice(list(random.choice(cells)))) for _ in range(n)]


bank_bodies, bank_logp, bank_ptr = [], [], 0


def bank_add(bodies, logps):
    global bank_ptr
    for b, lp in zip(bodies, logps):
        if len(bank_bodies) < bank_size:
            bank_bodies.append(b)
            bank_logp.append(lp)
        else:
            bank_bodies[bank_ptr] = b
            bank_logp[bank_ptr] = lp
            bank_ptr = (bank_ptr + 1) % bank_size


# eval ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate():
    learner.eval()
    res = {}
    for name, recs in eval_sets.items():
        tot, cnt = 0.0, 0
        for i in range(0, len(recs), 64):
            y = recs[i : i + 64].to(device)
            if TOK is not None:
                y = TOK[y]
            inp = torch.cat([torch.full((len(y), 1), PREFIX, device=device), y[:, :-1]], 1)
            with ctx:
                logits = learner(inp)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            tot += ce.item()
            cnt += y.numel()
        res[name] = tot / cnt / math.log(2)
    learner.train()
    return res


def save_ckpt(e):
    torch.save({
        "learner": learner.state_dict(), "generator": generator.state_dict(),
        "opt_l": opt_l.state_dict(), "opt_g": opt_g.state_dict(),
        "model_args": dict(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden),
        **({"nanochat_state": learner.model.state_dict(), "nanochat_config": learner.config_dict()} if learner_impl == "nanochat" else {}),
        "round": e, "config": config, "flops": counted_flops(), "flops_all": sum(FLOPS.values()), "flops_breakdown": dict(FLOPS),
    }, os.path.join(out_dir, "ckpt.pt"))


# main loop -------------------------------------------------------------------------
log_f = open(os.path.join(out_dir, "log.jsonl"), "a")
snapshots = {}  # round -> list of CPU param tensors
prev_params = None
tokens_seen = 0
t0 = time.time()
for e in range(max_rounds + 1):
    if e % eval_interval == 0 and eval_sets:
        ev = evaluate()
        print(f"round {e} | tokens {tokens_seen / 1e9:.3f}B | zero-shot bpb " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))
        log_f.write(json.dumps({"round": e, "tokens": tokens_seen, "eval": ev}) + "\n")
        log_f.flush()
    if e > 0 and (e % ckpt_interval == 0 or e == max_rounds):
        save_ckpt(e)
    over_budget = max_flops > 0 and counted_flops() >= max_flops
    if over_budget and not (e > 0 and e % ckpt_interval == 0):
        save_ckpt(e)
    if e == max_rounds or over_budget:
        break
    tr = time.time()

    # 1. pool ------------------------------------------------------------------------
    if reward_mode == "uniform":
        bodies = sample_uniform(pool_size)
        kind = np.zeros(pool_size, dtype=np.int8)
        old_logp = np.zeros(pool_size)
    else:
        n_rep = min(int(replay_frac * pool_size), len(bank_bodies))
        n_mut = int(mut_frac * (pool_size - n_rep)) if archive else 0
        n_fresh = pool_size - n_rep - n_mut
        fresh = sample_programs(n_fresh)
        muts = sample_mutations(n_mut)
        rep_idx = random.sample(range(len(bank_bodies)), n_rep) if n_rep else []
        bodies = fresh + muts + [bank_bodies[i] for i in rep_idx]
        kind = np.array([0] * len(fresh) + [1] * len(muts) + [2] * n_rep, dtype=np.int8)

    # 2. execute on U with fresh random tapes -------------------------------------------
    t_pool = time.time() - tr
    out, out_len, depth = M.run_programs(bodies, T=block_size, tape_len=tape_len, max_steps=max_steps,
                                          num_threads=exec_threads or None)
    t_exec = time.time() - tr - t_pool

    # 3. reward on theta_e (before the learner update) --------------------------------------
    if e % snap_every == 0:
        snapshots[e] = [p.cpu() for p in flat_params(learner)]
    if reward_mode == "uniform":
        rewards = np.zeros(pool_size)
    else:
        if reward_mode == "last_step":
            theta_past = prev_params
        else:
            past_keys = [k for k in snapshots if k <= e // 2]
            theta_past = snapshots[max(past_keys)] if e > 0 else None
            for k in [k for k in snapshots if k < max(past_keys)]:
                del snapshots[k]
        rewards = compute_rewards(out, out_len, theta_past)
        if check_reward and e == 2:
            verify_rewards(out, out_len, theta_past, rewards)
        if reward_mode != "signed":
            rewards = np.abs(rewards)
        if reward_mode == "shuffle":
            rewards = np.random.permutation(rewards)
    if reward_mode == "last_step":
        prev_params = [p.cpu() for p in flat_params(learner)]
    t_rew = time.time() - tr - t_pool - t_exec

    # 4. learner: one gradient step on all outputs ------------------------------------------------
    lr = learning_rate * min(1.0, (e + 1) / warmup_rounds)
    lloss = learner_step(out, out_len, lr)
    tokens_seen += int(out_len.sum()) if mask_padding else pool_size * block_size

    # 5. generator: policy gradient + expert iteration ------------------------------------------
    gstats = {}
    if reward_mode != "uniform":
        for g in opt_g.param_groups:
            g["lr"] = lr * gen_lr_ratio
        inp, tgt, gmask, logg0 = encode_programs(bodies)
        with torch.no_grad():
            logp_sample = gen_logprob(inp, tgt, gmask)
        if reward_mode != "uniform":
            fresh_logp = logp_sample[: len(fresh)].cpu().numpy()
            old_logp = np.concatenate([fresh_logp, np.zeros(len(muts)), np.array([bank_logp[i] for i in rep_idx])])
        r = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (r - r.mean()) / (r.std() + 1e-8) - kl_beta * (logp_sample - logg0)
        rho = torch.exp((logp_sample - torch.tensor(old_logp, device=device, dtype=torch.float32)).clamp(-20, 20))
        pg_mask = torch.tensor(kind != 1, device=device, dtype=torch.float32)  # mutations excluded from PG
        rpos = r.clamp(min=0)
        w = rpos / rpos.sum() if rpos.sum() > 0 else torch.zeros_like(rpos)
        logp = gen_logprob(inp, tgt, gmask)
        add_flops("gen_train", generator, *inp.shape, backward=False)  # the no-grad logprob pass
        add_flops("gen_train", generator, *inp.shape)
        loss_pg = -(pg_mask * rho * adv * logp).sum() / pg_mask.sum().clamp(min=1)
        loss_ei = -(w * logp).sum()
        (loss_pg + lambda_ei * loss_ei).backward()
        if gen_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), gen_grad_clip)
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        # bank stores non-replay programs with the probability they were sampled/admitted under
        nonrep = kind != 2
        bank_add([b for b, k in zip(bodies, nonrep) if k], logp_sample[torch.tensor(nonrep, device=device)].tolist())
        archive_update(bodies, rewards, depth)
        gstats = dict(pg=loss_pg.item(), ei=loss_ei.item(), kl=(logp_sample - logg0).mean().item(),
                      prog_len=float(np.mean([len(b) for b in bodies[: len(fresh)]] or [0])),
                      archive=sum(len(c) for c in archive.values()), bank=len(bank_bodies))

    dt = time.time() - tr
    if e % log_interval == 0:
        rec = dict(round=e, tokens=tokens_seen, flops=sum(FLOPS.values()), **{f"flops_{k}": v for k, v in FLOPS.items()}, lr=lr, loss=lloss, reward_mean=float(rewards.mean()),
                   reward_max=float(rewards.max()), out_len=float(out_len.mean()), depth=float(depth.mean()),
                   t_round=dt, t_pool=t_pool, t_exec=t_exec, t_reward=t_rew, **gstats)
        print(f"round {e} | {sum(FLOPS.values()):.2e} FLOPs | loss {lloss:.4f} | r {rec['reward_mean']:.3e} | emitted {rec['out_len']:.0f} "
              f"| proglen {gstats.get('prog_len', 0):.1f} | kl {gstats.get('kl', 0):.2f} "
              f"| {dt * 1000:.0f}ms (sample {t_pool * 1000:.0f}, exec {t_exec * 1000:.0f}, reward {t_rew * 1000:.0f})")
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
    if e % sample_interval == 0:
        top = np.argsort(-rewards)[:16]
        with open(os.path.join(out_dir, "programs.jsonl"), "a") as f:
            for i in top:
                f.write(json.dumps({"round": e, "reward": float(rewards[i]), "program": bodies[i],
                                    "output": out[i, : min(64, out_len[i])].tolist()}) + "\n")

print(f"done in {(time.time() - t0) / 3600:.2f}h, {tokens_seen / 1e9:.2f}B tokens, {sum(FLOPS.values()):.3e} FLOPs "
      + str({k: f"{v:.2e}" for k, v in FLOPS.items()}))

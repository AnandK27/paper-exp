"""
Byte-level nanoGPT training on natural data, from scratch or warm-started from a self-play learner
(the paper's appendix A.1 "pre-pretraining" experiment). Run both arms and compare val bits/byte:

    python train.py config/train_shakespeare_bytes.py --init_from=scratch  --out_dir=out-scratch
    python train.py config/train_shakespeare_bytes.py --init_from=out-selfplay-25m/ckpt.pt --out_dir=out-warm
    python compare.py out-scratch out-warm

Data: data/<dataset>/{train,val}.bin as raw uint8 bytes (see data/prepare_eval.py).
The model shape is taken from the self-play checkpoint when warm-starting, so keep n_layer/n_head/n_embd
equal across arms for a fair comparison. Learning rate / weight decay should be tuned per arm (paper does).
"""
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
out_dir = "out-train"
init_from = "scratch"  # 'scratch' or path to a selfplay ckpt.pt
dataset = "shakespeare_bytes"
eval_interval = 100
eval_iters = 50
log_interval = 10
batch_size = 32
block_size = 256
n_layer = 4
n_head = 4
n_embd = 128
mlp_hidden = 0
learning_rate = 1e-3
max_iters = 5000
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
total_budget = 0.0  # if > 0: max_flops = total_budget - FLOPs already spent in self-play (compute-matched arms)
max_flops = 0.0  # if > 0: size max_iters (and the LR schedule) to spend exactly this much training compute
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
compile = False
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurator.py")).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type != "cuda" or dtype == "float32" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
OUT_PREFIX = ord("O")  # the self-play learner always saw sequences prefixed by 'O'

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset)


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint8, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    y = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    x = torch.cat([torch.full((batch_size, 1), OUT_PREFIX, dtype=torch.long), y[:, :-1]], 1)
    return x.to(device), y.to(device)


if init_from == "scratch":
    model = GPT(GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden))
else:
    ck = torch.load(init_from, map_location="cpu")
    args = dict(ck["model_args"])
    print(f"warm start from self-play round {ck['round']}: {args}")
    for k in ("n_layer", "n_head", "n_embd", "mlp_hidden"):
        globals()[k] = args[k]
    args["block_size"] = max(block_size, args["block_size"])
    model = GPT(GPTConfig(**args))
    model.load_state_dict(ck["learner"])
model.to(device)
print(f"params: {model.get_num_params() / 1e6:.2f}M")
flops_offset = 0.0 if init_from == "scratch" else float(ck.get("flops", 0.0))  # self-play compute already spent
flops_per_iter = batch_size * block_size * model.flops_per_token(block_size)
if total_budget > 0:
    max_flops = total_budget - flops_offset
    assert max_flops > 0, f"self-play already spent {flops_offset:.3e} >= budget {total_budget:.3e}"
if max_flops > 0:
    max_iters = lr_decay_iters = max(1, int(max_flops // flops_per_iter))
    warmup_iters = min(warmup_iters, max_iters // 10)
    eval_interval = max(1, min(eval_interval, max_iters // 20))
print(f"{flops_per_iter:.3e} FLOPs/iter, {max_iters} iters, self-play offset {flops_offset:.3e}, "
      f"total {flops_offset + max_iters * flops_per_iter:.3e} FLOPs")
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
fwd = torch.compile(model) if compile else model


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_bpb():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = fwd(X)
            losses[k] = F.cross_entropy(logits.float().view(-1, 256), Y.view(-1)).item()
        out[split] = losses.mean().item() / math.log(2)
    model.train()
    return out


log_f = open(os.path.join(out_dir, "log.jsonl"), "w")
t0 = time.time()
for it in range(max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr
    if it % eval_interval == 0:
        bpb = estimate_bpb()
        tokens = it * batch_size * block_size
        print(f"step {it}: tokens {tokens / 1e6:.1f}M train bpb {bpb['train']:.4f}, val bpb {bpb['val']:.4f}")
        log_f.write(json.dumps({"iter": it, "tokens": tokens, "train_flops": it * flops_per_iter,
                                "total_flops": flops_offset + it * flops_per_iter, **bpb, "init_from": init_from}) + "\n")
        log_f.flush()
    if it == max_iters:
        break
    X, Y = get_batch("train")
    with ctx:
        logits = fwd(X)
    loss = F.cross_entropy(logits.float().view(-1, 256), Y.view(-1))
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if it % log_interval == 0:
        print(f"iter {it}: loss {loss.item() / math.log(2):.4f} bpb, {(time.time() - t0) * 1000 / (it + 1):.1f}ms/iter")



@torch.no_grad()
def full_val_bpb():
    """Deterministic bits/byte over every non-overlapping block of val.bin (for the final comparison)."""
    model.eval()
    data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint8).astype(np.int64)
    n = (len(data) // block_size) * block_size
    y_all = torch.from_numpy(data[:n]).view(-1, block_size)
    tot = 0.0
    for i in range(0, len(y_all), batch_size):
        y = y_all[i : i + batch_size].to(device)
        x = torch.cat([torch.full((len(y), 1), OUT_PREFIX, device=device), y[:, :-1]], 1)
        with ctx:
            logits = fwd(x)
        tot += F.cross_entropy(logits.float().view(-1, 256), y.reshape(-1), reduction="sum").item()
    return tot / n / math.log(2)


final = full_val_bpb()
total = flops_offset + max_iters * flops_per_iter
print(f"FINAL full val bpb {final:.4f} at total {total:.3e} FLOPs (self-play {flops_offset:.3e})")
log_f.write(json.dumps({"final_val_bpb": final, "total_flops": total, "selfplay_flops": flops_offset,
                        "init_from": init_from}) + "\n")
log_f.close()
torch.save({"model": model.state_dict(), "config": config}, os.path.join(out_dir, "ckpt.pt"))

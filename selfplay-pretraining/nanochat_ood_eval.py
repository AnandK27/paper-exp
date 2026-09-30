"""
Out-of-distribution bits/byte for nanochat checkpoints (scratch vs self-play warm starts).
Run with nanochat's venv:  NANOCHAT_BASE_DIR=~/.cache/nanochat ~/nanochat/.venv/bin/python nanochat_ood_eval.py [tag-glob]
Sets (all deterministic, seed 0):
  shakespeare  held-out tiny-shakespeare bytes (data/eval/text.bin)
  python       Python stdlib source (data/eval/python.bin, first 300 KB)
  arithmetic   "a+b=c" / "a-b=c" / "a*b=c" lines, operands 0..999
  sequences    comma-separated integer sequences: arithmetic/geometric progressions, Fibonacci-like,
               squares, cubes, triangular numbers, modular cycles
  copy         random lowercase strings repeated after " -> " (in-context copying)
  bf_outputs   raw outputs of uniform-prior Brainf*ck programs, fed as single-byte tokens (self-play's own domain)
bpb follows nanochat's evaluate_bpb: nats / (ln 2 * bytes), excluding zero-byte special-token targets.
"""
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/nanochat"))
from nanochat.checkpoint_manager import build_model, find_last_step  # noqa: E402
from nanochat.tokenizer import get_token_bytes, get_tokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 1024
device = torch.device("cuda")


def docs_arithmetic(rng, n=400):
    out = []
    for _ in range(n):
        lines = []
        for _ in range(40):
            a, b, op = rng.randint(0, 999), rng.randint(0, 999), rng.choice("+-*")
            lines.append(f"{a}{op}{b}={eval(f'{a}{op}{b}')}")
        out.append("\n".join(lines))
    return out


def docs_sequences(rng, n=1500):
    out = []
    for _ in range(n):
        k, L = rng.randrange(7), rng.randint(15, 40)
        a, d = rng.randint(0, 50), rng.randint(1, 20)
        if k == 0:
            s = [a + i * d for i in range(L)]
        elif k == 1:
            r = rng.choice([2, 3])
            s = [max(1, a) * r**i for i in range(min(L, 20))]
        elif k == 2:
            s = [a, d]
            while len(s) < L:
                s.append(s[-1] + s[-2])
        elif k == 3:
            s = [(a + i) ** 2 for i in range(L)]
        elif k == 4:
            s = [(a + i) ** 3 for i in range(L)]
        elif k == 5:
            s = [(a + i) * (a + i + 1) // 2 for i in range(L)]
        else:
            m = rng.randint(3, 12)
            s = [(a + i * d) % m for i in range(L)]
        out.append(", ".join(map(str, s)))
    return out


def docs_copy(rng, n=500):
    out = []
    for _ in range(n):
        lines = []
        for _ in range(20):
            w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(8, 32)))
            lines.append(f"{w} -> {w}")
        out.append("\n".join(lines))
    return out


def bf_byte_rows(n=512):
    sys.path.insert(0, HERE)
    import bf

    rng = random.Random(0)
    bodies = []
    while len(bodies) < 4 * n:
        b = []
        while len(b) < 64:
            t = rng.randrange(len(bf.ALPHABET))
            if bf.ALPHABET[t] == bf.END:
                break
            b.append(bf.ALPHABET[t])
        bodies.append("".join(b))
    out, out_len, _ = bf.run_programs(bodies, T=512, seeds=np.arange(1, len(bodies) + 1, dtype=np.uint64))
    return [out[i, : out_len[i]].tolist() for i in range(len(bodies)) if out_len[i] >= 64][:n]


def build_sets(tok):
    rng = random.Random(0)
    bos = tok.get_bos_token_id()
    enc = lambda docs: [bos] + [t for d in docs for t in tok.encode(d) + [bos]]
    raw = lambda path, n: open(os.path.join(HERE, path), "rb").read()[:n].decode("utf-8", errors="replace")
    chunk = lambda text, size=4000: [text[i : i + size] for i in range(0, len(text), size)]
    sets = {
        "shakespeare": enc(chunk(raw("data/eval/text.bin", 10**6))),
        "python": enc(chunk(raw("data/eval/python.bin", 300_000))),
        "arithmetic": enc(docs_arithmetic(rng)),
        "sequences": enc(docs_sequences(rng)),
        "copy": enc(docs_copy(rng)),
    }
    b2t = [tok.enc.encode_single_token(bytes([b])) for b in range(256)]
    sets["bf_outputs"] = [bos] + [t for row in bf_byte_rows() for t in [b2t[b] for b in row] + [bos]]
    return sets


@torch.no_grad()
def bpb(model, stream, token_bytes, bs=16):
    toks = torch.tensor(stream, dtype=torch.long)
    n = (len(toks) - 1) // SEQ
    x_all = toks[: n * SEQ].view(n, SEQ)
    y_all = toks[1 : n * SEQ + 1].view(n, SEQ)
    nats, nbytes = 0.0, 0
    for i in range(0, n, bs):
        x, y = x_all[i : i + bs].to(device), y_all[i : i + bs].to(device)
        loss = F.cross_entropy(model(x).float().view(-1, model.config.vocab_size), y.view(-1), reduction="none")
        nb = token_bytes[y.view(-1)]
        keep = nb > 0
        nats += loss[keep].sum().item()
        nbytes += nb[keep].sum().item()
    return nats / (math.log(2) * nbytes)


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "cm_*"
    base = os.path.join(os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")), "base_checkpoints")
    tok = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    sets = build_sets(tok)
    print("set sizes (tokens):", {k: len(v) for k, v in sets.items()}, file=sys.stderr)
    for d in sorted(glob.glob(os.path.join(base, pattern))):
        step = find_last_step(d)
        model, _, _ = build_model(d, step, device, phase="eval")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = {k: bpb(model, v, token_bytes) for k, v in sets.items()}
        print(json.dumps({"model": os.path.basename(d), "step": step, **res}), flush=True)
        del model
        torch.cuda.empty_cache()

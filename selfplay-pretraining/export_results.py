"""
Collect every compute-matched run into one JSON for plotting.
    python export_results.py > results.json
Series (budget, method): scratch comes from out-cm-<B>-lr*/f0.00, the paper-shape BF self-play from
out-cm-<B>-lr*/f0.xx, and the method sweep from out-m-<B>-<method>-lr*/f0.xx.
For each (series, fraction) we report every LR's mean/std over seeds and pick the LR with the lowest mean final.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

runs = defaultdict(list)  # (budget, method, frac, lr) -> list of run dicts
for path in glob.glob("out-cm-*-lr*/f*_s*/log.jsonl") + glob.glob("out-m-*-lr*/f*_s*/log.jsonl"):
    d = path.split("/")
    m = re.match(r"out-(cm|m)-([0-9e.]+)-(?:(\w+)-)?lr([0-9e.-]+)$", d[0])
    fm = re.match(r"f([0-9.]+)_s(\d+)$", d[1])
    if not m or not fm:
        continue
    kind, budget, method, lr = m.groups()
    frac = float(fm.group(1))
    if kind == "cm":
        method = "scratch" if frac == 0 else "bf_paper"
    rows = [json.loads(l) for l in open(path)]
    fin = next((r for r in rows if "final_val_bpb" in r), None)
    if fin is None:
        continue
    curve = [(r["total_flops"], r["val"]) for r in rows if "val" in r]
    rounds = None
    ck = os.path.join(d[0], f"sp_{d[1]}", "ckpt.pt")
    if os.path.exists(ck):
        rounds = int(torch.load(ck, map_location="cpu", weights_only=False)["round"])
    runs[(budget, method, frac, lr)].append(dict(final=fin["final_val_bpb"], best=min(v for _, v in curve),
                                                 sp_share=fin["selfplay_flops"] / fin["total_flops"],
                                                 total=fin["total_flops"], curve=curve, rounds=rounds))


def curve_mean(rs, n=40):
    total = rs[0]["total"]
    xs = np.linspace(0, total, n + 1)[1:]
    ys = []
    for x in xs:
        vals = [next((v for fl, v in r["curve"] if fl >= x), np.nan) for r in rs]
        vals = [v for v in vals if not np.isnan(v)]
        ys.append(float(np.mean(vals)) if vals else None)
    return [[float(x), y] for x, y in zip(xs, ys)]


out = []
groups = defaultdict(dict)
for (budget, method, frac, lr), rs in runs.items():
    groups[(budget, method, frac)][lr] = rs
for (budget, method, frac), by_lr in sorted(groups.items()):
    per_lr = {lr: dict(final_mean=float(np.mean([r["final"] for r in rs])), final_std=float(np.std([r["final"] for r in rs])),
                       best_mean=float(np.mean([r["best"] for r in rs])), seeds=len(rs)) for lr, rs in by_lr.items()}
    best_lr = min(per_lr, key=lambda lr: per_lr[lr]["final_mean"])
    rs = by_lr[best_lr]
    out.append(dict(budget=budget, method=method, frac=frac, best_lr=best_lr, per_lr=per_lr,
                    final_mean=per_lr[best_lr]["final_mean"], final_std=per_lr[best_lr]["final_std"],
                    best_mean=per_lr[best_lr]["best_mean"], seeds=len(rs),
                    sp_share=float(np.mean([r["sp_share"] for r in rs])),
                    rounds=[r["rounds"] for r in rs if r["rounds"] is not None],
                    curve=curve_mean(rs)))
print(json.dumps(out))

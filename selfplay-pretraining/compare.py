"""
Compare train.py runs at matched total compute (self-play FLOPs + nanoGPT training FLOPs).
    python compare.py out-cm/f*
Groups runs by the name before "_s<seed>", reports mean ± std of the exact final val bits/byte and
the val curve at fractions of the total budget (x-axis = total FLOPs including self-play).
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

groups = defaultdict(list)
for r in a.runs:
    path = os.path.join(r, "log.jsonl")
    if not os.path.isdir(r) or not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path)]
    fin = next((x for x in rows if "final_val_bpb" in x), None)
    if fin is None:
        print(f"skipping unfinished run {r}")
        continue
    curve = [(x["total_flops"], x["val"]) for x in rows if "val" in x]
    groups[re.sub(r"_s\d+$", "", os.path.basename(os.path.normpath(r)))].append((fin, curve))

fracs = (0.25, 0.5, 0.75)
print(f"{'arm':<10} {'seeds':>5} {'total FLOPs':>12} {'self-play':>10} {'final val bpb':>20} {'best val bpb':>13}  "
      + " ".join(f"{'@' + str(int(f * 100)) + '% budget':>13}" for f in fracs))
for name in sorted(groups):
    runs = groups[name]
    finals = np.array([f["final_val_bpb"] for f, _ in runs])
    total = runs[0][0]["total_flops"]
    sp = runs[0][0]["selfplay_flops"] / total * 100
    cells = []
    for f in fracs:
        vals = [next((v for fl, v in c if fl >= f * total), np.nan) for _, c in runs]
        # before the nanoGPT phase starts there is no natural-data val loss yet
        cells.append(f"{np.nanmean(vals):>13.4f}" if runs[0][0]["selfplay_flops"] <= f * total else f"{'(self-play)':>13}")
    std = f" ± {finals.std():.4f}" if len(finals) > 1 else ""
    best = np.mean([min(v for _, v in c) for _, c in runs])  # min over the (noisy, sampled) val curve
    print(f"{name:<10} {len(runs):>5} {total:>12.3e} {sp:>9.1f}% {finals.mean():>11.4f}{std:<9} {best:>13.4f}  " + " ".join(cells))

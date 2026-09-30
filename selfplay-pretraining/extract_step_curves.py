"""
Per-step training-loss curves from nanochat base_train logs (the "step 00037/01159 | loss: 2.519 ..." lines;
nanochat logs a debiased EMA of the training loss every step). Data is never repeated, so this tracks val loss.
    python extract_step_curves.py ~/nc-bytes ~/nc-cm > step_curves.json
Output: {"<dirname>/<run>": {"total": N, "every": k, "loss": [[step, loss_nats], ...]}} for every *.log that has
step lines, keeping every k-th step (k=5) plus the last. Self-play logs (sp_*) are skipped.
"""
import glob
import json
import os
import re
import sys

PAT = re.compile(r"^step (\d+)/(\d+) \([^)]*\) \| loss: ([0-9.]+)")
EVERY = 5
out = {}
for d in sys.argv[1:]:
    d = os.path.expanduser(d)
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        name = os.path.basename(f)[:-4]
        if name.startswith(("sp_", "ood", "driver")):
            continue
        pts, total = [], None
        with open(f, errors="replace") as fh:
            for line in fh:
                m = PAT.match(line)
                if m:
                    s, total = int(m.group(1)), int(m.group(2))
                    pts.append((s, float(m.group(3))))
        if not pts:
            continue
        pts = sorted(dict(pts).items())  # dedupe (a resumed log could repeat steps)
        keep = [[s, round(l, 5)] for i, (s, l) in enumerate(pts) if s % EVERY == 0 or i == len(pts) - 1]
        out[f"{os.path.basename(os.path.normpath(d))}/{name}"] = {"total": total, "every": EVERY, "loss": keep}
print(json.dumps(out))

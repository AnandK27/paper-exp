"""
Collect a nanochat sweep directory (nanochat_cm2.sh layout) into one JSON for the report.
    python nanochat_export.py ~/nc-bytes > nanochat_bytes_results.json
Names: scratch_s<seed>, <machine>_f<frac>_s<seed> (self-play warm start), control_f<frac>_s<seed>,
free_f<frac>_s<seed> (same self-play checkpoint + full nanochat budget, self-play not charged).
Also reads ood_all.jsonl (nanochat_ood_eval.py output; non-JSON lines are ignored).
"""
import glob
import json
import os
import re
import sys

d = os.path.expanduser(sys.argv[1])
ood = {}
if os.path.exists(os.path.join(d, "ood_all.jsonl")):
    for line in open(os.path.join(d, "ood_all.jsonl")):
        if line.startswith("{"):
            r = json.loads(line)
            ood[r["model"]] = {k: v for k, v in r.items() if k not in ("model", "step")}
runs = []
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    m = re.match(r"(scratch|control|[a-z]+)(?:_f([0-9.]+))?_s(\d+)$", name)
    if not m or name.startswith("ood"):
        continue
    L = [json.loads(l) for l in open(f)]
    fin = [r for r in L if "final_val_bpb" in r]
    if not fin:
        continue
    kind = m.group(1) if m.group(1) in ("scratch", "control", "free") else "selfplay"
    runs.append(dict(name=name, kind=kind, share=float(m.group(2) or 0), seed=int(m.group(3)),
                     final=fin[-1]["final_val_bpb"], sp_flops=fin[-1]["selfplay_flops"], total=fin[-1]["total_flops"],
                     iters=fin[-1]["num_iterations"], curve=[[r["total_flops"], r["val_bpb"]] for r in L if "val_bpb" in r],
                     steps=[[r["step"], r["val_bpb"]] for r in L if "val_bpb" in r],
                     ood=ood.get(name, {})))
print(json.dumps(runs))

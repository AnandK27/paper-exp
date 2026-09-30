#!/usr/bin/env bash
# nanochat compute-matched sweep, v2:
#   - byte-level tokenizer for nanochat too (NANOCHAT_BASE_DIR from nanochat_patch/make_byte_tokenizer.py),
#     so self-play and natural-data training share one token space
#   - self-play is charged only its learner-training FLOPs (--flops_budget=learner)
#   - matched controls: random init trained for exactly the warm arm's remaining budget
#   - at most MAXJOBS jobs at once (8 concurrent jobs exhausted RAM on the 1xH200 box before)
#   NANOCHAT_BASE_DIR=~/.cache/nanochat_bytes OUT=~/nc-bytes SEEDS="0 1" FRACS="0.05 0.1 0.2" ./nanochat_cm2.sh
set -euo pipefail
export DEPTH=${DEPTH:-8} MACHINE=${MACHINE:-bf} SP_LR=${SP_LR:-3e-4} SP_EXTRA=${SP_EXTRA:-}
export OUT=${OUT:-$HOME/nc-bytes} NC=${NC:-$HOME/nanochat}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_bytes} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
SEEDS=${SEEDS:-"0 1"}
FRACS=${FRACS:-"0.05 0.1 0.2"}
MAXJOBS=${MAXJOBS:-4}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

# budget = FLOPs of nanochat's default horizon for this depth, from the scratch seed-0 run (default ratio, no --target-flops)
if ! grep -q final_val_bpb "$OUT/scratch_s0.jsonl" 2>/dev/null; then
  rm -f "$OUT/scratch_s0.jsonl"
  (cd "$NC" && NANOCHAT_JSONL="$OUT/scratch_s0.jsonl" $NC/.venv/bin/python -m scripts.base_train --depth=$DEPTH \
     --eval-every=250 --core-metric-every=-1 --sample-every=-1 --run=dummy --seed=0 --model-tag=scratch_s0 \
     > "$OUT/scratch_s0.log" 2>&1)
fi
export B=$($NC/.venv/bin/python -c "import json; print(next(json.loads(l) for l in open('$OUT/scratch_s0.jsonl') if 'final_val_bpb' in l)['train_flops'])")
echo "budget B = $B FLOPs"

{
  for seed in $SEEDS; do [ "$seed" != 0 ] && echo "scratch $seed"; done
  for f in $FRACS; do for seed in $SEEDS; do echo "warm $seed $f"; echo "control $seed $f"; done; done
} | xargs -P "$MAXJOBS" -L 1 bash "$here/nanochat_job.sh"
echo ALL_DONE > "$OUT/ALL_DONE"

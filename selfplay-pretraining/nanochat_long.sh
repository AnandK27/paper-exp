#!/usr/bin/env bash
# Longer self-play on byte-level nanochat: does the self-play init's penalty vs random init cross zero?
# Warm arms at FRACS (learner-FLOP share, compute-matched) + matched random-init controls, and for each warm arm a
# "free" arm that reuses its self-play checkpoint with nanochat's full budget (self-play not charged).
# Reuses B from the finished scratch_s0 run in $OUT. Long self-play jobs are queued first.
#   FRACS="0.4 0.6" SEEDS="0 1" ./nanochat_long.sh
set -euo pipefail
export DEPTH=${DEPTH:-8} MACHINE=${MACHINE:-bf} SP_LR=${SP_LR:-3e-4} SP_EXTRA=${SP_EXTRA:-} FREE=1
export OUT=${OUT:-$HOME/nc-bytes} NC=${NC:-$HOME/nanochat}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_bytes} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
SEEDS=${SEEDS:-"0 1"}
FRACS=${FRACS:-"0.4 0.6"}
MAXJOBS=${MAXJOBS:-4}
here=$(cd "$(dirname "$0")" && pwd)
export B=$($NC/.venv/bin/python -c "import json; print(next(json.loads(l) for l in open('$OUT/scratch_s0.jsonl') if 'final_val_bpb' in l)['train_flops'])")
echo "budget B = $B FLOPs"
{
  for f in $(echo $FRACS | tr ' ' '\n' | sort -rn); do for seed in $SEEDS; do echo "warm $seed $f"; done; done
  for f in $FRACS; do for seed in $SEEDS; do echo "control $seed $f"; done; done
} | xargs -P "$MAXJOBS" -L 1 bash "$here/nanochat_job.sh"
echo ALL_DONE > "$OUT/ALL_DONE_long"

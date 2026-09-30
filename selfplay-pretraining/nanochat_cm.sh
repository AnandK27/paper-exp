#!/usr/bin/env bash
# Compute-matched warm-up test on karpathy/nanochat (the maintained successor of nanoGPT).
# Scratch = nanochat's own base_train recipe at --depth=$DEPTH. Each warm arm spends FRAC of the same FLOPs on
# self-play (paper method, nanochat GPT as the learner), then runs base_train --init-from with the remainder.
#   DEPTH=8 SEEDS="0 1" FRACS="0.05 0.1" ./nanochat_cm.sh
# Requires: ~/nanochat with nanochat_patch/apply_patch.py applied, tokenizer + data prepared.
set -euo pipefail
DEPTH=${DEPTH:-8}
SEEDS=${SEEDS:-"0 1"}
FRACS=${FRACS:-"0.05 0.1"}
MACHINE=${MACHINE:-bf}
SP_LR=${SP_LR:-3e-4}
SP_EXTRA=${SP_EXTRA:-}  # extra selfplay.py args, e.g. smaller micro-batches when the GPU is shared
OUT=${OUT:-$HOME/nc-cm}
NC=${NC:-$HOME/nanochat}
PY=$NC/.venv/bin/python
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
BT_ARGS="--depth=$DEPTH --eval-every=250 --core-metric-every=-1 --sample-every=-1 --run=dummy"
mkdir -p "$OUT"
here=$(cd "$(dirname "$0")" && pwd)

# budget = FLOPs of nanochat's default (data:param-ratio) horizon, taken from the scratch seed-0 run
# (pass B=... to skip this and start the warm arms while the seed-0 scratch run is still going)
if [ -z "${B:-}" ]; then
  if ! grep -q final_val_bpb "$OUT/scratch_s0.jsonl" 2>/dev/null; then
    (cd "$NC" && NANOCHAT_JSONL="$OUT/scratch_s0.jsonl" $PY -m scripts.base_train $BT_ARGS --seed=0 --model-tag=cm_scratch_s0 > "$OUT/scratch_s0.log" 2>&1)
  fi
  B=$($PY -c "import json; print(next(json.loads(l) for l in open('$OUT/scratch_s0.jsonl') if 'final_val_bpb' in l)['train_flops'])")
fi
echo "budget B = $B FLOPs"

run_seed() {
  local seed=$1
  if [ "$seed" != 0 ] && ! grep -q final_val_bpb "$OUT/scratch_s$seed.jsonl" 2>/dev/null; then
    (cd "$NC" && NANOCHAT_JSONL="$OUT/scratch_s$seed.jsonl" $PY -m scripts.base_train $BT_ARGS --seed=$seed \
       --target-flops=$B --model-tag=cm_scratch_s$seed > "$OUT/scratch_s$seed.log" 2>&1)
  fi
  for f in $FRACS; do
    local name="${MACHINE}_f${f}_s${seed}"
    grep -q final_val_bpb "$OUT/$name.jsonl" 2>/dev/null && continue
    local sp=$($PY -c "print($f*$B)")
    (cd "$here" && $PY selfplay.py config/selfplay_1m.py --learner_impl=nanochat --nanochat_depth=$DEPTH \
       --machine=$MACHINE --block_size=1024 --pool_size=128 --learning_rate=$SP_LR --exec_threads=4 --seed=$seed \
       --max_flops=$sp --max_rounds=100000000 --eval_interval=100000000 --ckpt_interval=100000000 \
       ${SP_EXTRA:-} --out_dir="$OUT/sp_$name" > "$OUT/sp_$name.log" 2>&1)
    local rest=$($PY -c "import torch; c=torch.load('$OUT/sp_$name/ckpt.pt', map_location='cpu', weights_only=False); print($B - c['flops'])")
    (cd "$NC" && NANOCHAT_JSONL="$OUT/$name.jsonl" $PY -m scripts.base_train $BT_ARGS --seed=$seed \
       --init-from="$OUT/sp_$name/ckpt.pt" --target-flops=$rest --model-tag=cm_$name > "$OUT/$name.log" 2>&1)
  done
}
for seed in $SEEDS; do run_seed "$seed" & done
wait
echo ALL_DONE > "$OUT/ALL_DONE"

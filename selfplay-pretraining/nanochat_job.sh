#!/usr/bin/env bash
# One job of the nanochat compute-matched sweep (called by nanochat_cm2.sh through a bounded xargs queue).
#   nanochat_job.sh scratch SEED        nanochat recipe from random init, budget B
#   nanochat_job.sh warm    SEED FRAC   self-play for FRAC*B counted FLOPs, then nanochat for the rest
#   nanochat_job.sh control SEED FRAC   random init, nanochat for (1-FRAC)*B (same steps as the warm arm)
#   (warm with FREE=1 also runs free_f<FRAC>_s<SEED>: the same checkpoint + nanochat for the full budget B, uncharged)
# Env: B, OUT, NC, DEPTH, MACHINE, SP_LR, SP_EXTRA, NANOCHAT_BASE_DIR (all exported by nanochat_cm2.sh)
set -euo pipefail
kind=$1 seed=$2 frac=${3:-0}
PY=$NC/.venv/bin/python
here=$(cd "$(dirname "$0")" && pwd)
BT_ARGS="--depth=$DEPTH --eval-every=250 --core-metric-every=-1 --sample-every=-1 --run=dummy --seed=$seed"
bt() {  # name, extra base_train args...
  local name=$1; shift
  grep -q final_val_bpb "$OUT/$name.jsonl" 2>/dev/null && { echo "skip $name"; return; }
  rm -f "$OUT/$name.jsonl"
  (cd "$NC" && NANOCHAT_JSONL="$OUT/$name.jsonl" $PY -m scripts.base_train $BT_ARGS --model-tag=$name "$@" > "$OUT/$name.log" 2>&1)
  echo "done $name: $(grep -o '"final_val_bpb": [0-9.]*' "$OUT/$name.jsonl")"
}
case $kind in
  scratch) bt "scratch_s$seed" --target-flops=$B ;;
  control) bt "control_f${frac}_s$seed" --target-flops=$($PY -c "print((1-$frac)*$B)") ;;
  warm)
    name="${MACHINE}_f${frac}_s$seed"
    grep -q final_val_bpb "$OUT/$name.jsonl" 2>/dev/null && { echo "skip $name"; exit 0; }
    if [ ! -f "$OUT/sp_$name/ckpt.pt" ]; then
      (cd "$here" && $PY selfplay.py config/selfplay_1m.py --learner_impl=nanochat --nanochat_depth=$DEPTH \
         --machine=$MACHINE --block_size=1024 --pool_size=128 --learning_rate=$SP_LR --exec_threads=4 --seed=$seed \
         --max_flops=$($PY -c "print($frac*$B)") --flops_budget=learner --max_rounds=100000000 \
         --eval_interval=100000000 --ckpt_interval=100000000 $SP_EXTRA --out_dir="$OUT/sp_$name" > "$OUT/sp_$name.log" 2>&1)
    fi
    rest=$($PY -c "import torch; c=torch.load('$OUT/sp_$name/ckpt.pt', map_location='cpu', weights_only=False); print($B - c['flops'])")
    bt "$name" --init-from="$OUT/sp_$name/ckpt.pt" --target-flops=$rest
    # FREE=1: also treat the self-play checkpoint as free (the paper's framing) and give nanochat the full budget B
    if [ "${FREE:-0}" = 1 ]; then
      bt "free_f${frac}_s$seed" --init-from="$OUT/sp_$name/ckpt.pt" --target-flops=$B
    fi ;;
esac

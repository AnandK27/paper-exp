#!/usr/bin/env bash
# Compute-matched test: does spending part of a fixed FLOP budget on self-play pre-pretraining
# beat spending all of it on ordinary byte-level nanoGPT training?
#
#   BUDGET=1e17 FRACS="0 0.05 0.1 0.25 0.5" SEEDS="0 1 2" DEVICE=cuda ./compute_matched.sh
#
# For each fraction f: self-play until f*BUDGET FLOPs, then nanoGPT on shakespeare bytes for
# (1-f)*BUDGET FLOPs from that checkpoint. f=0 is the from-scratch baseline. Same model shape everywhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
BUDGET=${BUDGET:-1e16}
FRACS=${FRACS:-"0 0.1 0.25 0.5"}
SEEDS=${SEEDS:-"0"}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-out-cm}
MODEL=${MODEL:-"--n_layer=4 --n_head=4 --n_embd=128"}    # ~1M params; 24.4M: --n_layer=8 --n_head=8 --n_embd=512 --mlp_hidden=1280
SP_ARGS=${SP_ARGS:-"--block_size=4096 --pool_size=1024"}  # self-play round shape (paper: 1024 x 4096)
TRAIN_ARGS=${TRAIN_ARGS:-""}                              # e.g. --learning_rate=3e-3 (tune per arm!)
WARMUP=${WARMUP:-selfplay}                                # selfplay (pick --machine=bf|stack in SP_ARGS) | pcfg

[ -f data/shakespeare_bytes/train.bin ] || $PY data/prepare_eval.py
mkdir -p "$OUT"
for seed in $SEEDS; do
  for f in $FRACS; do
    name=$(printf "f%.2f_s%s" "$f" "$seed")
    if [ -f "$OUT/$name/log.jsonl" ] && grep -q final_val_bpb "$OUT/$name/log.jsonl"; then
      echo "skip $name (done)"; continue
    fi
    sp_flops=$($PY -c "print($f*$BUDGET)")
    if $PY -c "import sys; sys.exit(0 if $f > 0 else 1)"; then
      echo "[$name] self-play for $sp_flops FLOPs"
      script=selfplay.py; [ "$WARMUP" = pcfg ] && script=pcfg_pretrain.py
      $PY $script config/selfplay_1m.py $MODEL $SP_ARGS --device=$DEVICE --seed=$seed \
          --max_flops=$sp_flops --max_rounds=100000000 --eval_interval=100000000 --ckpt_interval=100000000 \
          --out_dir="$OUT/sp_$name" > "$OUT/sp_$name.log" 2>&1
      init="$OUT/sp_$name/ckpt.pt"
    else
      init=scratch
    fi
    echo "[$name] nanoGPT for the rest of $BUDGET FLOPs from $init"
    $PY train.py config/train_shakespeare_bytes.py $MODEL $TRAIN_ARGS --device=$DEVICE --seed=$seed \
        --total_budget=$BUDGET --init_from=$init --out_dir="$OUT/$name" > "$OUT/$name.log" 2>&1
    grep FINAL "$OUT/$name.log"
  done
done
$PY compare.py "$OUT"/f*

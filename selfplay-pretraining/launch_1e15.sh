#!/usr/bin/env bash
# 1M model, 1e15 FLOP budget: 5 self-play fractions x 3 seeds x 2 nanoGPT LRs, 6 drivers in parallel on one GPU.
# Launch detached:  setsid nohup ./launch_1e15.sh > logs_launch.txt 2>&1 < /dev/null &
# Results:          .venv/bin/python compare.py out-cm-1e15-lr1e-3/f*  (and lr3e-3)
cd ~/nanogpt-selfplay
mkdir -p logs
for lr in 1e-3 3e-3; do
  for seed in 0 1 2; do
    PY=.venv/bin/python BUDGET=1e15 FRACS="0 0.05 0.1 0.25 0.5" SEEDS=$seed DEVICE=cuda \
      OUT=out-cm-1e15-lr$lr TRAIN_ARGS="--learning_rate=$lr --log_interval=1000" \
      nohup ./compute_matched.sh > logs/driver_lr${lr}_s${seed}.log 2>&1 &
  done
done
wait
echo ALL_DONE > logs/ALL_DONE

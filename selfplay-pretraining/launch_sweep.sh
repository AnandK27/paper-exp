#!/usr/bin/env bash
# Compute-matched sweep at one budget: 5 self-play fractions x 3 seeds x 2 nanoGPT LRs, 6 drivers in parallel.
#   setsid nohup ./launch_sweep.sh 1e14 [wait_for_file] > logs_launch_1e14.txt 2>&1 < /dev/null &
# If wait_for_file is given, block until it exists (queue behind another sweep).
BUDGET=${1:?budget}
WAIT_FOR=${2:-}
cd ~/nanogpt-selfplay
mkdir -p logs
if [ -n "$WAIT_FOR" ]; then
  until [ -f "$WAIT_FOR" ]; do sleep 60; done
fi
for lr in 1e-3 3e-3; do
  for seed in 0 1 2; do
    PY=.venv/bin/python BUDGET=$BUDGET FRACS="0 0.05 0.1 0.25 0.5" SEEDS=$seed DEVICE=cuda \
      OUT=out-cm-$BUDGET-lr$lr TRAIN_ARGS="--learning_rate=$lr --log_interval=1000" \
      nohup ./compute_matched.sh > logs/driver_${BUDGET}_lr${lr}_s${seed}.log 2>&1 &
  done
done
wait
echo ALL_DONE > logs/ALL_DONE_$BUDGET

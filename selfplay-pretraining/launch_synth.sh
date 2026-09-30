#!/usr/bin/env bash
# Compute-matched sweep over the hand-designed warm-ups in synth.py (scratch f=0 comes from launch_sweep.sh).
#   setsid nohup ./launch_synth.sh 1e14 [wait_for_file] > logs_launch_synth.txt 2>&1 < /dev/null &
BUDGET=${1:?budget}
WAIT_FOR=${2:-}
cd ~/nanogpt-selfplay
mkdir -p logs
if [ -n "$WAIT_FOR" ]; then
  until [ -f "$WAIT_FOR" ]; do sleep 60; done
fi
for task in dyck induction zipf mix; do
  for lr in 1e-3 3e-3; do
    for seed in 0 1 2; do
      PY=.venv/bin/python BUDGET=$BUDGET FRACS="0.05 0.1 0.25" SEEDS=$seed DEVICE=cuda WARMUP=pcfg \
        SP_ARGS="--block_size=1024 --task=$task" OUT=out-m-$BUDGET-$task-lr$lr \
        TRAIN_ARGS="--learning_rate=$lr --log_interval=1000" \
        nohup ./compute_matched.sh > logs/s_${BUDGET}_${task}_lr${lr}_s${seed}.log 2>&1 &
    done
  done
done
wait
echo ALL_DONE > logs/ALL_DONE_synth_$BUDGET

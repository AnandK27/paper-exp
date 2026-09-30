#!/usr/bin/env bash
# Compute-matched sweep across warm-up methods at one budget (scratch f=0 comes from launch_sweep.sh).
#   setsid nohup ./launch_methods.sh 1e14 > logs_launch_methods.txt 2>&1 < /dev/null &
# Methods: bf    = self-play on the paper's Brainf*ck machine (small rounds: 128 programs x 1024 bytes)
#          stack = self-play on the stack machine (stackm.c), same round shape
#          pcfg  = random-PCFG pretraining (paper appendix H), 1024-byte rows
BUDGET=${1:?budget}
cd ~/nanogpt-selfplay
mkdir -p logs
for method in bf stack pcfg; do
  if [ $method = pcfg ]; then warm=pcfg; sp="--block_size=1024"
  else warm=selfplay; sp="--block_size=1024 --pool_size=128 --machine=$method --exec_threads=2"; fi
  for lr in 1e-3 3e-3; do
    for seed in 0 1 2; do
      PY=.venv/bin/python BUDGET=$BUDGET FRACS="0.05 0.1 0.25 0.5" SEEDS=$seed DEVICE=cuda WARMUP=$warm \
        SP_ARGS="$sp" OUT=out-m-$BUDGET-$method-lr$lr TRAIN_ARGS="--learning_rate=$lr --log_interval=1000" \
        nohup ./compute_matched.sh > logs/m_${BUDGET}_${method}_lr${lr}_s${seed}.log 2>&1 &
    done
  done
done
wait
echo ALL_DONE > logs/ALL_DONE_methods_$BUDGET

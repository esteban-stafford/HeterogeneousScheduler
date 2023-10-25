#!/bin/bash

WORKLOAD="data/lublin_256.swf"
PLATFORM="data/cluster_x4_64procs.json"
EXP_NAME="test"
TRAIN_SEED=0
MODEL_PATH="data/logs/$EXP_NAME/${EXP_NAME}_s${TRAIN_SEED}"
SCORE_TYPE=0

python3.7 ppo-pick-jobs.py \
  --workload $WORKLOAD \
  --platform $PLATFORM \
  --gamma 1 \
  --seed $TRAIN_SEED \
  --trajs 10 \
  --epochs 1 \
  --exp_name $EXP_NAME \
  --pre_trained 0 \
  --trained_model $MODEL_PATH \
  --shuffle 0 \
  --backfil 0 \
  --score_type $SCORE_TYPE \
  --batch_job_slice 0

python3.7 compare-heterog.py \
  --workload $WORKLOAD \
  --platform $PLATFORM \
  --rlmodel $MODEL_PATH \
  --len 1024 \
  --seed 500 \
  --iter 20 \
  --shuffle 0 \
  --skip 0 \
  --score_type $SCORE_TYPE \
  --batch_job_slice 0
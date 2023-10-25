#!/bin/bash

python3.7 ppo-pick-jobs.py \
  --workload data/lublin_256.swf \
  --platform data/cluster_x4_64procs.json \
  --model data/lublin_256.schd \
  --gamma 1 \
  --seed 5 \
  --trajs 10 \
  --epochs 1 \
  --exp_name ppo \
  --pre_trained 0 \
  --trained_model data/logs/ppo_temp/ppo_temp_s0 \
  --shuffle 0 \
  --backfil 0 \
  --score_type 0 \
  --batch_job_slice 0


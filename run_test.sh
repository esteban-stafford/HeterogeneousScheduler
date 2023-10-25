#!/bin/bash

python3.7 compare-heterog.py \
  --workload data/lublin_256.swf \
  --platform data/cluster_x4_64procs.json \
  --rlmodel data/logs/ppo/ppo_s0 \
  --len 1024 \
  --seed 500 \
  --iter 20 \
  --shuffle 0 \
  --skip 0 \
  --score_type 0 \
  --batch_job_slice 0
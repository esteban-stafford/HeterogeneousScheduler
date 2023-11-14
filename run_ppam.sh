#!/bin/bash

PYTHON=python3
WORKLOAD="data/lublin_256.swf"
PLATFORM="data/cluster_x4_64procs.json"

models=( data/logs/Exp4_SquaredBSLD/Exp4_SquaredBSLD_s2406 data/logs/Exp5_SquaredSLD/Exp5_SquaredSLD_s2406 data/logs/Exp6_SquaredAVGW/Exp6_SquaredAVGW_s2406)

echo Comparing...
$PYTHON compare-heterog.py \
  --workload $WORKLOAD \
  --platform $PLATFORM \
  --rlmodel ${models[*]} \
  --len 1024 \
  --seed 500 \
  --iter 20 \
  --shuffle 0 \
  --skip 0 \
  --batch_job_slice 0 > data/logs/ppam_experiments.dat

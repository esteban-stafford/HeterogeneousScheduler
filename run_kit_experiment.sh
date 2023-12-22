#!/bin/bash

PYTHON=python3

traces="KIT-FH2-2016-1"
scores=(BSLD AVGW AVGT RESU SLD)

for i in 1 2 3 4 6 8 10; do
   data/generate_platform $i,{4,8,16,32,64},{2.5,3,3.5,4} > data/hetero_$i.json
   compare_platforms="$compare_platforms hetero_$i"
done

for trace in $traces; do
   WORKLOAD="data/$trace.swf"
   echo Comparing with $trace...
   for platform in $compare_platforms; do
      PLATFORM="data/$platform.json"
      $PYTHON compare-heterog.py \
        --workload $WORKLOAD \
        --platform $PLATFORM \
        --len 1024 \
        --seed 500 \
        --iter 20 \
        --shuffle 0 \
        --skip 0 \
        --batch_job_slice 0 > data/logs/compare_models:${trace}:${platform}.dat &
   done
   wait
done

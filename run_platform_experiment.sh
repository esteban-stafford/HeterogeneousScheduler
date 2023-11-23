#!/bin/bash

PYTHON=python3
WORKLOAD="data/PIK-IPLEX-2009-1.swf"
TRAIN_SEED=2406

scores=(BSLD AVGW AVGT RESU SLD)

data/generate_platform 20,64,1.26 > data/homo.json
data/generate_platform 2,64,{2.1,3,3.5,4} > data/hetero_freq.json
data/generate_platform 5,64,{0.54,1,1.5,2} > data/hetero_freq.json
data/generate_platform 1,{4,8,16,32,64},{2.5,3,3.5,4} > data/hetero.json

for platform in homo hetero; do
   for score in 0; do
      MODEL_PATH="data/logs/model_${platform}_${scores[$score]}/model_${platform}_${scores[$score]}_s${TRAIN_SEED}"
      models+=($MODEL_PATH)
      mkdir -p $MODEL_PATH/tf1_save
      echo Training score_type=${scores[$score]} with platform=$platform...

      $PYTHON ppo-pick-jobs.py \
        --workload $WORKLOAD \
        --platform data/$platform.json \
        --gamma 0.99 \
        --seed $TRAIN_SEED \
        --trajs 20 \
        --epochs 60 \
        --exp_name model_${scores[$score]} \
        --pre_trained 0 \
        --trained_model $MODEL_PATH \
        --shuffle 0 \
        --backfil 0 \
        --score_type $score \
        --batch_job_slice 0
   done
done

echo Comparing...
for platform in homo hetero_freq hetero_core hetero; do
   $PYTHON compare-heterog.py \
     --workload $WORKLOAD \
     --platform data/$platform.json \
     --rlmodel ${models[*]} \
     --len 1024 \
     --seed 500 \
     --iter 20 \
     --shuffle 0 \
     --skip 0 \
     --batch_job_slice 0 > data/logs/compare_models_$platform.dat &
done

wait

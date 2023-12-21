#!/bin/bash

PYTHON=python3
TRAIN_SEED=2406

traces="lublin_256"
train_platforms="hetero"
compare_platforms="hetero"
scores=(BSLD AVGW AVGT RESU SLD)

data/generate_platform 4,{4,8,16,32,64},{2.5,3,3.5,4} > data/hetero.json

for trace in $traces; do
   models=()
   WORKLOAD="data/$trace.swf"
   for platform in $train_platforms; do
      PLATFORM="data/$platform.json"
      for score in 0; do
         model=model:cl:${trace}:${platform}:${scores[$score]}
         MODEL_PATH="data/logs/${model}/${model}_s${TRAIN_SEED}"
         models+=($MODEL_PATH)
         mkdir -p $MODEL_PATH/tf1_save
         [ -e $MODEL_PATH/tf1_save/saved_model.pb ] && continue
         echo Training score_type=${scores[$score]} with platform=$platform and trace=$trace...

         epochs=2

         $PYTHON ppo-pick-jobs.py \
           --workload $WORKLOAD \
           --platform $PLATFORM \
           --gamma 0.99 \
           --seed $TRAIN_SEED \
           --trajs 1 \
           --epochs $epochs \
           --exp_name $model \
           --pre_trained 0 \
           --trained_model $MODEL_PATH \
           --shuffle 0 \
           --backfil 0 \
           --clustering_size 2 \
           --score_type $score \
           --batch_job_slice 0 
      done
   done
   echo Comparing with $trace...
   for platform in $compare_platforms; do
      PLATFORM="data/$platform.json"
      $PYTHON compare-heterog.py \
        --workload $WORKLOAD \
        --platform $PLATFORM \
        --rlmodel ${models[*]} \
        --len 16 \
        --seed 500 \
        --iter 1 \
        --shuffle 0 \
        --skip 0 \
        --clustering_size 2 \
        --batch_job_slice 0 \
        > data/logs/compare_models:cl:${trace}:${platform}.dat &
   done
   wait
done

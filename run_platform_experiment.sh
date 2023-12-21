#!/bin/bash

PYTHON=python3
TRAIN_SEED=2406

traces="lublin_256 lublin_1024 PIK-IPLEX-2009-1 CTC-SP2-1996-3.1-cln"
train_platforms="homo hetero"
compare_platforms="homo hetero_freq hetero_core hetero hetero_diag hetero_rand"
scores=(BSLD AVGW AVGT RESU SLD)

data/generate_platform 20,64,1.26 > data/homo.json
data/generate_platform 4,{4,8,16,32,64},3.25 > data/hetero_core.json
data/generate_platform 5,64,{0.54,1,1.5,2} > data/hetero_freq.json
data/generate_platform 1,{4,8,16,32,64},{2.5,3,3.5,4} > data/hetero.json
data/generate_platform $(echo 1,{4,32,16,8,64},{2.5,3,3.5,4} | tr ' ' '\n' | sort -R | tr '\n' ' ') > data/hetero_rand.json
data/generate_platform 4,{4\,4.35,8\,4,16\,3.7,32\,3.2,64\,3} > data/hetero_diag.json

for trace in $traces; do
   models=()
   WORKLOAD="data/$trace.swf"
   for platform in $train_platforms; do
      PLATFORM="data/$platform.json"
      for score in 0; do
         model=model:${trace}:${platform}:${scores[$score]}
         MODEL_PATH="data/logs/${model}/${model}_s${TRAIN_SEED}"
         models+=($MODEL_PATH)
         mkdir -p $MODEL_PATH/tf1_save
         [ -e $MODEL_PATH/tf1_save/saved_model.pb ] && continue
         echo Training score_type=${scores[$score]} with platform=$platform and trace=$trace...

         epochs=60
         #[ "$trace" == "lublin_256" ] && epochs=60

         $PYTHON ppo-pick-jobs.py \
           --workload $WORKLOAD \
           --platform $PLATFORM \
           --gamma 0.99 \
           --seed $TRAIN_SEED \
           --trajs 20 \
           --epochs $epochs \
           --exp_name $model \
           --pre_trained 0 \
           --trained_model $MODEL_PATH \
           --shuffle 0 \
           --backfil 0 \
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
        --len 1024 \
        --seed 500 \
        --iter 20 \
        --shuffle 0 \
        --skip 0 \
        --batch_job_slice 0 > data/logs/compare_models:${trace}:${platform}.dat &
   done
   wait
done

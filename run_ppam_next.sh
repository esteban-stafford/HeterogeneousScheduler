#!/bin/bash

PYTHON=python3
WORKLOAD="data/SDSC-BLUE-2000-4.2-cln.swf"
PLATFORM="data/cluster_x4_64procs.json"
TRAIN_SEED=2406

scores=(BSLD AVGW AVGT RESU SLD)

for score in 0 1 4; do
   MODEL_PATH="data/logs/model_${scores[$score]}/model_${scores[$score]}_s${TRAIN_SEED}"
   models+=($MODEL_PATH)
   [ -e $MODEL_PATH/tf1_save/saved_model.pb ] && continue
   echo Training score_type=${scores[$score]}...


   $PYTHON ppo-pick-jobs.py \
     --workload $WORKLOAD \
     --platform $PLATFORM \
     --gamma 0.99 \
     --seed $TRAIN_SEED \
     --trajs 20 \
     --epochs 100 \
     --exp_name model_${scores[$score]} \
     --pre_trained 0 \
     --trained_model $MODEL_PATH \
     --shuffle 0 \
     --backfil 0 \
     --score_type $score \
     --batch_job_slice 0
done

for workload in CTC-SP2-1996-3.1-cln.swf HPC2N-2002-2.2-cln.swf lublin_1024.swf \
                lublin-aaroh.swf RICC-2010-2.swf SDSC-BLUE-2000-4.2-cln.swf \
                SDSC-SP2-1998-4.2-cln.swf; do
echo Comparing with $workload...
$PYTHON compare-heterog.py \
  --workload data/$workload \
  --platform $PLATFORM \
  --rlmodel ${models[*]} \
  --len 1024 \
  --seed 500 \
  --iter 20 \
  --shuffle 0 \
  --skip 0 \
  --batch_job_slice 0 > data/logs/compare_models_$workload.dat &
done

wait

#!/bin/bash

PYTHON=python3
TRAIN_SEED=2406

traces="KIT-FH2-2016-1"
train_platforms="homo_x8 hetero_x8"
scores=(BSLD AVGW AVGT RESU SLD)

for fac in 8; do
   data/generate_platform $(( $fac * 8 )),64,3.1485 > data/homo_x$fac.json
   compare_platforms="$compare_platforms homo_x$fac"
   data/generate_platform $(( $fac * 4 )),{4,8,16,32,64},3.25 > data/hetero_core_x$fac.json
   compare_platforms="$compare_platforms hetero_core_x$fac"
   data/generate_platform $(( $fac * 2 )),64,{2.094,3,3.5,4} > data/hetero_freq_x$fac.json
   compare_platforms="$compare_platforms hetero_freq_x$fac"
   data/generate_platform $(( $fac * 1 )),{4,8,16,32,64},{2.5,3,3.5,4} > data/hetero_x$fac.json
   compare_platforms="$compare_platforms hetero_x$fac"
   data/generate_platform $(echo $(( $fac * 1 )),{4,32,16,8,64},{2.5,3,3.5,4} | tr ' ' '\n' | sort -R | tr '\n' ' ') > data/hetero_rand_x$fac.json
   compare_platforms="$compare_platforms hetero_rand_x$fac"
   data/generate_platform $(( $fac * 4 )),{4\,4.35,8\,4,16\,3.7,32\,3.2,64\,3} > data/hetero_diag_x$fac.json
   compare_platforms="$compare_platforms hetero_diag_x$fac"
done

for cluster in 4 ; do #8 16 32; do
   for trace in $traces; do
      models=()
      training=()
      WORKLOAD="data/$trace.swf"
      for platform in $train_platforms; do
         PLATFORM="data/$platform.json"
         for score in 0; do
            model=model:cl${cluster}:${trace}:${platform}:${scores[$score]}
            MODEL_PATH="data/logs/${model}/${model}_s${TRAIN_SEED}"
            models+=($MODEL_PATH)
            mkdir -p $MODEL_PATH/tf1_save
            [ -e $MODEL_PATH/tf1_save/saved_model.pb ] && echo Skipping training $MODEL_PATH/tf1_save/saved_model.pb && continue
            #echo Training cluster=$cluster score_type=${scores[$score]} with platform=$platform and trace=$trace...

            sed 's/^[[:blank:]]*//' <<-EOF >> train_job_$$.sh
               #!/bin/bash

               #SBATCH -J training
               #SBATCH -o %x_%j.out
               #SBATCH -N 1
               #SBATCH -n 10
               #SBATCH -t 12:00:00

               epochs=60
               export TF_NUM_INTEROP_THREADS=$SLURM_CPUS_PER_TASK
               export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK

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
                 --clustering_size $cluster \
                 --score_type $score \
                 --batch_job_slice 0 \
                 --max_queue_size 64 \
                 --job_sequence_size 512
               exit 0
EOF
            train_job=$(sbatch train_job_$$.sh)
            training+=(${train_job##* })
            rm -f train_job_$$.sh
         done
      done
      for platform in $compare_platforms; do
         PLATFORM="data/$platform.json"
         #echo Comparing with cluster=$cluster and trace=$trace...
         sed 's/^[[:blank:]]*//' <<-EOF >> compare_job_$$.sh
            #!/bin/bash

            #SBATCH -J comparing
            #SBATCH -o %x_%j.out
            #SBATCH -N 1
            #SBATCH -n 20
            #SBATCH -t 12:00:00

            export TF_NUM_INTEROP_THREADS=$SLURM_CPUS_PER_TASK
            export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK

            $PYTHON compare-heterog.py \
              --workload $WORKLOAD \
              --platform $PLATFORM \
              --rlmodel ${models[*]} \
              --len 2048 \
              --seed 500 \
              --iter 20 \
              --shuffle 0 \
              --skip 0 \
              --clustering_size $cluster \
              --batch_job_slice 0 \
              --max_queue_size 64 \
              > data/logs/compare_models:cl${cluster}:${trace}:${platform}.dat
            exit 0
EOF
         sbatch --dependency=afterok:$(IFS=:; echo "${training[*]}") compare_job_$$.sh
         rm -f compare_job_$$.sh
      done
   done
done

#!/bin/bash

MODEL_PREFIX=$1
MODEL_PATH=$2

#Run refcoco training
python tools/sweeps/sweep_refcoco_auto.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "refcoco_train" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_refcoco_train" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH

#Run ve training
python tools/sweeps/sweep_ve_auto.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 2 \
-n 1 \
--comment "ve_train" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_ve_train" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH

#Run nlvr2 training
python tools/sweeps/sweep_nlvr2_auto.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "nlvr2_train" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_nlvr2_train" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH

# #Run vqa training
python tools/sweeps/sweep_vqa_auto.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 4 \
-n 1 \
--comment "vqa_train" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_vqa_train" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH


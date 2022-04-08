#!/bin/bash

MODEL_PREFIX=$1
MODEL_PATH=$2

#eval zs0
python tools/sweeps/sweep_itm_eval.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "itm_test" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_itm_test_zs0" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH


#eval zs1
python tools/sweeps/sweep_itm_eval_test2.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "itm_test" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_itm_test_zs1" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH


#eval zs2
python tools/sweeps/sweep_itm_eval_test3.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "itm_test" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_itm_test_zs2" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH

#eval zs3
python tools/sweeps/sweep_itm_eval_test4.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "itm_test" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_itm_test_zs3" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH

#eval zs4
python tools/sweeps/sweep_itm_eval_test5.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "itm_test" \
--partition a100 \
-p "visualbert_"$MODEL_PREFIX"_vinvl_itm_test_zs4" \
--backend slurm \
--extra_args checkpoint.resume_file $MODEL_PATH
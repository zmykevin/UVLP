#!/bin/bash

# #Run vqa eval
python tools/sweeps/sweep_vqa_eval.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 2 \
-n 1 \
--comment "vqa_test" \
--partition a100 \
-p "visualbert_all_cc_OMVM_MRTM_vinvl_final_vqa_test" \
--backend slurm \
--extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_all_cc_OMVM_MRTM_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 


# python tools/sweeps/sweep_vqa_eval.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 2 \
# -n 1 \
# --comment "vqa_test" \
# --partition a100 \
# -p "visualbert_all_cc_OMVM_vinvl_final_vqa_test" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_all_cc_OMVM_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 

# python tools/sweeps/sweep_vqa_eval.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 2 \
# -n 1 \
# --comment "vqa_test" \
# --partition a100 \
# -p "visualbert_unpaired_0.8_vinvl_final_vqa_test" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_unpaired_0.8_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 

# python tools/sweeps/sweep_vqa_eval.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 2 \
# -n 1 \
# --comment "vqa_test" \
# --partition a100 \
# -p "visualbert_unpaired_vinvl_final_vqa_test" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_unpaired_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 

# python tools/sweeps/sweep_vqa_eval.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 2 \
# -n 1 \
# --comment "vqa_test" \
# --partition a100 \
# -p "visualbert_unpaired_pretrain_65000_vinvl_final_vqa_test" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_unpairedpretrain_65000_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 

# python tools/sweeps/sweep_vqa_eval.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 2 \
# -n 1 \
# --comment "vqa_test" \
# --partition a100 \
# -p "visualbert_round_and_robin_vinvl_final_vqa_test" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_round_and_robin_vinvl_vqa_train..ngpu4/models/model_28000.ckpt 


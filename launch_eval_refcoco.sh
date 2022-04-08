#Run refcoco training


python tools/sweeps/sweep_refcoco_test.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "refcoco_test" \
--partition a100 \
-p "visualbert_all_cc_OMVM_MRTM_vinvl_refcoco_testB" \
--backend slurm \
--extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_all_cc_OMVM_vinvl_refcoco_train..ngpu1/best.ckpt 

# python tools/sweeps/sweep_refcoco_test.py --resume_finished \
# --resume_failed \
# --checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
# -t 1 \
# -g 1 \
# -n 1 \
# --comment "refcoco_test" \
# --partition a100 \
# -p "visualbert_all_cc_licheng_rerun_vinvl_refcoco_testB" \
# --backend slurm \
# --extra_args checkpoint.resume_file /fsx/zmykevin/experiments/sweep_jobs/visualbert_all_cc_licheng_rerun_vinvl_refcoco_train..ngpu1/best.ckpt 
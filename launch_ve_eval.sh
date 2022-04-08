python tools/sweeps/sweep_ve_eval.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "visualbert_itm_filtering_bookcorpus_pretrain_vinvl_ve_test" \
--partition a100 \
-p visualbert_itm_filtering_bookcorpus_pretrain_vinvl_ve_test \
--backend slurm

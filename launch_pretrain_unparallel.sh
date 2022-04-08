python tools/sweeps/sweep_visualbert_pretrain_unparallel.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 4 \
-n 1 \
--comment "visual_bert_unparallel_pretrain_full" \
--partition a100 \
-p visual_bert_unparallel_pretrain_full \
--backend slurm
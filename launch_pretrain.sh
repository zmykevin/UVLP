python tools/sweeps/sweep_visualbert_pretrain.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 4 \
-n 1 \
--comment "pretraining" \
--partition a100 \
-p visual_bert_all_cc_vinvl_pretrain \
--backend slurm

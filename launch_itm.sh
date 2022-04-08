python tools/sweeps/sweep_itm.py --resume_finished \
--resume_failed \
--checkpoints_dir /home/zmykevin/fb_intern/exp/mmf_exp \
-t 1 \
-g 4 \
-n 1 \
--comment "itm_train" \
--partition a100 \
-p visualbert_vinvl_itm_train_debug_expand_train \
--backend slurm

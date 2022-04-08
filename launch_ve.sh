python tools/sweeps/sweep_ve.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 2 \
-n 1 \
--comment "ve_train" \
--partition a100 \
-p visualbert_region_tag_region_phrase_sentence_image_vinvl_ve_train \
--backend slurm

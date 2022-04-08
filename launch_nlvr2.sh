python tools/sweeps/sweep_nlvr2.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 1 \
-n 1 \
--comment "nlvr2_train" \
--partition a100 \
-p visualbert_region_tag_region_phrase_sentence_image_vinvl_nlvr2_train \
--backend slurm
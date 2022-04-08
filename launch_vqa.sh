python tools/sweeps/sweep_vqa.py --resume_finished \
--resume_failed \
--checkpoints_dir /fsx/zmykevin/experiments/sweep_jobs \
-t 1 \
-g 4 \
-n 1 \
--comment "vqa_train" \
--partition a100 \
-p visualbert_region_tag_region_phrase_sentence_image_vinvl_vqa_train \
--backend slurm

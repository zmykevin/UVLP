python tools/sweeps/sweep_visualbert_pretrain.py --resume_finished \
--resume_failed \
--checkpoints_dir /checkpoint/zmykevin/sweep_jobs \
-t 1 \
-g 4 \
-n 1 \
--comment "cvpr pretraining" \
--partition devlab \
-p visual_bert_sentence_image_tfidf_vinvl_pretrain_devlab \
--backend slurm

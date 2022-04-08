#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 mmf_run config=projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml model=visual_bert \
#                        dataset=itm_flickr30k run_type=test checkpoint.resume_file=/fsx/zmykevin/experiments/sweep_jobs/visual_bert_paired_pretrain..ngpu4/models/model_130000.ckpt \
#                        checkpoint.resume_pretrained=True training.batch_size=200 evaluation.predict=True \
#                        dataset_config.itm_flickr30k.annotations.test=flickr30k/defaults/annotations/flickr30k_itm_test_final_0.jsonl &> ../logs/eval_itm_test0.log &

CUDA_VISIBLE_DEVICES=0 mmf_run config=projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml model=visual_bert \
                       dataset=itm_flickr30k run_type=test checkpoint.resume_file=/fsx/zmykevin/experiments/sweep_jobs/visual_bert_paired_pretrain..ngpu4/models/model_130000.ckpt \
                       checkpoint.resume_pretrained=True training.batch_size=200 evaluation.predict=True \
                       dataset_config.itm_flickr30k.annotations.test=flickr30k/defaults/annotations/flickr30k_itm_test_final_1.jsonl&> ../logs/eval_itm_test1.log &


CUDA_VISIBLE_DEVICES=1 mmf_run config=projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml model=visual_bert \
                       dataset=itm_flickr30k run_type=test checkpoint.resume_file=/fsx/zmykevin/experiments/sweep_jobs/visual_bert_paired_pretrain..ngpu4/models/model_130000.ckpt \
                       checkpoint.resume_pretrained=True training.batch_size=200 evaluation.predict=True \
                       dataset_config.itm_flickr30k.annotations.test=flickr30k/defaults/annotations/flickr30k_itm_test_final_2.jsonl &> ../logs/eval_itm_test2.log &

# CUDA_VISIBLE_DEVICES=1 mmf_run config=projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml model=visual_bert \
#                        dataset=itm_flickr30k run_type=test checkpoint.resume_file=/fsx/zmykevin/experiments/sweep_jobs/visual_bert_paired_pretrain..ngpu4/models/model_130000.ckpt \
#                        checkpoint.resume_pretrained=True training.batch_size=200 evaluation.predict=True \
#                        dataset_config.itm_flickr30k.annotations.test=flickr30k/defaults/annotations/flickr30k_itm_test_final_3.jsonl &> ../logs/eval_itm_test3.log &  

CUDA_VISIBLE_DEVICES=2 mmf_run config=projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml model=visual_bert \
                       dataset=itm_flickr30k run_type=test checkpoint.resume_file=/fsx/zmykevin/experiments/sweep_jobs/visual_bert_paired_pretrain..ngpu4/models/model_130000.ckpt \
                       checkpoint.resume_pretrained=True training.batch_size=200 evaluation.predict=True \
                       dataset_config.itm_flickr30k.annotations.test=flickr30k/defaults/annotations/flickr30k_itm_test_final_4.jsonl &> ../logs/eval_itm_test4.log &  
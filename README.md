# UVLP
[Unsupervised Vision-and-Language Pre-training via Retrieval-based Multi-Granular Alignment](https://arxiv.org/abs/2203.00242)

[Mingyang Zhou*](https://github.com/zmykevin), [Licheng Yu*](https://lichengunc.github.io/),[Amanpreet Singh](https://apsdehal.in/), [Mengjiao Wang](https://scholar.google.co.uk/citations?user=98J-rNMAAAAJ&hl=en), [Zhou Yu](https://www.cs.columbia.edu/~zhouyu/), [Ning Zhang](https://n-zhang.github.io/) 

This is the official repository of UVLP,  a retrieval-based unsupervised vision and language pre-training framework. In this repository we provide code to support the end-to-end pre-training and finetuning for [NLVR2](https://lil.nlp.cornell.edu/nlvr/) and [RefCOCO+](https://github.com/lichengunc/refer) Task.

# Installation
To use the code, set up the conda virtual environment with the following command.
```
conda create -n mmf python=3.7
conda activate mmf
git clone https://github.com/zmykevin/UVLP.git
cd UVLP
pip install --editable .
```
Our code only supports Linux with NVIDIA GPUs. We test our code on Ubuntu 18.04 and A100 cards.

# Data and Pretrained Checkpoints

# Pre-training

# Downstream Task Fine-tuning
## NLVR2
1. Download data
TODO: Add the repository to download the data
2. Finetuning
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=projects/visual_bert/configs/nlvr2/vinvl_defaults.yaml \
model=visual_bert \ 
dataset=nlvr2 \
checkpoint.resume_pretrained=True \
checkpoint.resume_file=/PATH/TO/MODEL/best.ckpt \ 
env.save_dir=/PATH/TO/SAVE \
training.fp16=True training.batch_size=8
```

# Citation
If you find this code useful for your research, please consider citing: 
```
@inproceedings{zhou2022uvlp,
  author    = {Mingyang Zhou and
               Licheng Yu and
               Amanpreet Singh and
               Mengjiao Wang and
               Zhou Yu and
               Ning Zhang},
  title     = {Unsupervised Vision-and-Language Pre-training via Retrieval-based
               Multi-Granular Alignment},
  booktitle={CVPR}
  year= {2022},
}
```
# Acknowledge
Our code is developed on top of MMF. We thank the author and the collegues at Meta AI for their helpful discussion on code implementation. We also thank the anonymous reviewers for their constructive feedback. 

# Liscense
BSD

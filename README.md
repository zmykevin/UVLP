# UVLP
[Unsupervised Vision-and-Language Pre-training via Retrieval-based Multi-Granular Alignment](https://arxiv.org/abs/2203.00242)

[Mingyang Zhou*](https://github.com/zmykevin), [Licheng Yu*](https://lichengunc.github.io/),[Amanpreet Singh](https://apsdehal.in/), [Mengjiao Wang](https://scholar.google.co.uk/citations?user=98J-rNMAAAAJ&hl=en), [Zhou Yu](https://www.cs.columbia.edu/~zhouyu/), [Ning Zhang](https://n-zhang.github.io/) 

This is the official repository of our CVPR 2022 (Oral) Work UVLP,  a retrieval-based unsupervised vision and language pre-training framework. In this repository we provide code to support the end-to-end pre-training and finetuning for [NLVR2](https://lil.nlp.cornell.edu/nlvr/) and [RefCOCO+](https://github.com/lichengunc/refer) Task.

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
# Pre-trained Checkpoints and Dataset
Download the pre-trained checkpoints and Dataset from [here](https://drive.google.com/file/d/1zekXYjt-bnf0UQhZY3Q5xL2A6okUt7pV/view?usp=sharing).

The visual features for CC is too large to be uploaded to the cloud drive. You can generate the CC features for pre-traning with the following steps:
1. Download the CC feature from the [VinVL Repository](https://github.com/pzzhang/VinVL/blob/main/DOWNLOAD.md)
2. Change [these lines](https://github.com/zmykevin/UVLP/blob/35ab9955bc67e0f15b97c3ec07b05a7acf2229ea/data_preparation/convert_cc_vinvl.py#L275-L277) based on your saved feature path (dataset_cc.json is included in the downnloaded tar file). Run the following command to generate the CC Visual Features
```
python data_preparation/convert_cc_vinvl.py
```
The code will prepare the traininng visual features of CC for you. The Validation CC Image features is included in the downloaded tar file. 
# Pre-training
**Lauch Pretraining**

After you prepare the visual features, change the dataset directory in your pretraining config file accordingly based on your saved visual feature directory. you can launch pretraining with the following command:
```
mmf_run config=projects/visual_bert/configs/masked_conceptual_captions/pretrain.yaml \
run_type=train_val \
model=visual_bert \
dataset=masked_conceptual_captions,masked_conceptual_captions_image_tag,masked_conceptual_captions_image_phrase,itm_conceptual_captions \ env.save_dir=/PATH/TO/SAVE \
training.fp16=True 
```

# Downstream Task Fine-tuning
### NLVR2
**Download data**
Download the nlvr2 dataset from this [link](https://drive.google.com/file/d/1uzxFAxolDf8eLJYSaZIwllJmtmCjD4fE/view?usp=sharing).
Change the config file's dataset repository based on your save path for the downloaded data. 

**Finetuning**
```
mmf_run config=projects/visual_bert/configs/nlvr2/vinvl_defaults.yaml \
run_type=train_val_test \
model=visual_bert \ 
dataset=nlvr2 \
checkpoint.resume_pretrained=True \
checkpoint.resume_file=/PATH/TO/MODEL/best.ckpt \ 
env.save_dir=/PATH/TO/SAVE \
training.fp16=True 
```
### RefCOCO+
**Download data**
Download the refcoco+ dataset from this [link](https://drive.google.com/file/d/1oRLqVRTeE54b6T8xedbEiGcqKnOAH6Vm/view?usp=sharing).
Change the config file's dataset repository based on your save path for the downloaded data. 

**Finetuning**
```
mmf_run config=projects/visual_bert/configs/refcoco/vinvl_defaults.yaml \
run_type=train_val_test \
model=visual_bert \
dataset=refcoco \
checkpoint.resume_pretrained=True \
checkpoint.resume_file=/PATH/TO/MODEL/best.ckpt \ 
env.save_dir=/PATH/TO/SAVE \
training.fp16=True
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

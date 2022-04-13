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

# Data and Pretrained Checkpoints

# Pre-training

# Downstream Task Fine-tuning

# Citation
If you find this code useful for your research, please consider citing: 
```
@article{DBLP:journals/corr/abs-2203-00242,
  author    = {Mingyang Zhou and
               Licheng Yu and
               Amanpreet Singh and
               Mengjiao Wang and
               Zhou Yu and
               Ning Zhang},
  title     = {Unsupervised Vision-and-Language Pre-training via Retrieval-based
               Multi-Granular Alignment},
  journal   = {CoRR},
  volume    = {abs/2203.00242},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2203.00242},
  doi       = {10.48550/arXiv.2203.00242},
  eprinttype = {arXiv},
  eprint    = {2203.00242},
  timestamp = {Wed, 16 Mar 2022 16:39:52 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2203-00242.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# Acknowledge
Our code is developed on top of MMF. We thank the author and the collegues at Meta AI for their helpful discussion on code implementation. We also thank the anonymous reviewers for their constructive feedback. 

# Liscense
BSD

#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "train_val_test"),
        hyperparam("config", "projects/visual_bert/configs/nlvr2/vinvl_defaults.yaml"),
        hyperparam("dataset", "nlvr2"),
        hyperparam("model", "visual_bert"),
<<<<<<< HEAD
        hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/visual_bert_region_tag_region_phrase_sentence_image_vinvl_pretrain..ngpu4/models/model_130000.ckpt"),
=======
        hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/unpaired_pretrain..ngpu4/models/model_100000.ckpt"),
>>>>>>> 8205ab17 (minor update)
        hyperparam("checkpoint.resume_pretrained","True"),
        hyperparam("training.fp16", "True"),
        hyperparam("training.tensorboard", "True"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

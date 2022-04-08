#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "train_val_test"),
        hyperparam("config", "projects/visual_bert/configs/visual_entailment/vinvl_image_position_embedding_defaults.yaml"),
        hyperparam("dataset", "visual_entailment"),
        hyperparam("model", "visual_bert"),
        hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/unpaired_pretrain..ngpu4/models/model_100000.ckpt"),
        hyperparam("checkpoint.resume_pretrained","True"),
        hyperparam("training.fp16", "True"),
        hyperparam("training.tensorboard", "True"),
        hyperparam("training.checkpoint_interval", 500),
        hyperparam("training.evaluation_interval", 500),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

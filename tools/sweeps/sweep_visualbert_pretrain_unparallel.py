#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "train_val"),
        hyperparam("config", "projects/visual_bert/configs/masked_conceptual_captions/vinvl_unparallel_defaults.yaml"),
        hyperparam("dataset", "masked_conceptual_captions"),
        hyperparam("model", "visual_bert"),
        hyperparam("training.fp16", "True")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

#!/usr/bin/env python
  
# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "val"),
        hyperparam("config", "projects/visual_bert/configs/visual_entailment/vinvl_image_position_embedding_defaults.yaml"),
        hyperparam("dataset", "vqa2"),
        hyperparam("model", "visual_bert"),
        hyperparam("checkpoint.resume_file", "/fsx/zmykevin/experiments/sweep_jobs/visualbert_itm_filtering_bookcorpus_pretrain_vinvl_ve_train..ngpu2/best.ckpt"),
        hyperparam("evaluation.predict", "true"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

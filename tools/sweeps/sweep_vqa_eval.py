#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "test"),
        hyperparam("config", "projects/visual_bert/configs/vqa2/vinvl_image_position_embedding_defaults.yaml"),
        hyperparam("dataset", "vqa2"),
        hyperparam("model", "visual_bert"),
        #hyperparam("checkpoint.resume_file", "/fsx/zmykevin/experiments/sweep_jobs/visualbert_region_tag_sentence_image_vinvl_vqa_train..ngpu4/best.ckpt"),
        #hyperparam("checkpoint.resume_file", "/fsx/zmykevin/experiments/sweep_jobs/visualbert_unpaired_0.6_vinvl_vqa_train..ngpu4/models/model_28000.ckpt"),
        hyperparam("evaluation.predict", "true"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

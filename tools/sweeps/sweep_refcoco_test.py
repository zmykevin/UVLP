#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):
    return [
        hyperparam("run_type", "test"),
        hyperparam("config", "projects/visual_bert/configs/refcoco/vinvl_defaults.yaml"),
        hyperparam("dataset", "refcoco"),
        hyperparam("model", "visual_bert"),
        #hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/visual_bert_ucla_tag_bookcorpus_pretrain..ngpu4/models/model_177000.ckpt"),
        # hyperparam("checkpoint.resume_file","{}".format(args.extra_args)),
        # hyperparam("evaluation.predict", "True"),
        hyperparam("training.tensorboard", "True"),
        hyperparam("dataset_config.refcoco.annotations.test","refcoco/defaults/annotations/refcoco_plus_testB.jsonl")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

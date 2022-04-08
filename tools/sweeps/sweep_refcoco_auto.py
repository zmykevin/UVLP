#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):
    # resume_path=args.extra_args[0].split()[1]
    return [
        hyperparam("run_type", "train_val_test"),
        hyperparam("config", "projects/visual_bert/configs/refcoco/vinvl_defaults.yaml"),
        hyperparam("dataset", "refcoco"),
        hyperparam("model", "visual_bert"),
        #hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/visual_bert_ucla_tag_bookcorpus_pretrain..ngpu4/models/model_177000.ckpt"),
        # hyperparam("checkpoint.resume_file","{}".format(resume_path)),
        hyperparam("checkpoint.resume_pretrained","True"),
        hyperparam("training.fp16", "True"),
        # hyperparam("evaluation.predict", "True"),
        hyperparam("training.tensorboard", "True"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

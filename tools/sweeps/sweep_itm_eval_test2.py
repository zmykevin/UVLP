#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "test"),
        hyperparam("config", "projects/visual_bert/configs/flickr30k/vinvl_itm_defaults.yaml"),
        hyperparam("dataset", "itm_flickr30k"),
        hyperparam("model", "visual_bert"),
        #hyperparam("checkpoint.resume_file","/fsx/zmykevin/experiments/sweep_jobs/visual_bert_sentence_image_pretrain..ngpu4/models/model_130000.ckpt"),
        hyperparam("checkpoint.resume_pretrained","True"),
        # hyperparam("training.fp16", "True"),
        hyperparam("dataset_config.itm_flickr30k.annotations.test", "flickr30k/defaults/annotations/flickr30k_itm_test_final_1.jsonl"),
        hyperparam("training.batch_size", 200),
        hyperparam("evaluation.predict", "True"),
        hyperparam("training.tensorboard", "True"),
        # hyperparam("optimizer.params.lr", "2e-4")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

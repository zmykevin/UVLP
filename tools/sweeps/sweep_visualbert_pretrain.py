#/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

def get_grid(args):

    return [
        hyperparam("run_type", "train_val"),
        #hyperparam("config", "projects/visual_bert/configs/masked_conceptual_captions/all_bc.yaml"),
        hyperparam("config", "projects/visual_bert/configs/masked_conceptual_captions/tag_region_phrase_region_sentence_image.yaml"),
        #hyperparam("config", "projects/visual_bert/configs/masked_conceptual_captions/phrase_region.yaml"),
        #hyperparam("config", "projects/visual_bert/configs/masked_conceptual_captions/tag_region_phrase_region.yaml"),
        #hyperparam("dataset", "masked_conceptual_captions_image_tag,masked_conceptual_captions,itm_conceptual_captions"),
        hyperparam("dataset", "masked_conceptual_captions,masked_conceptual_captions_image_tag,masked_conceptual_captions_image_phrase,itm_conceptual_captions"),
        #hyperparam("dataset", "masked_conceptual_captions_image_phrase"),
        #hyperparam("dataset", "masked_conceptual_captions_image_phrase,masked_conceptual_captions_image_tag"),
        hyperparam("model", "visual_bert"),
        #hyperparam("checkpoint.resume", "True"),
        hyperparam("training.fp16", "True"),
        hyperparam("training.tensorboard", "True"),
        hyperparam("training.checkpoint_interval", 5000),
        hyperparam("training.evaluation_interval", 5000),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

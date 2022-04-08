# Copyright (c) Facebook, Inc. and its affiliates.

"""
Copy this file to sweep.py (same directory), then run the following command:
buck run faim/mmf:fblearner_sweep -- \
    --run-as-secure-group=<Your_Security_Group> \
    --checkpoints_dir=manifold://faim/tree/snie_mmf_hateful_memes_fg \
    --entitlement=<Entitlement> \
    --config=projects/hateful_memes/fine_grained/configs/visual_bert/hateful_pc_attack.yaml \
    -t=-1 -g=4 -n=1 \
    -p=<Your Prefix>
"""  # noqa

import tools.sweeps.lib as sweep
from tools.sweeps.lib import hyperparam


def get_grid(args):
    max_update = 10000
    return [
        hyperparam("run_type", "train_val"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("training.num_workers", 5),
        hyperparam("scheduler.params.num_training_steps", max_update),
        hyperparam("env.data_dir", "memcache_manifold://fair_mmf/tree/data"),
        hyperparam("dataset", "hateful_memes"),
        hyperparam(
            "dataset_config.hateful_memes.features.train[0]",
            "hateful_memes/defaults/features/raw_features/",
        ),
        hyperparam(
            "dataset_config.hateful_memes.features.val[0]",
            "hateful_memes/defaults/features/raw_features/",
        ),
        hyperparam(
            "dataset_config.hateful_memes.features.test[0]",
            "hateful_memes/defaults/features/raw_features/",
        ),
        hyperparam("model", "visual_bert", save_dir_key=lambda val: val),
        hyperparam(
            "training.max_updates", max_update, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", True),
        hyperparam("training.log_interval", 100),
        hyperparam("training.checkpoint_interval", 500),
        hyperparam("training.evaluation_interval", 500),
        hyperparam("training.tensorboard", True),
        hyperparam("training.find_unused_parameters", False),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

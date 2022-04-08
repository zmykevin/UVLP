# Copyright (c) Facebook, Inc. and its affiliates.
# isort:skip_file

import sys

sys.path.append("tools/sweeps/")  # noqa

import lib as sweep  # noqa
from lib import hyperparam  # noqa


def get_grid(args):
    max_update = 22000

    return [
        hyperparam("run_type", "train_val"),
        hyperparam("config", "projects/hateful_memes/configs/mmbt/defaults.yaml"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("training.num_workers", 5),
        hyperparam("env.data_dir", "memcache_manifold://fair_mmf/tree/data"),
        hyperparam("dataset", "hateful_memes"),
        hyperparam("model", "mmbt", save_dir_key=lambda val: val),
        hyperparam("training.batch_size", [32], save_dir_key=lambda val: f"bs{val}"),
        hyperparam("training.seed", [100], save_dir_key=lambda val: f"s{val}"),
        hyperparam("scheduler.type", ["warmup_cosine"]),
        hyperparam("scheduler.params.num_warmup_steps", 2000),
        hyperparam("scheduler.params.num_training_steps", max_update),
        hyperparam("optimizer.type", "adam_w", save_dir_key=lambda val: val),
        hyperparam("optimizer.params.lr", [1e-5], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("optimizer.params.eps", 1e-8),
        hyperparam(
            "training.max_updates", max_update, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", True),
        hyperparam("training.log_interval", 1000),
        hyperparam("training.checkpoint_interval", 1000),
        hyperparam("training.evaluation_interval", 1000),
        hyperparam("training.find_unused_parameters", True),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

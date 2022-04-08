# Copyright (c) Facebook, Inc. and its affiliates.
# isort:skip_file
# To run on everstore: change
# - config to projects/fb/mmbt/configs/political_ads/everstore.yaml
# - dataset to "everstore"
import sys

sys.path.append("tools/sweeps/")  # noqa

import lib as sweep  # noqa
from lib import hyperparam  # noqa


def get_grid(args):

    return [
        hyperparam("run_type", "train_val_test"),
        hyperparam("config", "projects/fb/mmbt/configs/political_ads/defaults.yaml"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("training.num_workers", 0),
        hyperparam("env.data_dir", "memcache_manifold://fair_mmf/tree/data"),
        hyperparam("env.cache_dir", "memcache_manifold://fair_mmf/tree/torch/mmf"),
        hyperparam("dataset", "hive"),
        hyperparam("model", "mmbt", save_dir_key=lambda val: val),
        hyperparam("training.batch_size", [256], save_dir_key=lambda val: f"bs{val}"),
        hyperparam("training.seed", [100], save_dir_key=lambda val: f"s{val}"),
        hyperparam("scheduler.type", ["warmup_cosine"]),
        hyperparam("scheduler.params.num_warmup_steps", 2000),
        hyperparam("optimizer.type", "adam_w", save_dir_key=lambda val: val),
        hyperparam("optimizer.params.lr", [1e-5], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("optimizer.params.eps", 1e-8),
        hyperparam(
            "training.max_updates", [300000], save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", False),
        hyperparam("training.log_interval", 1000),
        hyperparam("training.checkpoint_interval", 10000),
        hyperparam("training.evaluation_interval", 4000),
        hyperparam("training.find_unused_parameters", True),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

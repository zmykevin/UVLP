# Copyright (c) Facebook, Inc. and its affiliates.

# Modify values in this file to run sweeps on fblearner and checkpoint to manifold
# buck run faim/mmf:fblearner_sweep -- -p sweep_try -t -1 -g 2 -n 1 --checkpoints_dir manifold://your_manifold_bucket # noqa

import tools.sweeps.lib as sweep
from tools.sweeps.lib import hyperparam


def get_grid(args):

    return [
        hyperparam("run_type", "train_val_test"),
        hyperparam("config", "projects/fb/mmbt/configs/political_ads/defaults.yaml"),
        hyperparam("training.fp16", [True], save_dir_key=lambda val: f"fp16{val}"),
        hyperparam("training.num_workers", 0, save_dir_key=lambda val: f"nw{val}"),
        hyperparam("dataset", "hive"),
        hyperparam("model", "mmbt", save_dir_key=lambda val: val),
        hyperparam("training.batch_size", [64], save_dir_key=lambda val: f"bs{val}"),
        hyperparam("training.seed", [100], save_dir_key=lambda val: f"s{val}"),
        hyperparam("scheduler.type", ["warmup_cosine"]),
        hyperparam("scheduler.params.num_warmup_steps", 2000),
        hyperparam("optimizer.type", "adam_w", save_dir_key=lambda val: val),
        hyperparam("optimizer.params.lr", [1e-5], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("optimizer.params.eps", 1e-8),
        hyperparam(
            "training.max_updates", [22000], save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", False),
        hyperparam("training.log_interval", 1000),
        hyperparam("training.checkpoint_interval", 4000),
        hyperparam("training.evaluation_interval", 4000),
        hyperparam("training.find_unused_parameters", True),
        hyperparam("training.tensorboard", True),  # we turn on tensorboard on default
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

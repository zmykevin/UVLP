# Copyright (c) Facebook, Inc. and its affiliates.
# isort:skip_file
import argparse
from typing import Optional, List
# @fb-only: import torch.fb.rendezvous.zeus  # noqa

from mmf.utils.configuration import Configuration
from mmf.utils.distributed import infer_init_method
from mmf.utils.env import setup_imports
from mmf.utils.flags import flags
from mmf.utils.fb.distributed import (
    get_fb_worker_world_size,
    get_fb_worker_local_rank,
    get_fb_worker_local_world_size,
    get_fb_worker_group_rank,
)
from mmf.utils.fb.env import setup_fb_worker_env
from mmf_cli.run import distributed_main, main, run


def get_fb_training_parser():
    parser = flags.get_parser()
    # [FB] Additional FB specific cmd args
    return parser


def fb_run(device_id, configuration, start_rank, log_path=None):
    """[FB] entry point for each worker process."""
    config = configuration.get_config()
    config.distributed.rank = start_rank + device_id

    if config.distributed.world_size > 1:
        distributed_main(device_id, configuration)
    else:
        config.device_id = 0
        main(configuration)


def fb_scheduler_run(opts: Optional[List[str]] = None, predict: bool = False):
    """[FB-SCHEDULER] Entry point for each worker process deployed via torchX
    We need this fb_scheduler_run fn because torchx already initializes n processes
    per node and sets up the env per node. The MMF framework doesn't need to
    spawn processes as it would otherwise do in the mmf_cli/run.py file
    """
    setup_fb_worker_env()
    setup_imports()

    group_rank = get_fb_worker_group_rank()
    local_rank = get_fb_worker_local_rank()
    local_world_size = get_fb_worker_local_world_size()
    start_rank = local_world_size * group_rank

    if opts is None:
        parser = get_fb_training_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    configuration = Configuration(args)
    # Do set runtime args which can be changed by MMF
    configuration.args = args
    config = configuration.get_config()
    if config.distributed.init_method is None:
        infer_init_method(config)
    config.distributed.world_size = get_fb_worker_world_size()
    config.start_rank = start_rank

    fb_run(local_rank, configuration, start_rank)


if __name__ == "__main__":
    run()

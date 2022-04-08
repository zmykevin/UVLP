# Copyright (c) Facebook, Inc. and its affiliates.
import os

from mmf.utils.distributed import is_dist_initialized, get_rank, get_world_size


def get_fb_worker_world_size():
    """[FB-SCHEDULER] get the rank of the worker. Here the environment for
    each MAST/FLOW worker is already initialized with the appropriate rank.
    """
    # If mmf was launched using torchx + dist.ddp
    # the environment on each process is already setup
    # and the 'WORLD_SIZE' env variable should be populated
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    # If mmf is launched without torchx + dist.ddp or through
    # some other launcher, it'll expect that torch.distributed
    # is initialized and that will be responsible to set
    # up the environment
    elif is_dist_initialized():
        world_size = get_world_size()
    else:
        world_size = 1
    return world_size


def get_fb_worker_rank():
    """[FB-SCHEDULER] get the rank of the worker. Here the environment for
    each MAST/FLOW worker is already initialized with the appropriate rank.
    """
    # If mmf was launched using torchx + dist.ddp
    # the environment on each process is already setup
    # and the 'RANK' env variable should be populated
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    # If mmf is launched without torchx + dist.ddp or through
    # some other launcher, it'll expect that torch.distributed
    # is initialized and that will be responsible to set
    # up the environment
    elif is_dist_initialized():
        rank = get_rank()
    else:
        rank = 0
    return rank


def get_fb_worker_group_rank():
    """[FB-SCHEDULER] get the group rank of the worker. Here the environment for
    each MAST/FLOW worker is already initialized with the appropriate rank.
    """
    return int(os.getenv("GROUP_RANK", 0))


def get_fb_worker_local_world_size():
    """[FB-SCHEDULER] get the local world size of the worker. Here the environment for
    each MAST/FLOW worker is already initialized with the appropriate rank.
    """
    return int(os.getenv("LOCAL_WORLD_SIZE", 1))


def get_fb_worker_local_rank():
    """[FB-SCHEDULER] get the local rank of the worker. Here the environment for
    each MAST/FLOW worker is already initialized with the appropriate rank.
    """
    return int(os.getenv("LOCAL_RANK", 0))

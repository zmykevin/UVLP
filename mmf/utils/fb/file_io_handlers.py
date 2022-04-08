# Copyright (c) Facebook, Inc. and its affiliates.

import functools


MAX_PARALLEL = 16
TIMEOUT_IN_SECONDS = 1800


@functools.lru_cache()
def register_handlers(ioPathManager):
    from iopath.fb.manifold import ManifoldPathHandler

    ioPathManager.register_handler(
        ManifoldPathHandler(max_parallel=MAX_PARALLEL, timeout_sec=TIMEOUT_IN_SECONDS)
    )

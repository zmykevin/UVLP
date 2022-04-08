# Copyright (c) Facebook, Inc. and its affiliates.
import os

from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from tensorboard.fb.manifoldio import ManifoldFileSystem


def setup_fb_worker_env():
    """"[FB-SCHEDULER] FB Specific env setup"""
    # Register the manifold file system for jobs launched via torchx
    # This enables writers/loggers to write to manifold:// path
    register_filesystem("manifold", ManifoldFileSystem())
    # Setup the cache dir default env variable. Without this option
    # the default configuration will try to create the cache dir
    # under the project root which is read-only on MAST workers
    if "MMF_CACHE_DIR" not in os.environ:
        os.environ["MMF_CACHE_DIR"] = "/tmp"

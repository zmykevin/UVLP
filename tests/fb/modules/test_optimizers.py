# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

import torch
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleModel, skip_if_non_fb


class TestOptimizerStateSharding(unittest.TestCase):
    def setUp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.GLOO, rank=0, world_size=1
        )
        self.config = OmegaConf.create(
            {
                "optimizer": {
                    "type": "adam_w",
                    "enable_state_sharding": True,
                    "params": {"lr": 5e-5},
                },
                "training": {
                    "fp16": False,
                },
            }
        )

    def tearDown(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @skip_if_non_fb
    def test_build_sharded_optimizer(self):
        from fairscale.optim.oss import OSS

        model = SimpleModel({"in_dim": 1})
        model.build()

        optimizer = build_optimizer(model, self.config)
        self.assertTrue(isinstance(optimizer, OSS))
        self.assertEqual(len(optimizer.param_groups), 1)

# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

import tests.test_utils as test_utils
from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root
from tests.test_utils import dummy_args, skip_if_non_fb


class TestMMFTFairSeqTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
        model_name = "mmf_transformer"
        config_path = os.path.join(
            get_mmf_root(), "configs/fb/models/mmf_transformer/fairseq.yaml"
        )
        args = dummy_args(model=model_name)
        args.opts.append(f"config={config_path}")
        configuration = Configuration(args)
        config = configuration.get_config()
        model_class = registry.get_model_class(model_name)
        self.finetune_model = model_class(config.model_config[model_name])
        self.finetune_model.build()

    @skip_if_non_fb
    def test_load_save_finetune_model(self):
        self.assertTrue(test_utils.verify_torchscript_models(self.finetune_model))

    @skip_if_non_fb
    def test_finetune_model(self):
        model = self.finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(model, vocab_size=150000)
        )

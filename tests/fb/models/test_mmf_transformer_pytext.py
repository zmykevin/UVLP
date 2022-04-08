# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root
from tests.test_utils import dummy_args, skip_if_non_fb


XLM_ROBERTA_VOCAB_SIZE = 250002


class TestMMFTPytextTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        self.model_name = "mmf_transformer"
        config_path = os.path.join(
            get_mmf_root(), "configs/fb/models/mmf_transformer/pytext.yaml"
        )
        args = dummy_args(model=self.model_name)
        args.opts.append(f"config={config_path}")
        configuration = Configuration(args)
        self.config = configuration.get_config()
        self.model_class = registry.get_model_class(self.model_name)

    @skip_if_non_fb
    def test_load_save_finetune_model(self):
        finetune_model = self.model_class(self.config.model_config[self.model_name])
        finetune_model.build()
        self.assertTrue(test_utils.verify_torchscript_models(finetune_model))

    @skip_if_non_fb
    def test_finetune_model(self):
        finetune_model = self.model_class(self.config.model_config[self.model_name])
        finetune_model.build()
        model = finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=XLM_ROBERTA_VOCAB_SIZE
            )
        )

    @skip_if_non_fb
    def test_load_old_checkpoint(self):
        from iopath.fb.fb_file_io import g_pathmgr

        model_class = registry.get_model_class(self.model_name)
        config = self.config.model_config[self.model_name]

        config.heads[0].num_labels = 9048
        config.modalities[1].encoder.type = "identity"
        config.modalities[1].embedding_dim = 256
        finetune_model = model_class(config)
        finetune_model.build()
        checkpoint_path = "manifold://fair_mmf/tree/test_data/mmft_best.ckpt"

        with g_pathmgr.open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device("cpu"))
        incompatible_keys = finetune_model.load_state_dict(
            checkpoint["model"], strict=True
        )
        self.assertEqual(len(incompatible_keys.missing_keys), 0)

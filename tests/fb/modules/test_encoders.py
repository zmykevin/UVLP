# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from omegaconf import OmegaConf
from tests.test_utils import (
    skip_if_non_fb,
    skip_if_no_network,
    skip_if_no_cuda,
    setup_proxy
)
from mmf.modules.fb.clip_encoders import (
    ClipImageEncoder,
    ClipTextEncoder
)


class TestModuleEncoders(unittest.TestCase):
    def setUp(self):
        setup_proxy()

    # Clip Image Encoder Tests
    @skip_if_non_fb
    @skip_if_no_cuda
    @skip_if_no_network
    def test_clip_image_encoder_pretrained(self):
        torch.manual_seed(2)
        params_config = OmegaConf.create(
            {
                "pretrained_model_name": "RN50x4",
                "is_pretrained": True,
            }
        )
        encoder = ClipImageEncoder(params_config)
        encoded = encoder(torch.randn((8, 3, 288, 288)))
        self.assertAlmostEqual(encoded.mean().item(), 0.0063, 4)

    @skip_if_non_fb
    @skip_if_no_cuda
    @skip_if_no_network
    def test_clip_image_encoder_default(self):
        # ViT-B/32
        torch.manual_seed(2)
        encoder = ClipImageEncoder(OmegaConf.create({
            "is_pretrained": True
        }))
        encoded = encoder(torch.randn((8, 3, 224, 224)))
        self.assertAlmostEqual(encoded.mean().item(), -0.0185, 4)

    @skip_if_non_fb
    def test_clip_image_encoder_random_init(self):
        # RN50
        torch.manual_seed(2)
        params_config = OmegaConf.create(
            {
                "backbone": "modified_resnet",
                "embed_dim": 1024,
                "vision_layers": [3, 4, 6, 3],
                "vision_width": 64,
                "vision_patch_size": None,
                "image_resolution": 224,
                "vision_heads": 32,
                "is_pretrained": False,
            }
        )
        encoder = ClipImageEncoder(params_config)
        encoded = encoder(torch.randn((8, 3, 224, 224)))
        self.assertAlmostEqual(encoded.mean().item(), -0.006693754345178604, 2)

    @skip_if_non_fb
    @skip_if_no_cuda
    def test_clip_image_encoder_manifold(self):
        # RN101
        torch.manual_seed(2)
        m = (
            "manifold://fair_mmf/tree/torch/mmf/model/"
            + "clip/encoders/clip_image_encoder_RN101.pt"
        )
        params_config = OmegaConf.create({
            "pretrained_model_name": "RN101",
            "is_pretrained": True,
            "pretrained_model": {
                "url": m
            }
        })
        encoder = ClipImageEncoder(params_config)
        encoded = encoder(torch.randn((8, 3, 224, 224)))
        self.assertAlmostEqual(encoded.mean().item() * 1E5, 9.58, 2)

    # Clip Text Encoder Tests
    @skip_if_non_fb
    @skip_if_no_cuda
    @skip_if_no_network
    def test_clip_text_encoder_pretrained(self):
        torch.manual_seed(2)
        params_config = OmegaConf.create(
            {
                "pretrained_model_name": "RN50x4",
                "is_pretrained": True,
            }
        )
        encoder = ClipTextEncoder(params_config)
        encoded = encoder(torch.randint(0, 45000, (8, 77)))
        self.assertAlmostEqual(encoded.mean().item(), 0.01, 2)

    @skip_if_non_fb
    @skip_if_no_cuda
    @skip_if_no_network
    def test_clip_text_encoder_default(self):
        # ViT-B/32
        torch.manual_seed(2)
        params_config = OmegaConf.create({
            "is_pretrained": True,
        })
        encoder = ClipTextEncoder(params_config)
        encoded = encoder(torch.randint(0, 45000, (8, 77)))
        self.assertAlmostEqual(encoded.mean().item(), -0.0043, 4)

    @skip_if_non_fb
    @skip_if_no_cuda
    def test_clip_text_encoder_random_init(self):
        # RN50
        torch.manual_seed(2)
        params_config = OmegaConf.create({
            "embed_dim": 1024,
            "vocab_size": 49408,
            "context_length": 77,
            "transformer_width": 512,
            "transformer_layers": 12,
            "transformer_heads": 8,
            "is_pretrained": False,
        })
        encoder = ClipTextEncoder(params_config)
        encoded = encoder(torch.randint(0, 45000, (8, 77)))
        self.assertEqual(encoded.mean().item(), 0)

    @skip_if_non_fb
    @skip_if_no_cuda
    def test_clip_text_encoder_manifold(self):
        # RN101
        torch.manual_seed(2)
        m = (
            "manifold://fair_mmf/tree/torch/mmf/model/"
            + "clip/encoders/clip_text_encoder_RN101.pt"
        )
        params_config = OmegaConf.create({
            "pretrained_model_name": "RN101",
            "pretrained_model": {
                "url": m,
            },
            "is_pretrained": True,
        })
        encoder = ClipTextEncoder(params_config)
        encoded = encoder(torch.randint(0, 45000, (8, 77)))
        self.assertAlmostEqual(encoded.mean().item() * 1E4, -48.33, 2)

# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import unittest

import torch
from omegaconf import OmegaConf
from tests.test_utils import skip_if_non_fb


MODEL_PATH = "manifold://nlp_technologies/tree/xlm/models/xlm_r/model"
VOCAB_PATH = "manifold://nlp_technologies/tree/xlm/models/xlm_r/vocab"


class TestTransformerProcessors(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "tokenizer_config": {
                    "type": "pytext_spm",
                    "params": {
                        "model_path": MODEL_PATH,
                        "vocab_path": VOCAB_PATH,
                        "max_vocab": 250002,
                        "do_lower_case": True,
                    },
                },
                "mask_probability": 0,
                "max_seq_length": 128,
            }
        )

    @skip_if_non_fb
    def test_sentencepiece_tokenizer(self):
        from mmf.datasets.fb.processors.transformer_processors import SPMTokenizer

        processor = SPMTokenizer(self.config)

        # Test normal caption
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.ones(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [0, 3293, 1221, 186, 10, 3034, 111, 47, 84694, 32, 2], dtype=torch.long
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:11] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test empty caption
        arg = {"text": ""}
        results = processor(arg)
        expected_input_ids = torch.ones(128, dtype=torch.long)
        expected_input_ids[:3] = torch.tensor([0, 1, 2], dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:3] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test long caption
        arg = {"text": "I am working for facebook " * 100}  # make a long sentence
        results = processor(arg)
        expected_input_ids = [87, 444, 20697, 100, 10899] * 100
        expected_input_ids.insert(0, 0)  # <s>
        expected_input_ids = expected_input_ids[:128]
        expected_input_ids[-1] = 2  # </s>
        expected_input_ids = torch.tensor(expected_input_ids, dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.ones(128, dtype=torch.long)
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test two captions
        arg = {
            "text_a": "This will be a test of tokens?",
            "text_b": "I am working for facebook"
        }
        results = processor(arg)
        expected_input_ids = torch.ones(128, dtype=torch.long)
        expected_input_ids[:18] = torch.tensor(
            [0, 3293, 1221, 186, 10, 3034, 111, 47, 84694, 32, 2]
            + [2, 87, 444, 20697, 100, 10899, 2],
            dtype=torch.long
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:18] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

    @skip_if_non_fb
    def test_masked_sentencepiece_tokenizer(self):
        from mmf.datasets.fb.processors.transformer_processors import (
            MultiSentenceSPMTokenizer,
        )

        new_config = copy.deepcopy(self.config)
        new_config.mask_probability = 1.0
        processor = MultiSentenceSPMTokenizer(new_config)

        # Test masked caption
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.ones(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [0, 3293, 1221, 186, 10, 3034, 111, 47, 84694, 32, 2], dtype=torch.long
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        self.assertFalse(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))

        # Test <mask> token is present
        self.assertTrue(250001 in results["input_ids"])

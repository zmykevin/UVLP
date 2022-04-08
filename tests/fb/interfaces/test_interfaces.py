# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

import numpy as np
import tests.test_utils as test_utils
import torch
from mmf.utils.build import build_processors
from mmf.utils.env import setup_imports


class TestMMFTInterfaces(unittest.TestCase):
    def setUp(self):
        from mmf.models.fb.interfaces.mmft_interface import MMFTInterface

        test_utils.setup_proxy()
        setup_imports()

        save_dir = "manifold://multimodal_training/tree/categorization/test"
        yaml_path = os.path.join(save_dir, "config.yaml")

        self.model = MMFTInterface(yaml_path)
        self.model.to("cpu")
        self.model.eval()
        self.config = self.model.config

        model_name = self.config.model
        model_config = self.config.model_config[model_name]
        image_dim = model_config.image_encoder.params.in_dim

        self.test_text = ["Test samples."] * 2
        self.image_feature = torch.rand(2, 1, image_dim)

    @test_utils.skip_if_non_fb
    def test_mmft_interface_text_processor(self):
        from pytext.torchscript.utils import ScriptBatchInput

        text_processor = self._load_text_processor_from_config()
        text_processor_to_script = self.model.text_processor
        text = self.test_text[0]

        unscripted_text = text_processor({"text": text})
        scripted_text = text_processor_to_script(
            ScriptBatchInput(texts=[[text]], tokens=None, languages=None)
        )

        unscripted_input_ids = unscripted_text["input_ids"].detach().numpy()
        scripted_input_ids = scripted_text[0][0].detach().numpy()
        np.testing.assert_equal(unscripted_input_ids, scripted_input_ids)

        unscripted_input_mask = unscripted_text["input_mask"].detach().numpy()
        scripted_input_mask = scripted_text[1][0].detach().numpy()
        np.testing.assert_equal(unscripted_input_mask, scripted_input_mask)

        unscripted_segment_ids = unscripted_text["segment_ids"].detach().numpy()
        scripted_segment_ids = scripted_text[2][0].detach().numpy()
        np.testing.assert_equal(unscripted_segment_ids, scripted_segment_ids)

    @test_utils.skip_if_non_fb
    def test_mmft_interface_torchscripted_model(self):
        from pytext.torchscript.utils import ScriptBatchInput

        text = [[t] for t in self.test_text]
        script_text = self.model.text_processor(
            ScriptBatchInput(texts=text, tokens=None, languages=None)
        )

        sample = {
            "input_ids": script_text[0],
            "input_mask": script_text[1],
            "segment_ids": script_text[2],
            "image_feature_0": self.image_feature,
        }

        script_model = self._generate_script_model()

        model_output = self.model.model(sample)
        script_model_output = script_model(
            {"image_feature_0": sample["image_feature_0"].squeeze()},
            self.test_text,
            ["label", "confidence", "scores"],
        )

        model_output_score = (
            torch.nn.functional.softmax(model_output["scores"], dim=1).detach().numpy()
        )
        script_model_output_score = script_model_output["scores"].detach().numpy()
        np.testing.assert_almost_equal(
            model_output_score, script_model_output_score, decimal=5
        )

    def _load_text_processor_from_config(self):
        config = self.config
        dataset_name = config.datasets
        dataset_config = config.dataset_config[dataset_name]
        extra_params = {"data_dir": dataset_config.data_dir}
        processor_dict = build_processors(
            dataset_config.processors.batch_processor.params.processors, **extra_params
        )

        text_processor = processor_dict["text_processor"]
        return text_processor

    def _generate_script_model(self):
        script_model = self.model.torchscriptify()
        script_model.eval()
        return script_model

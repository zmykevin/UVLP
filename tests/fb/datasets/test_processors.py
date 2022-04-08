# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import tempfile
import unittest
from unittest.mock import patch

import torch
from mmf.common.sample import SampleList
from omegaconf import OmegaConf
from tests.test_utils import (
    compare_tensors,
    setup_proxy,
    skip_if_non_fb,
    skip_if_no_network,
)


def identity(x):
    return x


RET_VALUE = {"feature_processor": identity, "text_processor": identity}
CLIP_VOCAB_HASH = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
CLIP_CONTEXT_LENGTH = 77
CLIP_MANIFOLD = "manifold://fair_mmf/tree/data/vocabs/bpe_simple_vocab_16e6.txt.gz"


@contextlib.contextmanager
def mock_env_build_processors():
    patched = patch("mmf.utils.build.build_processors", return_value=RET_VALUE)
    patched.start()
    yield
    patched.stop()


@contextlib.contextmanager
def mock_env_build_kd_processors():
    KD_RET_VALUE = RET_VALUE.copy()
    KD_RET_VALUE.update({"teacher_text_processor": identity})
    patched = patch("mmf.utils.build.build_processors", return_value=KD_RET_VALUE)
    patched.start()
    yield
    patched.stop()


class TestDatasetProcessors(unittest.TestCase):
    def setUp(self):
        setup_proxy()

    @skip_if_non_fb
    def test_xray_hash_processor(self):
        from mmf.datasets.fb.processors.batch_processors import XrayHashProcessor

        config = OmegaConf.create({})
        processor = XrayHashProcessor(config)

        expected = torch.tensor([[1, 1, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.float)
        expected = expected.unsqueeze(1)

        # Test hashes saved as string
        arg = ["[1, 1, 0, 1, 1]", "[0, 0, 1, 1, 1]"]
        self.assertTrue(compare_tensors(processor(arg).image_feature_0, expected))

        # Test direct tensor input
        arg = [torch.tensor([1, 1, 0, 1, 1]), torch.tensor([0, 0, 1, 1, 1])]
        self.assertTrue(compare_tensors(processor(arg).image_feature_0, expected))

        # Test tensors with metadata, assigned None here.
        arg = [
            (torch.tensor([1, 1, 0, 1, 1]), None),
            (torch.tensor([0, 0, 1, 1, 1]), None),
        ]
        self.assertTrue(compare_tensors(processor(arg).image_feature_0, expected))

    @skip_if_non_fb
    def test_xray_hash_processor_unpack(self):
        from mmf.datasets.fb.processors.batch_processors import XrayHashProcessor

        # Test unpack packed xray hash
        config = OmegaConf.create({"is_packed": True})
        processor = XrayHashProcessor(config)

        expected = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -0.0,
                    -0.0,
                ]
            ]
        )
        expected = expected.unsqueeze(1)

        arg = [torch.tensor([12, -4])]
        self.assertTrue(compare_tensors(processor(arg).image_feature_0, expected))

    def _test_processor(self, processor_cls, config, arg, expected):
        with mock_env_build_processors():
            processor = processor_cls(config)
            self.assertEqual(processor(arg), expected)

    @skip_if_non_fb
    def test_feature_and_text_processor(self):
        from mmf.datasets.fb.processors.batch_processors import (
            FeaturesWithTextBatchProcessor,
        )

        config = OmegaConf.create(
            {
                "text_keys": ["post", "ocr"],
                "label_key": "label",
                "feature_key": "feature",
                "id_key": "id",
            }
        )

        data = {
            "feature": {"image_feature_0": 1},
            "post": "1",
            "ocr": "2",
            "id": (1, 1),
            "label": (torch.tensor(1), 1),
        }

        sample_list = SampleList()
        sample_list.image_feature_0 = 1
        sample_list.text = [["1", "2"]]
        sample_list.id = 1
        sample_list.targets = torch.tensor(1, dtype=torch.long)
        self._test_processor(FeaturesWithTextBatchProcessor, config, data, sample_list)

        config.text_keys = "post"
        sample_list.text = [["1"]]
        self._test_processor(FeaturesWithTextBatchProcessor, config, data, sample_list)

        config.text_keys = ["ocr"]
        sample_list.text = [["2"]]
        self._test_processor(FeaturesWithTextBatchProcessor, config, data, sample_list)

    def _test_clip_processor(self, config):
        from mmf.datasets.fb.processors.clip_processor import CLIPProccessor

        clip_processor = CLIPProccessor(config)
        text = "Taken with my analogue EOS 500N with black & white film."
        data = {"text": text}
        processed_ids = [
            49406,
            2807,
            593,
            607,
            46031,
            17805,
            276,
            271,
            271,
            333,
            593,
            1449,
            261,
            1579,
            1860,
            269,
            49407,
        ]
        expected_tensor = torch.tensor(
            processed_ids + [0] * (CLIP_CONTEXT_LENGTH - len(processed_ids)),
            dtype=torch.long,
        )
        output = clip_processor(data)
        self.assertEqual(
            output["text"].type(torch.DoubleTensor).mean(),
            expected_tensor.type(torch.DoubleTensor).mean(),
        )

    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_root")
    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_env")
    @skip_if_non_fb
    def test_clip_processor_on_disk(self, env, mmf_dir):
        from mmf.utils.file_io import PathManager

        local_file = PathManager.get_local_path(CLIP_MANIFOLD)
        config = OmegaConf.create(
            {
                "max_seq_length": CLIP_CONTEXT_LENGTH,
                "vocab_file": {
                    "url": local_file,
                    "hashcode": CLIP_VOCAB_HASH,
                    "file_name": "bpe_simple_vocab_16e6.txt.gz",
                },
            }
        )

        tmp_dir = tempfile.TemporaryDirectory()
        mmf_dir.return_value = tmp_dir.name
        env.return_value = ""
        self._test_clip_processor(config)
        tmp_dir.cleanup()

    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_root")
    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_env")
    @skip_if_non_fb
    def test_clip_processor_manifold(self, env, mmf_dir):
        config = OmegaConf.create(
            {
                "max_seq_length": CLIP_CONTEXT_LENGTH,
                "vocab_file": {
                    "url": CLIP_MANIFOLD,
                    "hashcode": CLIP_VOCAB_HASH,
                    "file_name": "bpe_simple_vocab_16e6.txt.gz",
                },
            }
        )

        tmp_dir = tempfile.TemporaryDirectory()
        mmf_dir.return_value = tmp_dir.name
        env.return_value = ""
        self._test_clip_processor(config)
        tmp_dir.cleanup()

    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_root")
    @patch("mmf.datasets.fb.processors.clip_processor.get_mmf_env")
    @skip_if_non_fb
    @skip_if_no_network
    def test_clip_processor_web(self, env, mmf_dir):
        config = OmegaConf.create(
            {
                "max_seq_length": CLIP_CONTEXT_LENGTH,
            }
        )

        tmp_dir = tempfile.TemporaryDirectory()
        mmf_dir.return_value = tmp_dir.name
        env.return_value = ""
        self._test_clip_processor(config)
        tmp_dir.cleanup()

    def _test_kd_processor(self, processor_cls, config, arg, expected):
        with mock_env_build_kd_processors():
            processor = processor_cls(config)
            processed_data = processor(arg)

            for key in (
                "teacher_texts",
                "student_texts",
                "teacher_features",
                "student_features",
            ):
                self.assertTrue(key in processed_data)
                self.assertEqual(processed_data[key], expected[key])

    @skip_if_non_fb
    def test_knowledge_distillation_processor(self):
        from mmf.datasets.fb.processors.batch_processors import (
            KnowlegeDistillationBatchProcessor,
        )

        # define processor keys
        config = OmegaConf.create(
            {
                "feature_key": "feature",
                "teacher_feature_key": "teacher_feature",
                "text_keys": ["text"],
                "label_key": "label",
                "id_key": "id",
            }
        )

        # define input features
        data = {
            "feature": {"image_feature_0": 1},
            "teacher_feature": {"image_feature_0": 1},
            "text": "1",
            "id": (1, 1),
            "label": (torch.tensor(1), 1),
        }

        sample_list = SampleList()
        # build student features
        sample_list.student_features = SampleList()
        sample_list.student_features.image_feature_0 = 1
        # build student texts
        sample_list.student_texts = SampleList()
        sample_list.student_texts.text = [["1"]]
        # build teacher features
        sample_list.teacher_features = SampleList()
        sample_list.teacher_features.image_feature_0 = 1
        # build teacher texts
        sample_list.teacher_texts = SampleList()
        sample_list.teacher_texts.text = [["1"]]

        sample_list.id = 1
        sample_list.targets = torch.tensor(1, dtype=torch.long)

        self._test_kd_processor(
            KnowlegeDistillationBatchProcessor, config, data, sample_list
        )

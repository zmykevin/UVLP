# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Type

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.datasets.processors.processors import (
    BatchProcessor,
    BatchProcessorConfigType,
    ProcessorConfigType,
)
from PIL import Image


@dataclass
class MisinfoProcessorsType:
    text_processor: ProcessorConfigType
    feature_processor: ProcessorConfigType


@dataclass
class MisinfoBatchProcessorConfigType(BatchProcessorConfigType):
    processors: MisinfoProcessorsType
    text_keys: List[str]
    feature_key: str
    label_key: str
    id_key: str


@dataclass
class XRayHashProcessorConfigType(BatchProcessorConfigType):
    is_packed: bool = False
    feature_dim: int = 256
    key_name: str = "image_feature_0"


@dataclass
class TextRayProcessorConfigType(BatchProcessorConfigType):
    key_name: str = "textray"
    max_seq_length: int = 256
    unpack_cls_token_only: bool = False


@registry.register_processor("xray_hash")
class XrayHashProcessor(BatchProcessor):
    """Use this processor to process XRay hashes stored in the Hive table.
    This wil convert them to proper tensor and return that back in a sample
    list. The features will be saved in the attribute image_feature_0 which
    is what most of the MMF models expect. Expected to be run on batch level
    processors.
    """

    def __init__(self, config: BatchProcessorConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_packed = config.get("is_packed", False)
        self.feature_dim = config.get("feature_dim", 256)
        # Most models expect image_feature_0 as the first image feature
        self.key_name = config.get("key_name", "image_feature_0")

    def __call__(self, item: List[Any], *args, **kwargs) -> Type[SampleList]:
        sample_list = SampleList()
        xray_hashes = []
        was_direct_tensor = False
        for x in item:
            if x is not None:
                # If the xray hash is already parsed via HiveDataset then we don't
                # need to parse anything and return it as it is
                if isinstance(x, tuple) and torch.is_tensor(x[0]):
                    xray_hash = x[0]
                    was_direct_tensor = True
                # The col value in consideration is already a tensor so use it directly
                elif torch.is_tensor(x):
                    xray_hash = x
                    was_direct_tensor = True
                else:
                    xray_hash = list(map(lambda x: int(x), x[1:-1].split(",")))

                if self.is_packed:
                    xray_hash = self.integer2bit(xray_hash).view(-1)
            else:
                # Handles the case where a sample doesn't have an image XrayHash
                xray_hash = torch.zeros(self.feature_dim)
                was_direct_tensor = True

            xray_hashes.append(xray_hash)

        if was_direct_tensor:
            xray_tensor = torch.stack(xray_hashes)
        else:
            xray_tensor = torch.tensor(xray_hashes, dtype=torch.float)

        # B X D => B X 1 X D and 1 as number of image features
        sample_list[self.key_name] = xray_tensor.unsqueeze(1).float()
        return sample_list

    def integer2bit(self, integer: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
        """Turn integer tensor to binary representation.
        In current binarized data xray was not flattened to bits
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
        """
        exponent_bits = -torch.arange(-(num_bits - 1), 1, dtype=integer.dtype)
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        out = (out - (out % 1)) % 2
        return out.view(out.shape[0], -1)


@registry.register_processor("textray")
class TextRayProcessor(BatchProcessor):
    """Use this processor to process TextRay 2D embedding stored in Hive tables."""

    def __init__(self, config: TextRayProcessorConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.key_name = config.get("key_name", "textray")
        self.max_seq_len = config.get("max_seq_len", 256)
        self.unpack_cls_token_only = config.get("unpack_cls_token_only", False)

        torch.ops.load_library(
            "//caffe2/torch/fb/nlp/operators:textray_feature_extractor"
        )

    def __call__(self, item: List[Any], *args, **kwargs) -> Type[SampleList]:
        sample_list = SampleList()
        textray_embeddings = []
        for feature in item["text"]:
            textray_embeddings.append(self._unpack_textray(feature))
        # Pad to max seq_len and batch
        sample_list[self.key_name] = torch.cat(textray_embeddings, dim=0)
        return sample_list

    def _unpack_textray(self, feature):
        feature = (torch.zeros(1).long(),) + feature
        if self.unpack_cls_token_only:
            cls_embedding = (
                torch.ops.internal.batch_textray_1d_feature_from_int_tensor_feature(
                    feature
                )
            )
            outputs = cls_embedding[0].unsqueeze(0)
        else:
            token_embeddings = (
                torch.ops.internal.batch_textray_2d_feature_from_int_tensor_feature(
                    feature
                )[0]
            )
            # Shape (bsz, max_seq_length, dim)
            outputs = F.pad(
                token_embeddings,
                [0, 0, 0, self.max_seq_len - token_embeddings.shape[0]],
            ).unsqueeze(0)
        return outputs


@registry.register_processor("features_with_text")
class FeaturesWithTextBatchProcessor(BatchProcessor):
    """Batch processor specific for datasets which have features
    and some text channels. Depending on which feature and
    text processor are defined, it returns back a SampleList
    that is usable by transformer based models.

    In configuration, specify feature column as feature_key and
    text columns as text_keys. Additionally, unique id column can
    be specified as id_key and the labels as label_key.

    As of now, all of the text from text_keys column is concatenated
    and passed as single string from the text processor that
    has been defined.

    If you want to define a custom version of this processor, follow
    the steps:
    - Inherit this class and register a new processor for it
    - Override the method you want among all of the `process_` methods
    - If you add a new `process_` method for processing some new column
    make sure to override the pipeline property to include that
    method into the pipeline.

    You can check an example override below in ImageWithTextBatchProcessor.
    """

    def __init__(self, config: MisinfoBatchProcessorConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._text_keys = config.text_keys
        # If not list, convert to list. Note that, we can't directly
        # compare OmegaConf.ListConfig to list due to OmegaConf limitation
        if not isinstance(self._text_keys, collections.abc.MutableSequence):
            self._text_keys = [self._text_keys]
        # Convert to container list
        self._text_keys = list(self._text_keys)

        if "feature_key" not in config:
            warnings.warn(
                """Feature key is missing. This is only intended to be used
                for ImageWithTextBatchProcessor without any feature inputs."""
            )
        self._feature_key = config.get("feature_key", None)

        if "label_key" not in config:
            warnings.warn(
                "Label key is missing. For downstream tasks this might be an issue."
            )
        self._label_key = config.get("label_key", None)

        self._id_key = config.id_key

    @property
    def pipeline(self):
        if self._feature_key is not None:
            processors = [self.process_features]
        else:
            processors = []
        processors.extend([self.process_text, self.process_id])
        if self._label_key:
            processors.append(self.process_targets)
        return processors

    def process_features(self, data):
        return self.processors["feature_processor"](data[self._feature_key])

    def process_text(self, data):
        sources = []
        for col in self._text_keys:
            if col in data:
                sources.extend([data[col]])

        texts = []
        for item in zip(*sources):
            sample = Sample()
            item = [it or "" for it in item]
            processed_text = self.processors["text_processor"]({"text": list(item)})
            sample.update(processed_text)
            texts.append(sample)

        texts = SampleList(texts)
        return texts

    def process_targets(self, data):
        sample_list = SampleList()
        targets = data[self._label_key][0]
        sample_list.targets = targets.long()
        return sample_list

    def process_id(self, data):
        sample_list = SampleList()
        sample_list.id = data[self._id_key][0]
        return sample_list

    def __call__(self, data: Dict[str, Any]) -> Type[SampleList]:
        sample_list = SampleList()
        for func in self.pipeline:
            sample_list.update(func(data))
        return sample_list


@registry.register_processor("image_with_text")
class ImageWithTextBatchProcessor(FeaturesWithTextBatchProcessor):
    def process_image(self, data):
        sample_list = SampleList()
        images = data["image"]
        processed_images = []

        for image in images:
            image = image.numpy().tobytes()
            with Image.open(BytesIO(image), mode="r") as pil_img:
                image = pil_img.convert("RGB")
                processed_images.append(self.processors["image_processor"](image))

        sample_list.image = torch.stack(processed_images)
        return sample_list

    @property
    def pipeline(self):
        pipeline = super().pipeline
        return pipeline + [self.process_image]


@registry.register_processor("multiclass_features_with_text")
class MultiClassFeaturesWithText(FeaturesWithTextBatchProcessor):
    def process_targets(self, data):
        sample_list = SampleList()
        labels = data[self._label_key]
        sample_list.targets = torch.stack(
            [
                self.processors["label_processor"]({"label": label})["class_index"]
                for label in labels
            ]
        )
        return sample_list


@registry.register_processor("kd")
class KnowlegeDistillationBatchProcessor(FeaturesWithTextBatchProcessor):
    """A batch processor used in knowledge distillation

    Supports to process texts and features in teacher and student model (can be
    different).
    """

    def __init__(self, config: MisinfoBatchProcessorConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._teacher_feature_key = config.get("teacher_feature_key", self._feature_key)

        assert (
            self._teacher_feature_key or self._text_keys
        ), "KeyError: At least one of feature key or text keys should be present"

    @property
    def pipeline(self):
        pipelines = super().pipeline + [self.process_teacher_text]
        if self._teacher_feature_key:
            pipelines.append(self.process_teacher_features)
        return pipelines

    def process_features(self, data):
        features = super().process_features(data)
        return SampleList({"student_features": features})

    def process_teacher_features(self, data):
        features = self.processors["feature_processor"](data[self._teacher_feature_key])
        return SampleList({"teacher_features": features})

    def process_text(self, data):
        texts = super().process_text(data)
        return SampleList({"student_texts": texts})

    def process_teacher_text(self, data):
        sources = []
        for col in self._text_keys:
            if col in data:
                sources.extend([data[col]])

        texts = []
        for item in zip(*sources):
            sample = Sample()
            item = [it or "" for it in item]
            processed_text = self.processors["teacher_text_processor"](
                {"text": list(item)}
            )
            sample.update(processed_text)
            texts.append(sample)

        texts = SampleList(texts)
        return SampleList({"teacher_texts": texts})

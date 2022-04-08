# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder

from .masked_text_dataset import MaskedConceptualCaptionsTextDataset


@registry.register_builder("masked_conceptual_captions_text")
class MaskedConceptualCaptionsTextBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_conceptual_captions_text"
        self.set_dataset_class(MaskedConceptualCaptionsTextDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/masked_text.yaml"

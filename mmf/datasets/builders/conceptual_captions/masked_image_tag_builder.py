# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder

from .masked_image_tag_dataset import MaskedConceptualCaptionsImageTagDataset


@registry.register_builder("masked_conceptual_captions_image_tag")
class MaskedConceptualCaptionsImageTagBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_conceptual_captions_image_tag"
        self.set_dataset_class(MaskedConceptualCaptionsImageTagDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/masked_image_tag.yaml"
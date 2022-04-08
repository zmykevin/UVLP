from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder

from .itm_dataset import ITMConceptualCaptionsDataset
@registry.register_builder("itm_conceptual_captions")
class ITMConceptualCaptionsBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "itm_conceptual_captions"
        self.set_dataset_class(ITMConceptualCaptionsDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/itm.yaml"
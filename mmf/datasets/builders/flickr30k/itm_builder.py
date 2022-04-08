from mmf.common.registry import registry
from mmf.datasets.builders.vqa2.builder import VQA2Builder

from .itm_dataset import ITMFlickr30KDataset

@registry.register_builder("itm_flickr30k")
class ITMFlickr30KBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "itm_flickr30k"
        self.set_dataset_class(ITMFlickr30KDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/flickr30k/itm.yaml"
# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.sample import Sample
from mmf.datasets.builders.coco import MaskedCOCODataset
import torch
import random

class MaskedConceptualCaptionsImageDataset(MaskedCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_conceptual_captions_image"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if self._use_features:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )

            if self.config.get("use_image_feature_masks", False):
                current_sample.update(
                    {
                        "image_labels": self.masked_region_processor(
                            features["image_feature_0"]
                        )
                    }
                )
            #Added by Mingyang Zhou
            if sample_info.get("objects_ids", None) is not None:
                objects_ids=torch.ones_like(current_sample.image_labels)*(-1)
                objects_ids[:len(sample_info["objects_ids"])] = torch.tensor(sample_info["objects_ids"], dtype=torch.long)
                current_sample.update({"objects_ids": objects_ids})
            current_sample.update(features)
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample)

        #Make the input_ids and lm_label_ids  corresponding value
        #print(current_sample["input_ids"])
        # print(current_sample[""])
        return current_sample
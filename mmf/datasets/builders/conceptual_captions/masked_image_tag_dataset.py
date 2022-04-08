# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.sample import Sample
from mmf.datasets.builders.coco import MaskedCOCODataset
import torch
import random

class MaskedConceptualCaptionsImageTagDataset(MaskedCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_conceptual_captions_image_tag"
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
            
            #Added by Mingyang
            if sample_info.get("objects_ids", None) is not None:
                objects_ids=torch.ones_like(current_sample.image_labels)*(-1)
                objects_ids[:len(sample_info["objects_ids"])] = torch.tensor(sample_info["objects_ids"], dtype=torch.long)
                current_sample.update({"objects_ids": objects_ids})

            current_sample.update(features)

        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample)
        return current_sample
    def _add_masked_caption(self, sample_info, current_sample):
        captions = sample_info["captions"]
        if sample_info.get("objects", None) is not None:
            objects = sample_info["objects"]
            #print(objects)
            #print(len(objects))
            bbox = current_sample["image_info_0"]["bbox"]
            #print(bbox.shape)
            # print(bbox[0])
            # print(bbox[19])
        image_id = sample_info["image_id"]
        #get the bbox info, and we want to create visual_tag_bbox from this information
        
        selected_caption = objects
        other_caption = None
        is_correct = -1

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_a_bbox": bbox,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

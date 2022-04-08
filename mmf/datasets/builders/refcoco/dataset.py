import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2 import VQA2Dataset


LABEL_TO_INT_MAPPING = {"entailment": 0, "neutral": 1, "contradiction": 2}


class RefCOCODataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="refcoco",
            *args,
            **kwargs
        )
        # print(self.config.dataset_name)
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        
        processed_sentence = self.text_processor({"text": sample_info["sentence"]})
        label = sample_info["label"]
        label_list = sample_info["label_list"]
        current_split = sample_info["split"]
        # print(sample_info["split"])

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)
        
        assert self._use_features is True
        if self._use_features is True:
            # Remove sentence id from end
            #identifier = sample_info["Flikr30kID"].split(".")[0]
            # Load img0 and img1 features
            sample_info["feature_path"] = sample_info['file_name']
            features = self.features_db[idx]
            
            # if current_split == "train":
            #     masked_feature = features["image_feature_0"]
            #     for x in label_list:
            #         #x is the index that has IOU > 0.5 with GT
            #         if x != label and x != 100:
            #             masked_feature[x] = 0

            #     features["image_feature_0"] = masked_feature

                
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        #This index record the index for the ground truth regions
        current_sample.targets = torch.tensor(label, dtype=torch.long)
        current_sample.targets_list = torch.tensor(label_list, dtype=torch.long)
        return current_sample

    def format_for_prediction(self, report):
        return []

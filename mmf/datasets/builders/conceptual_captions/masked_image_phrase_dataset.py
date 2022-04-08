# Copyright (c) Facebook, Inc. and its affiliates.
import random
import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.coco import MaskedCOCODataset


class MaskedConceptualCaptionsImagePhraseDataset(MaskedCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_conceptual_captions_image_phrase"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
        self._top1 = config.get("top1_enabled", False)
        self.enable_MRTM = config.get("mrtm_enabled", False)
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        
        #Compute OMVM_scores
        OMVM_scores = []

        #get the scores
        current_sample, selected_caption_index = self._add_masked_caption(sample_info, current_sample)

        if self._use_features:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )

            if self.config.get("use_image_feature_masks", False):
                #prepare OMVM_scores
                objects_ids = sample_info["objects_ids"]
                # if sample_info.get("noun_phrases", None) is not None:
                #     object_nps = sample_info["noun_phrases"][selected_caption_index]
                # else:

                if sample_info.get("objects_cs", None) is not None:
                    #get the obejct_nps
                    object_nps = sample_info["noun_phrases"][selected_caption_index]
                    
                    #selected
                    OMVM_scores = sample_info["objects_cs"][:,selected_caption_index]
                    objects_cs_args = sample_info["objects_cs_arg"][:,selected_caption_index]
                    
                    MRTM_nps = []
                    # if len(object_nps) > 0:
                    for cs, cs_arg in zip(OMVM_scores, objects_cs_args):
                        # print(cs)
                        # print(cs_arg)
                        if cs > 0 and len(object_nps) > 0:
                            MRTM_nps.append(object_nps[int(cs_arg)])
                        else:
                            MRTM_nps.append(None)
                    #print(MRTM_nps)

                    assert len(OMVM_scores) == len(objects_ids)
                    assert len(OMVM_scores) -- len(MRTM_nps)
                else:
                    OMVM_scores = [1]*len(objects_ids)
                    MRTM_nps = []

                if self.enable_MRTM:
                    #print(self.masked_region_processor.type)
                    image_labels, MRTM_labels = self.masked_region_processor(features["image_feature_0"], OMVM_scores, MRTM_nps)
                    current_sample.update({"image_labels": image_labels, "mrtm_labels": MRTM_labels})
                else:
                    #print(self.masked_region_processor.type)
                    image_labels, _ = self.masked_region_processor(features["image_feature_0"], OMVM_scores, None)
                    #image_labels = self.masked_region_processor(features["image_feature_0"])
                    objects_ids=torch.ones_like(image_labels)*(-1)
                    objects_ids[:len(sample_info["objects_ids"])] = torch.tensor(sample_info["objects_ids"], dtype=torch.long)
                    current_sample.update({"image_labels": image_labels, "objects_ids": objects_ids})

            current_sample.update(features)
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        #current_sample = self._add_masked_caption(sample_info, current_sample)
        return current_sample
    
    # def _process_masked_region_OMVM()
    # def _process_masked_region_MRTM()
    def _add_masked_caption(self, sample_info, current_sample):
        captions = sample_info["captions"]
        # if self._top1:
        #     captions = [sample_info["captions"][0]]
        image_id = sample_info["image_id"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1)
        other_caption_indices = [
            i for i in range(num_captions) if i != selected_caption_index
        ]
        selected_caption = captions[selected_caption_index]
        #add the selected_noun_phrase
        if sample_info.get("noun_phrases", None) is not None:
            selected_noun_phrase = sample_info["noun_phrases"][selected_caption_index]
        else:
            selected_noun_phrase = []
        other_caption = None
        is_correct = -1

        # if self._two_sentence:
        #     if random.random() > self._two_sentence_probability:
        #         other_caption = self._get_mismatching_caption(image_id)
        #         is_correct = False
        #     else:
        #         other_caption = captions[random.choice(other_caption_indices)]
        #         is_correct = True
        # elif self._false_caption:
        #     if sample_info.get("neg_captions", None) is not None:
        #         if sample_info["neg_captions"]:
        #             selected_caption = sample_info["neg_captions"][0]
        #             is_correct = False
        #         else:
        #             is_correct = True
        #     else: 
        #         if random.random() < self._false_caption_probability:
        #             selected_caption = self._get_mismatching_caption(image_id)
        #             is_correct = False
        #         else:
        #             is_correct = True
        
        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_a_np": selected_noun_phrase,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample, selected_caption_index

    def _get_mismatching_caption(self, image_id):
        other_item = self.annotation_db[random.randint(0, len(self.annotation_db) - 1)]

        while other_item["image_id"] == image_id:
            other_item = self.annotation_db[
                random.randint(0, len(self.annotation_db) - 1)
            ]

        other_caption = other_item["captions"][
            random.randint(0, len(other_item["captions"]) - 1)
        ]
        return other_caption
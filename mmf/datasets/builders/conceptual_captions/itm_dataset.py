import random

from mmf.common.sample import Sample
from mmf.datasets.builders.coco import MaskedCOCODataset
import torch

class ITMConceptualCaptionsDataset(MaskedCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_conceptual_captions"
        self._two_sentence = config.get("two_sentence", False)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
        self._top1 = config.get("top1_enabled", False)
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        if sample_info.get("filename", None) is not None:
            identifier = sample_info["filename"].split(".")[0]
            sample_info["feature_path"] = "{}.npy".format(identifier)
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

            current_sample.update(features)
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample)
        return current_sample

    def _add_masked_caption(self, sample_info, current_sample):
        captions = sample_info["captions"]
        if self._top1:
            captions = [sample_info["captions"][0]]
        if sample_info.get("filename", None) is not None:
            image_id = sample_info["filename"]
        else:
            image_id = sample_info["image_id"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1)
        other_caption_indices = [
            i for i in range(num_captions) if i != selected_caption_index
        ]
        selected_caption = captions[selected_caption_index]
        other_caption = None
        is_correct = -1
        # print(is_correct)
        # print(self._two_sentence)
        if self._two_sentence:
            if random.random() > self._two_sentence_probability:
                other_caption = self._get_mismatching_caption(image_id)
                is_correct = False
            else:
                other_caption = captions[random.choice(other_caption_indices)]
                is_correct = True
        elif self._false_caption:
            #print(sample_info.get("neg_captions", None))
            if sample_info.get("neg_captions", None) is not None:
                if sample_info["neg_captions"]:
                    selected_caption = sample_info["neg_captions"][0]
                    is_correct = False
                else:
                    is_correct = True
            else: 
                if random.random() < self._false_caption_probability:
                    selected_caption = self._get_mismatching_caption(image_id)
                    is_correct = False
                else:
                    is_correct = True
                # print(is_correct)

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

    def _get_mismatching_caption(self, image_id):
        other_item = self.annotation_db[random.randint(0, len(self.annotation_db) - 1)]

        if other_item.get("filename", None) is None: 
            while other_item["image_id"] == image_id:
                other_item = self.annotation_db[
                    random.randint(0, len(self.annotation_db) - 1)
                ]
        else:
            while other_item["filename"] == image_id:
                other_item = self.annotation_db[
                    random.randint(0, len(self.annotation_db) - 1)
                ]
        

        other_caption = other_item["captions"][
            random.randint(0, len(other_item["captions"]) - 1)
        ]
        return other_caption

    def format_for_prediction(self, report):
        #print(report.scores.size())
        # print(report)
        with torch.no_grad():
            answers=report.scores[:,1]
            #answers = self.softmax(report.scores)[:,1]
            #targets = report.targets
        
        predictions = []
        

        for idx, image_id in enumerate(report.image_info_0.image_id):
            answer_id = answers[idx].item()
            #gold_answer = targets[idx].item()

            predictions.append(
                {
                    "filename": image_id,
                    "answer": answer_id
                }
            )

        return predictions
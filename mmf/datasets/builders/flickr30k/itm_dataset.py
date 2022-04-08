import random
import copy
import json
import torch
#import torch.nn as nn
from torch import nn

from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2 import VQA2Dataset


class ITMFlickr30KDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        #super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="itm_flickr30k",
            *args,
            **kwargs
        )
        # self.dataset_name = "itm_flickr30k"
        self._false_caption = config.get("false_caption", True)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
        assert self._false_caption
        assert self._false_caption_probability == 0.5
        self.softmax = nn.Softmax(dim=1)
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        identifier = sample_info["filename"].split(".")[0]
        sample_info["feature_path"] = "{}.npy".format(identifier)
        current_sample = Sample()
        
        captions = sample_info["captions"]
        image_id = sample_info["filename"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1)
        selected_caption = captions[selected_caption_index]
        
        #print(sample_info["split"])
        # if self._false_caption and sample_info["split"] != "test":
        if sample_info["split"] in ["val", "test"]:
            #verify neg_captions exist
            if sample_info["split"] == "val":
                assert sample_info.get("neg_captions", None) is not None
                if sample_info["neg_captions"]:
                    selected_caption = sample_info["neg_captions"][0]
                    is_correct = False
                else:
                    is_correct = True
            else:
                is_correct = sample_info["label"]

            processed_sentence = self.text_processor({"text": selected_caption})
            processed_sentence.pop("tokens")
            # print(processed_sentence.keys())
            # print("input ids: ")
            # print(type(processed_sentence["input_ids"]))
            # print(processed_sentence["input_ids"])
            # print("input masks:")
            # print(type(processed_sentence["input_mask"]))
            # print(processed_sentence["input_mask"])
            # print("segment_ids: ")
            # print(type(processed_sentence["segment_ids"]))
            # print("lm_label_ids: ")
            # print(type(processed_sentence["lm_label_ids"]))
            # print("tokens: ")
            # print(type(processed_sentence["tokens"]))
            # print(processed_sentence["tokens"])
            # print("text")
            # print(type(processed_sentence["text"]))
            # print(processed_sentence["text"])
            # return

            current_sample.text = processed_sentence["text"]
            #print(processed_sentence.keys())
            if "input_ids" in processed_sentence:
                current_sample.update(processed_sentence)

            

            #current_sample = self._add_caption(sample_info, current_sample)
            
            #current_sample = self._add_caption(sample_info, current_sample)
            assert self._use_features
            if self._use_features:
                features = self.features_db[idx]
                if hasattr(self, "transformer_bbox_processor"):
                    features["image_info_0"] = self.transformer_bbox_processor(
                        features["image_info_0"]
                    )
                    #print(features["image_info_0"])
                #print(features["image_feature_0"].size())
                #print(features["image_info_0"].keys())
                # print(features["image_info_0"]["num_boxes"])
                # print(features["image_info_0"]["bbox"])
                # print(features["image_info_0"]["max_features"])

                #disable the num boexs which only matters to refcoco
                features["image_info_0"].pop("num_boxes")
                current_sample.update(features)
        else:
            assert sample_info["split"] == "train"

            #sample two negative captions
            selected_neg_cap_1 = self._get_mismatching_caption(image_id)
            selected_neg_cap_2 = self._get_mismatching_caption(image_id)
            while selected_neg_cap_2 == selected_neg_cap_1:
                selected_neg_cap_2 = self._get_mismatching_caption(image_id)
            #Get the processed_sentence
            processed_pos_sent = self.text_processor({"text": selected_caption})
            processed_neg_sent_1 = self.text_processor({"text": selected_neg_cap_1})
            processed_neg_sent_2 = self.text_processor({"text": selected_neg_cap_2})

            #Merge the Information
            processed_sentence = {}
            with torch.no_grad():
                processed_sentence["input_ids"] = torch.cat((processed_pos_sent["input_ids"].unsqueeze(0),
                                                             processed_neg_sent_1["input_ids"].unsqueeze(0),
                                                             processed_neg_sent_2["input_ids"].unsqueeze(0)
                                                            ), 0)
                processed_sentence["input_mask"] = torch.cat((processed_pos_sent["input_mask"].unsqueeze(0),
                                                             processed_neg_sent_1["input_mask"].unsqueeze(0),
                                                             processed_neg_sent_2["input_mask"].unsqueeze(0)
                                                            ), 0)
                processed_sentence["segment_ids"] = torch.cat((processed_pos_sent["segment_ids"].unsqueeze(0),
                                                             processed_neg_sent_1["segment_ids"].unsqueeze(0),
                                                             processed_neg_sent_2["segment_ids"].unsqueeze(0)
                                                            ), 0)
                processed_sentence["lm_label_ids"] = torch.cat((processed_pos_sent["lm_label_ids"].unsqueeze(0),
                                                             processed_neg_sent_1["lm_label_ids"].unsqueeze(0),
                                                             processed_neg_sent_2["lm_label_ids"].unsqueeze(0)
                                                            ), 0)
                #update the processed  text
                processed_sentence["text"] = [processed_pos_sent["text"], processed_neg_sent_1["text"], processed_neg_sent_2["text"]]

                current_sample.update(processed_sentence)
            #NOTICE: is_correct might need to be adjusted for val later
            is_correct = True
                
            #Update the features
            assert self._use_features
            if self._use_features:
                features = self.features_db[idx]
                if hasattr(self, "transformer_bbox_processor"):
                    features["image_info_0"] = self.transformer_bbox_processor(
                        features["image_info_0"]
                    )
                
                features["image_feature_0"] = features["image_feature_0"].unsqueeze(0).expand(3,-1,-1)

                features["image_info_0"]["bbox"] = features["image_info_0"]["bbox"].unsqueeze(0).expand(3,-1,-1)
                #print(features["image_info_0"]["bbox"].size())
                features["image_info_0"].pop("num_boxes")
                current_sample.update(features)
        #Always use the same images
        
        
        label = int(is_correct)
        #print(label)
        current_sample.targets = torch.tensor(label, dtype=torch.long)
        return current_sample
        

    # def _add_caption(self, sample_info, current_sample):
    #     captions = sample_info["captions"]
    #     image_id = sample_info["filename"]
    #     num_captions = len(captions)
    #     selected_caption_index = random.randint(0, num_captions - 1)
    #     selected_caption = captions[selected_caption_index]

    #     #print(sample_info["split"])
    #     if self._false_caption and sample_info["split"] != "test":
    #         #if sample_info.get("neg_captions", None) is not None :
    #         if sample_info["split"] == "val":
    #             #verify neg_captions exist
    #             assert sample_info.get("neg_captions", False)
    #             if sample_info["neg_captions"]:
    #                 selected_caption = sample_info["neg_captions"][0]
    #                 is_correct = False
    #             else:
    #                 is_correct = True
    #         else: 
    #             if random.random() < self._false_caption_probability:
    #                 selected_caption = self._get_mismatching_caption(image_id)
    #                 is_correct = False
    #             else:
    #                 is_correct = True
    #     #get the ground truth label
    #     if sample_info["split"] == "test":
    #         is_correct = sample_info["label"]
        
    #     processed_sentence = self.text_processor({"text": selected_caption})
    #     # processed.pop("tokens")
    #     current_sample.text = processed_sentence["text"]
    #     if "input_ids" in processed_sentence:
    #         current_sample.update(processed_sentence)

    #     label = int(is_correct)
    #     current_sample.targets = torch.tensor(label, dtype=torch.long)


    #     return current_sample

    def _get_mismatching_caption(self, image_id):
        other_item = self.annotation_db[random.randint(0, len(self.annotation_db) - 1)]

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
            targets = report.targets
        
        predictions = []
        

        for idx, image_id in enumerate(report.image_info_0.image_id):
            answer_id = answers[idx].item()
            gold_answer = targets[idx].item()

            predictions.append(
                {
                    "filename": image_id,
                    "answer": answer_id,
                    "gold_answer": gold_answer
                    # "actual_answers": actual_answer,
                    # "question_tokens": report.question_tokens[idx],
                    # "image_id": report.image_id[idx].item()
                }
            )

        return predictions
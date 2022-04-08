# Copyright (c) Facebook, Inc. and its affiliates.

import os

import torch
from mmf.common.registry import registry
from mmf.datasets.fb.processors.clip_processor import CLIPTokenizer
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_encoder
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.distributed import is_main, synchronize
from mmf.utils.download import download as download_file
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_current_device


@registry.register_model("clip")
class CLIPModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

        image_encoder_config = config.get("image_encoder", None)
        if image_encoder_config:
            self.image_encoder = build_encoder(image_encoder_config)

        text_encoder_config = config.get("text_encoder", None)
        if text_encoder_config:
            self.text_encoder = build_encoder(text_encoder_config)

        self.image_key = config.get("image_key", "image")
        self.text_key = config.get("text_key", "text")

        # exp(-config.temperature_init) should be 0.07 according to the CLIP paper
        # so config.temperature_init should be 2.659260036932778
        temperature = config.get("temperature_init", 2.659260036932778)
        self.learned_temperature = torch.nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32)
        )

    @classmethod
    def config_path(cls):
        return "configs/fb/models/clip/defaults.yaml"

    def build(self):
        task = self.config.get("task", None)
        if task and task.type == "zero_shot":
            self.build_zeroshot_task()

    def build_zeroshot_task(self):
        assert (
            len(self.config.task.params.label_id_to_name_map_file) > 0
        ), "need to provide id to text map for clip accuracy"
        assert (
            len(self.config.task.params.vocab_file) > 0
        ), "need to provide vocab file to allow for tokenization"

        templates_file = self.config.task.params.get("templates_file", None)
        if templates_file is None:
            templates = self.config.task.params.get("templates", None)
            if not templates:
                templates = ["a photo of a {}\n"]
        else:
            template_file_path = self.download(templates_file)
            with PathManager.open(template_file_path) as f:
                templates = f.readlines()

        classnames = None
        label_id_to_name_map_file_path = self.download(
            self.config.task.params.label_id_to_name_map_file
        )
        with PathManager.open(label_id_to_name_map_file_path) as f:
            lines = f.readlines()
            classnames = [line.split(",")[-1] for line in lines]

        vocab_file_path = self.download(self.config.task.params.vocab_file)
        context_length = self.config.task.params.context_length
        tokenizer = CLIPTokenizer(vocab_file_path)
        self.zeroshot_weights = self.zeroshot_classifier(
            tokenizer, classnames, templates, context_length
        )

    def zeroshot_classifier(self, tokenizer, classnames, templates, context_length):
        with torch.no_grad():
            zeroshot_weights = []
            device = get_current_device()
            self.text_encoder.to(device)
            for classname in classnames:
                texts = [
                    template.format(classname) for template in templates
                ]  # format with class; shape: len(templates)
                texts = self.tokenize(tokenizer, texts, context_length).to(device)
                class_embeddings = self.text_encoder(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(
                zeroshot_weights, dim=1
            )  # shape: emb_dim x #class
        return zeroshot_weights

    def tokenize(self, tokenizer, texts, context_length):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    def download(self, url):
        from hashlib import sha256

        if PathManager.exists(url):
            return url

        url_bytes = url.encode("utf-8")
        url_hash = sha256(url_bytes)
        filename = url_hash.hexdigest()

        dir_path = os.path.join(get_mmf_cache_dir(), "model", "clip")
        if not PathManager.exists(dir_path):
            PathManager.mkdirs(dir_path)

        path = os.path.join(dir_path, filename)
        if not is_main():
            synchronize()
            return path
        else:
            download_file(url, dir_path, filename, redownload=False)
            synchronize()
        return path

    def forward(self, sample_list):
        assert (
            self.image_key in sample_list
        ), f"{self.image_key} must exist in sample_list"

        img_features = self.image_encoder(sample_list[self.image_key])
        img_features = img_features / img_features.norm(
            dim=-1, keepdim=True
        )  # shape: batch_size x emb_dim

        task = self.config.get("task", None)
        if task and task.type == "zero_shot":
            scores = img_features @ self.zeroshot_weights * 100.0
            return {
                "scores": scores,
                "losses": {"dummy_loss": torch.tensor(0.0, device=img_features.device)},
            }

        assert (
            self.text_key in sample_list
        ), f"{self.text_key} must exist in sample_list"

        text = sample_list[self.config.text_key]
        txt_features = self.text_encoder(text)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        temperature = torch.exp(-self.learned_temperature)
        clip_temperature = self.config.get("clip_temperature", True)
        if clip_temperature:
            temperature_clip_range = self.config.get(
                "temperature_clip_range", [1e-2, 100]
            )
            assert len(temperature_clip_range) == 2, "range needs to be a list/tuple"
            temperature = torch.clip(
                temperature, temperature_clip_range[0], temperature_clip_range[1]
            )

        return {
            "embedding_1": img_features,
            "embedding_2": txt_features,
            "temperature": temperature
        }

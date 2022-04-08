# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List, Optional, Tuple

import torch
import yaml
from mmf.utils.build import build_model
from mmf.utils.file_io import PathManager
from omegaconf import OmegaConf
from torch import nn


try:
    from pytext.data.xlm_tensorizer import XLMTensorizer, XLMTensorizerScriptImpl
    from pytext.fb_core.data.monkeypatches import (
        patch_tokenizer_torchscript_implementations,
    )
    from pytext.torchscript.utils import ScriptBatchInput

    patch_tokenizer_torchscript_implementations()
except ImportError:

    class XLMTensorizer:
        Config = None

    class XLMTensorizerScriptImpl:
        pass

    ScriptBatchInput = None


class SentencePieceTensorizerScriptImpl(XLMTensorizerScriptImpl):
    """Sentece piece implementation will append `pad` token to `max_seq_len`,
    while XLM implementation doesn't have this step. In this numberize, all
    the return lists will have length `max_seq_len`.
    """

    def numberize(
        self,
        per_sentence_tokens: List[List[Tuple[str, int, int]]],
        unused_per_sentence_languages: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int], int, List[int]]:
        tokens: List[int] = []
        segment_labels: List[int] = []  # e.g language_ids
        seq_len: int = 0
        positions: List[int] = []
        max_seq_len: int = self.max_seq_len // len(per_sentence_tokens)

        for idx, single_sentence_tokens in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(
                single_sentence_tokens, max_seq_len=max_seq_len
            )[0]
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)

            tokens.extend(lookup_ids)

        # Change the first token to `bos` token
        tokens[0] = self.vocab.bos_idx

        # fill in `pad` token to `max_seq_len`
        seq_len = len(tokens)
        if seq_len < self.max_seq_len:
            tokens += [self.vocab.pad_idx] * (self.max_seq_len - seq_len)

        segment_labels = [0] * self.max_seq_len
        positions = list(range(self.max_seq_len))
        return tokens, segment_labels, seq_len, positions

    def tensorize(
        self,
        tokens_2d: List[List[int]],
        segment_labels_2d: List[List[int]],
        seq_lens_1d: List[int],
        positions_2d: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert instance level vectors into batch level tensors.
        No need to pad because `self.numberize` has already done that.
        """
        tokens = torch.tensor(tokens_2d).long()
        pad_mask = tokens.ne(self.vocab.pad_idx).long()
        segment_labels = torch.tensor(segment_labels_2d).long()
        positions = torch.tensor(positions_2d).long()
        return tokens, pad_mask, segment_labels, positions


class SentencePieceTensorizer(XLMTensorizer):
    """Sentence piece tokenizer has different special tokens than XLM tokenizer,
    and it uses different `dictionary_class`. Other parts are the same.
    """

    __TENSORIZER_SCRIPT_IMPL__ = SentencePieceTensorizerScriptImpl

    @classmethod
    def from_config(cls, config: XLMTensorizer.Config):
        from pytext.common.constants import SpecialTokens
        from pytext.config.component import ComponentType, create_component
        from pytext.data.bert_tensorizer import build_fairseq_vocab

        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        with PathManager.open(config.vocab_file) as file_path:
            vocab = build_fairseq_vocab(
                vocab_file=file_path,
                max_vocab=config.max_vocab,
                min_count=config.min_count,
                special_token_replacements={
                    "<unk>": SpecialTokens.UNK,
                    "<pad>": SpecialTokens.PAD,
                    "</s>": SpecialTokens.EOS,
                    "<s>": SpecialTokens.BOS,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            language_column=config.language_column,
            lang2id=config.lang2id,
            use_language_embeddings=config.use_language_embeddings,
            has_language_in_data=config.has_language_in_data,
        )


class MMFTInterface(nn.Module):
    """Interface for MMF Transformer models
    Build MMF Transformer entirely from yaml config, and if `checkpoint_path`
    is presented, interface will also load the saved model.
    """

    def __init__(self, yaml_path: str, checkpoint_path: str = ""):
        super().__init__()
        self.device = "cpu"

        self.config = self.__load_yaml_config(yaml_path)
        self.__setup_model()
        if checkpoint_path:
            self.__load_pretrained_model(checkpoint_path)
        self.__setup_processors()

    def __load_yaml_config(self, yaml_path):
        with PathManager.open(yaml_path) as f:
            config = yaml.load(f)

        config = OmegaConf.create(config)
        return config

    def __load_pretrained_model(self, checkpoint_path):
        with PathManager.open(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=lambda storage, loc: storage)

        assert (
            "model" in state_dict
        ), "Key `model` should be in the checkpoint of MMF models"
        ckpt = state_dict["model"]

        # Delete any loss-related parameters, since `self.model.losses` has been deleted
        for k in ckpt:
            if k.startswith("losses"):
                del ckpt[k]

        self.model.load_state_dict(ckpt)

    def __setup_model(self):
        config = self.config
        model_config = config.model_config[config.model]
        self.model = build_model(model_config)

        # loss_fn is not torchscriptable. The line is
        # `mmf/modules/losses.py: line 572`
        if hasattr(self.model, "losses"):
            delattr(self.model, "losses")

    def to(self, device: str):
        self.device = device
        self.model.to(self.device)

    def __setup_processors(self):
        from pytext.data.tokenizers import SentencePieceTokenizer

        config = self.config
        dataset_name = config.datasets
        dataset_config = config.dataset_config[dataset_name]
        processors = dataset_config.processors.batch_processor.params.processors
        tokenizer_params = processors.text_processor.params
        tokenizer_config = tokenizer_params.tokenizer_config.params

        model_path = tokenizer_config.model_path
        vocab_path = tokenizer_config.vocab_path
        max_seq_length = tokenizer_params.max_seq_length

        # pytext and fairseq backend have different behaviors and need to set
        # different `max_vocab`. pytext has `max_vocab` 250002 and should change
        # to -1 to pass inference test.
        max_vocab = tokenizer_config.max_vocab
        if max_vocab == 250002:
            max_vocab = -1

        # TODO: support more tensorizers based on the type of transformer used
        # from pytext.
        sp_config = SentencePieceTensorizer.Config(
            vocab_file=vocab_path,
            max_seq_len=max_seq_length,
            max_vocab=max_vocab,
            min_count=-1,
            tokenizer=SentencePieceTokenizer.Config(sp_model_path=model_path),
        )
        tensorizer = SentencePieceTensorizer.from_config(sp_config)
        self.text_processor = tensorizer.torchscriptify()

    def prepare_input_samples(
        self,
        features: Dict[str, torch.Tensor],
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        # Process text input, the texts in `ScriptBatchInput` is a list of
        # string list. Outer list represents batch, inner list represents
        # multiple texts like title and description and concatenates them.
        batched_texts = [[t] for t in texts]
        text = self.text_processor(
            ScriptBatchInput(texts=batched_texts, tokens=None, languages=None)
        )

        # Set up text modality
        sample = {
            "input_ids": text[0].to(self.device),
            "input_mask": text[1].to(self.device),
            "segment_ids": text[2].to(self.device),
        }

        # Process feature input, the final dimension should be
        # [batch_size, num_tokens, feature_dim]. If feature.dim() == 1, add
        # `batch_size` dimension first. If feature.dim() == 2 (assume
        # [batch_size, feature_dim]), add `num_tokens` dimension.
        for name, feature in features.items():
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            if feature.dim() == 2:
                feature = feature.unsqueeze(1)

            sample[name] = feature.to(self.device)

        return sample

    def post_process_output(
        self,
        model_output: Dict[str, torch.Tensor],
        output_blobs: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Check values in `output_blobs`
        if output_blobs is None:
            output_blobs = ["label", "confidence"]

        outputs = {}

        # Multihead MLP will have different score keys
        if "scores" in model_output:
            scores = nn.functional.softmax(model_output["scores"], dim=1)
            confidence, label = torch.max(scores, dim=1)
            outputs = {"label": label, "confidence": confidence}
        else:
            # Assign zero tensor for torchscriptable instead of None because
            # it should be tensor type
            scores = torch.zeros(1, dtype=torch.float32)
            model_output["scores"] = scores

        for blob in output_blobs:
            if blob == "logits":
                outputs[blob] = model_output["scores"]
            elif blob == "scores":
                outputs[blob] = scores
            elif blob in model_output:
                outputs[blob] = model_output[blob]

        return outputs

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        texts: List[str],
        output_blobs: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """MMFT interface accepts any modality feature embedding and raw text string
        as input to predict labels. `output_blobs` is used to get different model
        outputs, by default, `label` and `confidence`. Raw `logits` from classifier
        layer and `embedding` from Transformer encoder trunk are also supported.
        """
        sample = self.prepare_input_samples(features, texts)
        model_output = self.model(sample)
        outputs = self.post_process_output(model_output, output_blobs)
        return outputs

    @torch.jit.ignore
    def torchscriptify(self):
        return torch.jit.script(self)

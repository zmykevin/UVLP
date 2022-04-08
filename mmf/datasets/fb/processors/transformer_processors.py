# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Union

from mmf.common.registry import registry
from mmf.datasets.processors.bert_processors import (
    MaskedRobertaTokenizer,
    RobertaTokenizer,
    MultiSentenceRobertaTokenizer,
)


@registry.register_processor("masked_spm_tokenizer")
class MaskedSPMTokenizer(MaskedRobertaTokenizer):
    """Masked version of sentencepiece tokenizer wrapper for Pytext models"""

    def __init__(self, config, *args, **kwargs):
        from multimo.text.tokenizers import SentencePieceTokenizer

        assert (
            config.tokenizer_config.type == "pytext_spm"
        ), "Change tokenizer_config.type to pytext_spm"
        tokenizer_params = config.tokenizer_config.params
        self._tokenizer = SentencePieceTokenizer(
            sp_model_path=tokenizer_params.model_path,
            sp_vocab_path=tokenizer_params.vocab_path,
        )

        self._CLS_TOKEN = tokenizer_params.get("bos_token", "<s>")
        self._SEP_TOKEN = tokenizer_params.get("sep_token", "</s>")
        self._MASK_TOKEN = tokenizer_params.get("mask_token", "<mask>")
        self._PAD_TOKEN_ID = tokenizer_params.get("padding_idx", 1)
        self._vocab = self._tokenizer.get_vocab(max_size=tokenizer_params.max_vocab)

        # vocab.min_sequence_len is set to 2 in case of empty sentence
        self._vocab.min_sequence_len = 2

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, "mask_probability", 0.15)

    def get_vocab_size(self) -> int:
        return self._vocab.__len__()

    def tokenize(self, tokens: Union[str, List[str]]) -> List[str]:
        # we remove <s> and </s> from SPM tokenized results for now.
        # and will add back later in `_convert_to_indices` for easier
        # inheritance of MaskedTokenProcessor.
        return self._tokenizer(tokens)[1:-1]

    def _convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._vocab([tokens]).tolist()[0]
        else:
            return self._vocab(tokens).tolist()

    def _convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        return self._vocab.decode(ids)


@registry.register_processor("spm_tokenizer")
class SPMTokenizer(RobertaTokenizer, MaskedSPMTokenizer):
    """Sentencepiece tokenizer wrapper for Pytext models"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = config.get("mask_probability", 0)


@registry.register_processor("multi_sentence_spm_tokenizer")
class MultiSentenceSPMTokenizer(MultiSentenceRobertaTokenizer):
    """Extension of SPMTokenizer which supports multiple sentences.
    Similar to MultiSentenceRobertaTokenizer.
    """

    def __init__(self, config, *args, **kwargs):
        self.fusion_strategy = config.get("fusion", "concat")
        self.tokenizer = SPMTokenizer(config, *args, **kwargs)
        self._probability = config.get("mask_probability", 0)

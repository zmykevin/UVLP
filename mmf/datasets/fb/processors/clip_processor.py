# Copyright (c) Facebook, Inc. and its affiliates.
# Implementation adapted from:
# https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

import gzip
import html
import os
import warnings
from functools import lru_cache
from typing import Optional, Tuple, Set, Dict, List, Any

import ftfy
import regex as re
import torch
from mmf.common.constants import CLIP_VOCAB_CONSTS
from mmf.common.registry import registry
from mmf.datasets.processors import BaseProcessor
from mmf.utils.configuration import get_mmf_env
from mmf.utils.download import DownloadableFile
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_mmf_root
from omegaconf import DictConfig


@lru_cache()
def get_vocab_file(resource_config: Optional[DictConfig] = None) -> str:
    """ Get vocabulary file, if no input provided, return the default file.

    Args:
        resource_config (DictConfig): a config object which has three keys.
        `url`, it is the link to a local file or manifold path or download url
        `file_name`, it is the name of the file
        `hashcode`, this is the sha256 checksum for the file

    Returns:
        local_filename (str): downloaded local file path
    """
    if not resource_config or resource_config.get("url", None) is None:
        resource_config = CLIP_VOCAB_CONSTS
        warnings.warn(
            f"Processor vocab url is not provided, using: {CLIP_VOCAB_CONSTS}"
        )
    url = resource_config.get("url")
    if PathManager.isfile(url):
        return url

    resource = DownloadableFile(
        **resource_config,
        compressed=False,
    )
    download_folder = os.path.join(get_mmf_root(), get_mmf_env(key="data_dir"))
    parts = url.split("/")
    default_filename = parts[-1] if len(parts) > 0 else "vocab_file.txt"
    file_name = resource_config.get("file_name", default_filename)
    resource.download_file(download_folder)
    return os.path.join(download_folder, file_name)

@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want
    to avoid UNKs. When you're at something like a 10B token dataset you end up
    needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple) -> Set[Tuple[str, str]]:
    """ Get consecutive symbol pairs in a word

    Args:
        word (str): represented as tuple of symbols
        Symbols are variable-length strings. See bpe documentation:
        https://huggingface.co/transformers/tokenizer_summary.html#byte-pair-encoding-bpe

    Returns:
        pairs (Set[Tuple[str, str]]): set of consecutive symbol pairs in a word
    """
    return set(zip(word[:-1], word[1:]))


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class CLIPTokenizer:
    def __init__(self, bpe_path: str):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with PathManager.open(bpe_path, "rb") as f:
            gz = gzip.GzipFile(fileobj=f, mode="rb")
            merges = gz.read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|
            'll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def vocab_length(self) -> int:
        return len(list(self.encoder.keys()))


@registry.register_processor("clip_text_processor")
class CLIPProccessor(BaseProcessor):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config)
        file_path = get_vocab_file(config.get("vocab_file", None))
        self._tokenizer = CLIPTokenizer(file_path)
        self.sot_token = self._tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self._tokenizer.encoder["<|endoftext|>"]
        self.context_length = config.get("max_seq_length", 77)
        self.text_key = config.get("text_key", "text")

    def __call__(self, item: Any) -> Dict[str, torch.Tensor]:
        text = item[self.text_key]
        if isinstance(text, list) and len(text) >= 1:
            text = text[0]

        # ensure token length <= self.context_length, truncate if needed
        all_tokens = self._tokenizer.encode(text)
        all_tokens = all_tokens[: (min(len(all_tokens), self.context_length - 2))]
        all_tokens = [self.sot_token] + all_tokens + [self.eot_token]
        assert len(all_tokens) <= self.context_length

        result = torch.zeros(self.context_length, dtype=torch.long)
        result[: len(all_tokens)] = torch.tensor(all_tokens)

        return {"text": result}

    def get_vocab_size(self) -> int:
        return self._tokenizer.vocab_length()

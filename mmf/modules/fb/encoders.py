# Copyright (c) Facebook, Inc. and its affiliates.

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mmf.common.registry import registry
from mmf.modules.encoders import Encoder, TextEncoderFactory as TEF
from omegaconf import MISSING


try:
    from faim.encoder.non_temporal.text import DoCNNEncoder as DE
except ImportError:

    class DE:
        pass


class TextEncoderFactory(TEF):
    """The same TextEncoderFactory as in mmf/mopdules/encoders.py, except that it
    accept docnn_encoder, which is an internal text encoder model.

    If `config.type` is "docnn", it will initialize with `DoCNNEncoder`,
    otherwise, call its parent class to initialize.
    """

    def __init__(self, config: TEF.Config, *args, **kwargs):
        self._type = config.type
        if self._type == "docnn":
            _config = copy.deepcopy(config)
            _config.type = "identity"  # type as placeholder
            super().__init__(_config)
            self.module = DoCNNEncoder(config)
        else:
            super().__init__(config)


@registry.register_encoder("faim_docnn")
class DoCNNEncoder(DE, Encoder):
    """Similar DoCNNEncoder in faim/encoder/non_temporal/text/docnn_encoder.py.
    This class uses pytext transformer word embedding as embedding layer. The
    rest parts are the same as parent class.
    To make it torchscript, its forward function doesn't call `super().forward`.
    """

    @dataclass
    class Config(Encoder.Config):
        name: str = "faim_docnn"
        pytext_params: Dict[str, Any] = MISSING
        conv_filters: List[Tuple[int, int]] = MISSING
        mlp_sizes: List[int] = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        from faim.module.common.embedding import Embedding
        from pytext.fb_core.utils.manifold_utils import register_manifold_handler
        from pytext.models.roberta import RoBERTaEncoder

        register_manifold_handler()

        module = RoBERTaEncoder.from_config(
            config=RoBERTaEncoder.Config(**config.pytext_params)
        )
        word_embeddings = module.encoder.transformer.token_embedding
        num_embeddings, embedding_dim = word_embeddings.weight.size()
        embedding = Embedding(
            num_embeddings, embedding_dim, nn_embedding=word_embeddings
        )

        conv_filters = config.conv_filters
        mlp_sizes = config.mlp_sizes
        super().__init__("text", embedding, conv_filters, mlp_sizes)

    def forward(self, *args, **kwargs):
        # Cannot be torchscripted if using super().forward
        # copy from faim/encoder/non_temporal/text/docnn_encoder.py
        x = args[0]
        if self.use_hash:
            # here assumes that zero is padding index
            x = x % (self.cadinality - 1) + 1
        out = self.module(x)
        return out

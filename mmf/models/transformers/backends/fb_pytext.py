# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Dict, List, Type

import torch
from mmf.common.registry import registry
from mmf.models.transformers.backends.huggingface import HuggingfaceEmbeddings
from mmf.models.transformers.base import BaseTransformer, BaseTransformerBackend
from torch import Tensor, nn


class FBPytextEmbeddings(HuggingfaceEmbeddings):
    def init_weights(self, transformer: Type[nn.Module]):
        for idx, modality in enumerate(self.model_config.modalities):
            if modality.type == "text":
                if modality.get("consume_raw", True):
                    self.token_embeddings[idx] = transformer.transformer.token_embedding
                    self.layer_norms[idx] = transformer.transformer.embedding_layer_norm
                else:
                    del transformer.transformer.token_embedding

            if idx == 0:
                # save memory by sharing positional embeddings with transformer
                pos_embeddings = transformer.transformer.positional_embedding
            else:
                pos_embeddings = deepcopy(transformer.transformer.positional_embedding)
            self.pos_embeddings[idx] = pos_embeddings


@registry.register_transformer_backend("fb_pytext")
class FBPytextBackend(BaseTransformerBackend):
    """Transformer backend with pytext transformer models"""

    def __init__(self, config: BaseTransformer.Config, *args, **kwargs):
        from pytext.fb_core.utils.manifold_utils import register_manifold_handler

        # Register pytext manifold handlers
        register_manifold_handler()

        super().__init__(config)

    def build_transformer_config(self):
        """Build the transformer base model config."""
        self.backend_params = self.config.backend.get("params", {})

    def build_transformer_base(self):
        from pytext.models.roberta import RoBERTaEncoder

        """Build the transformer base model."""
        module = RoBERTaEncoder.from_config(
            config=RoBERTaEncoder.Config(
                embedding_dim=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                num_encoder_layers=self.config.num_hidden_layers,
                num_attention_heads=self.config.num_attention_heads,
                **self.backend_params,
            )
        )
        self.transformer = module.encoder

    def build_embeddings(self):
        """Build the multimodal embeddings using the transformer base
        embeddings.
        """
        self.embeddings = FBPytextEmbeddings(self.config, self.config, self.transformer)

    def get_config(self):
        """Return the transformer configuration."""
        return self.config

    def generate_embeddings(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        """Generate multimodal embeddings."""
        # Segment IDs not used in pytext model embeddings
        embedding = self.embeddings(
            tokens_ids=tokens_ids, position_ids=position_ids, segment_ids={}
        )
        embedding = embedding * (1 - attention_mask.unsqueeze(-1).type_as(embedding))

        # B x T x C -> T x B x C
        embedding = embedding.transpose(0, 1)

        return embedding

    def generate_attention_mask(self, masks: List[Tensor]) -> Tensor:
        """Generate attention mask."""
        attention_mask = torch.cat(masks, dim=-1)
        # Invert the attention mask tensor. Pytext requires valid mask
        # values to be 0
        attention_mask = ~(attention_mask.to(torch.bool))

        return attention_mask

    def generate_encoded_layers(self, embedding, attention_mask) -> List[Tensor]:
        """Generate the output from transformer layers. Return the encoded layers."""
        encoded = embedding
        states = [encoded]

        for layer in self.transformer.transformer.layers:
            encoded = layer(encoded, attention_mask)
            states.append(encoded)

        encoded_layers: List[Tensor] = []
        for state in states:
            encoded_layers.append(state.transpose(0, 1))

        return encoded_layers

# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Dict, List, Optional, Type

import torch
from mmf.common.registry import registry
from mmf.models.transformers.backends.huggingface import HuggingfaceEmbeddings
from mmf.models.transformers.base import BaseTransformerBackend
from mmf.utils.file_io import PathManager
from torch import Tensor, nn


class FBFairseqEmbeddings(HuggingfaceEmbeddings):
    def init_weights(self, transformer: Type[nn.Module]):
        for idx, modality in enumerate(self.model_config.modalities):
            if modality.type == "text":
                self.token_embeddings[idx] = transformer.embed_tokens
                self.layer_norms[idx] = transformer.emb_layer_norm

            self.pos_embeddings[idx] = deepcopy(transformer.embed_positions)


@registry.register_transformer_backend("fb_fairseq")
class FBFairseqBackend(BaseTransformerBackend):
    """Transformer backend with fairseq transformer models"""

    def build_transformer_config(self):
        self.backend_params = self.config.backend.get("params", {})
        self.pretrained_model_path = self.config.backend.get("model_path", None)

    def build_transformer_base(self):
        from fairseq.modules import TransformerSentenceEncoder

        self.transformer = TransformerSentenceEncoder(
            padding_idx=self.config.pad_token_id,
            vocab_size=self.config.vocab_size,
            num_encoder_layers=self.config.num_hidden_layers,
            embedding_dim=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            activation_fn=self.config.hidden_act,
            **self.backend_params,
        )
        if self.pretrained_model_path:
            with PathManager.open(self.pretrained_model_path, "rb") as f:
                encoder_weights = torch.load(f, map_location=torch.device("cpu"))
                encoder_weights = {
                    k.replace("sentence_encoder.", ""): v
                    for k, v in encoder_weights.items()
                }
            # TODO: Check if we can call `upgrade_state_dict` on Fairseq submodules
            # before loading the state dict to make old models compatible
            self.transformer.load_state_dict(encoder_weights)

        # TransformerSentenceEncoder `forward` is not scriptable(only traceable).
        # We replace it here with a dummy forward method that is a scriptable.
        self.transformer.forward = self._dummy_base_forward

    def build_embeddings(self):
        self.embeddings = FBFairseqEmbeddings(
            self.config, self.config, self.transformer
        )

    def _dummy_base_forward(
        self,
        tokens: torch.Tensor,
        segment_labels: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ):
        return torch.zeros(1), torch.zeros(1)

    def get_config(self):
        return self.config

    def generate_embeddings(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        # Segment IDs not used in pytext model embeddings
        embedding = self.embeddings(
            tokens_ids=tokens_ids, position_ids=position_ids, segment_ids={}
        )
        embedding = embedding * (1 - attention_mask.unsqueeze(-1).type_as(embedding))

        # B x T x C -> T x B x C
        embedding = embedding.transpose(0, 1)

        return embedding

    def generate_attention_mask(self, masks: List[Tensor]) -> Tensor:
        attention_mask = torch.cat(masks, dim=-1)
        # Invert the attention mask tensor. Fairseq requires valid mask
        # values to be 0
        attention_mask = ~(attention_mask.to(torch.bool))

        return attention_mask

    def generate_encoded_layers(self, embedding, attention_mask) -> List[Tensor]:
        encoded = embedding
        states = [encoded]
        for layer in self.transformer.layers:
            encoded, _ = layer(encoded, self_attn_padding_mask=attention_mask)
            states.append(encoded)

        encoded_layers: List[Tensor] = []
        for state in states:
            encoded_layers.append(state.transpose(0, 1))

        return encoded_layers

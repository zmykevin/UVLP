# Copyright (c) Facebook, Inc. and its affiliates.
# Code adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from mmf.common.registry import registry
from mmf.modules.bottleneck import AvgPoolBottleneck
from mmf.modules.encoders import Encoder
from mmf.modules.layers import AttnPool2d
from mmf.utils.distributed import synchronize
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_mmf_root
from omegaconf import OmegaConf
from torch import nn


class ClipTransformer(nn.Module):
    """
    Clip Implementation of Transformer
    https://arxiv.org/abs/2103.00020
    """

    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class ClipResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1,
      with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions,
      where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttnPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        layers = [AvgPoolBottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * AvgPoolBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(AvgPoolBottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class FloatLayerNorm(torch.nn.LayerNorm):
    """
    Subclass torch's LayerNorm to force perform LayerNorm
    in float form. Reverting to the previous dtype afterwards.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    Using Gaussian Error Linear Units:
    https://arxiv.org/pdf/1606.08415.pdf,
    where the 1.702 value comes from.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = FloatLayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = FloatLayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ClipVisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = FloatLayerNorm(width)

        self.transformer = ClipTransformer(width, layers, heads)

        self.ln_post = FloatLayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # x shape: torch.Size([batch, 768, 7, 7])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class ClipEncoder(Encoder):
    DEFAULT_MODEL = "ViT-B-32"

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def _get_pretrained_config(self, model_name: str, mode: str):
        import os
        from mmf.utils.file_io import PathManager

        variants_config_path = os.path.join("configs", "fb", "encoders", "clip")
        variants_config_path = os.path.join(get_mmf_root(), variants_config_path)
        variants = {}
        for variant_file in os.listdir(variants_config_path):
            key = variant_file.split(".")[0]
            with PathManager.open(
                os.path.join(variants_config_path, variant_file)
            ) as f:
                variants[key] = OmegaConf.load(f)
        variant_names = list(variants.keys())
        assert (
            model_name in variant_names
        ), f"pretrained model name {model_name} must be one of {variant_names}"
        return variants[model_name][mode]

    def _get_config(self, config, mode):
        default_config = OmegaConf.structured(self.Config)

        is_pretrained = config.is_pretrained
        if is_pretrained:
            encoder_model_name = config.get("pretrained_model_name", self.DEFAULT_MODEL)
            pretrained_config = self._get_pretrained_config(encoder_model_name, mode)
            default_config.update(**pretrained_config)

        default_config.update(**config)
        return default_config

    def _download(self, pretrained: Dict[str, str]) -> str:
        import os

        from mmf.utils.configuration import get_mmf_cache_dir
        from mmf.utils.download import DownloadableFile
        from mmf.utils.distributed import is_main

        url = pretrained.url

        if PathManager.isfile(url):
            return url

        default_filename = url.split("/")[-1] if len(url.split("/")) > 0 else url
        file_name = pretrained.get("file_name", default_filename)
        download_folder = os.path.join(get_mmf_cache_dir(), "model", "clip", "encoders")
        if not os.path.exists(download_folder):
            os.mkdir(download_folder)

        file_path = os.path.join(download_folder, file_name)
        if not is_main():
            synchronize()
            return file_path
        else:
            resource = DownloadableFile(**pretrained, compressed=False)
            resource.download_file(download_folder)
            synchronize()
        return file_path

    def _load_pretrained(self, pretrained: Dict[str, str]) -> None:
        state_dict_file = self._download(pretrained)
        with PathManager.open(state_dict_file, "rb") as f:
            state_dict = torch.load(f, map_location=next(self.parameters()).device)
            self.load_state_dict(state_dict)


@registry.register_encoder("clip_text")
class ClipTextEncoder(ClipEncoder):
    """Clip text encoder implementaton adapted from:
    https://github.com/openai/CLIP/blob/main/clip/model.py


    Pretrained ViT-B/32 can be downloaded here:
        https://dl.fbaipublicfiles.com/mmf/data/encoders/text_encoder-ViT-B-32.pt
    """

    @dataclass
    class Config(Encoder.Config):
        """
        You can specify the pretrained model name to be any of
        `ViT-B-32`, `RN50`, `RN101` and `RN50x4` and optionally override the checkpoint
        with your own model file by specifying `pretrained_model`, which takes
        the form of a dict including keys: `url`, `hashcode` (sha256 checksum), and
        `file_name`. If you do not specify your own model file, a default clip provided
        pretrained encoder checkpoint will be provided. If you wish to load a provided
        pretrained encoder, please do not specify any custom parameters.

        By default, if you do not pass in any params, it loads the ViT-B-32 encoder.

        Alternatievly, you can override all configuration of the encoder by directly
        setting the custom parameters, that is, if you wish to train from
        scratch with your own custom clip model. You need take the
        responsibility to set all the custom parameters correctly for your use case.
        You do not need to set the params: `pretrained_model_name` and
        `pretrained_model`.
        """

        name: str = "clip_text"
        # pretrained parameters
        pretrained_model_name: Optional[str] = "ViT-B-32"
        is_pretrained: bool = True
        pretrained_model: Optional[Dict[str, str]] = None
        freeze: bool = False

        # custom parameters
        embed_dim: int = 512
        vocab_size: int = 49408
        transformer_width: int = 512
        context_length: int = 77
        transformer_layers: int = 12
        transformer_heads: int = 8

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        config = self._get_config(config, "text")

        transformer_width = config.transformer_width
        context_length = config.context_length
        freeze = config.freeze

        self.transformer = ClipTransformer(
            width=transformer_width,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            attn_mask=self.build_attention_mask(context_length),
        )
        self.token_embedding = torch.nn.Embedding(config.vocab_size, transformer_width)
        self.positional_embedding = torch.nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
        self.ln_final = FloatLayerNorm(transformer_width)
        self.text_projection = torch.nn.Parameter(
            torch.empty(transformer_width, config.embed_dim)
        )

        pretrained_model = config.pretrained_model
        is_pretrained = config.is_pretrained
        if pretrained_model and is_pretrained:
            self._load_pretrained(pretrained_model)

            # ONLY check freeze if loaded as pretrained
            if freeze:
                self._freeze_parameters()
        else:
            # init all parameters properly
            self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, context_length: int) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision
        # tokens. Pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


@registry.register_encoder("clip_image")
class ClipImageEncoder(ClipEncoder):
    """Clip image encoder implementaton adapted from:
    https://github.com/openai/CLIP/blob/main/clip/model.py

    Pretrained ViT-B/32 can be downloaded here:
        https://dl.fbaipublicfiles.com/mmf/data/encoders/image_encoder-ViT-B-32.pt
    """

    @dataclass
    class Config(Encoder.Config):
        """
        You can specify the pretrained model name to be any of
        `ViT-B-32`, `RN50`, `RN101` and `RN50x4` and optionally override the checkpoint
        with your own model file by specifying `pretrained_model`, which takes
        the form of a dict including keys: `url`, `hashcode` (sha256 checksum), and
        `file_name`. If you do not specify your own model file, a default clip provided
        pretrained encoder checkpoint will be provided. If you wish to load a provided
        pretrained encoder, please do not specify any custom parameters.

        By default, if you do not pass in any params, it loads the ViT-B-32 encoder.

        Alternatievly, you can override all configuration of the encoder by directly
        setting the custom parameters, that is, if you wish to train from
        scratch with your own custom clip model. You need take the
        responsibility to set all the custom parameters correctly for your use case.
        You do not need to set the params: `pretrained_model_name` and
        `pretrained_model`.
        """

        name: str = "clip_image"

        # pretrained parameters
        pretrained_model_name: Optional[str] = "ViT-B-32"
        pretrained_model: Optional[Dict[str, str]] = None
        is_pretrained: bool = True
        freeze: bool = False

        # custom parameters
        backbone: str = "vit"
        embed_dim: int = 512
        num_vision_layers: Optional[int] = 12  # used for vit
        vision_layers: Optional[tuple] = (3, 6, 4, 3)  # used for resnet
        vision_width: int = 768
        vision_heads: int = 12
        vision_patch_size: Optional[int] = None
        image_resolution: int = 224

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        config = self._get_config(config, "image")

        backbone = config.backbone
        embed_dim = config.embed_dim
        vision_width = config.vision_width
        vision_heads = config.vision_heads
        image_resolution = config.image_resolution

        freeze = config.freeze

        if backbone == "modified_resnet":
            vision_layers = config.vision_layers
            self.visual = ClipResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        elif backbone == "vit":
            vision_layers = config.num_vision_layers
            vision_patch_size = config.vision_patch_size
            self.visual = ClipVisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )
        else:
            raise NotImplementedError("clip image encoder backbone not implemented")

        pretrained_model = config.pretrained_model
        is_pretrained = config.is_pretrained
        if pretrained_model and is_pretrained:
            self._load_pretrained(pretrained_model)

            # ONLY freeze if it is loaded as pretrained
            if freeze:
                self._freeze_parameters()
        else:
            self.initialize_parameters()

    def initialize_parameters(self):
        """
        Properly initialize all parameters
        """
        if isinstance(self.visual, ClipResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

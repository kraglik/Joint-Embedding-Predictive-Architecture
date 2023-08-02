from typing import Optional

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .transformer import Transformer
from .modules import (
    PatchEmbed,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dims: int = 256,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        input_channels: int = 3,
        class_token: bool = True,
        num_classes: Optional[int] = None,
        positional_embedding_type: str = "sincos",  # "sincos", "learned" or "none"
    ) -> None:
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.num_patches = num_patches
        self.num_classes = num_classes

        assert positional_embedding_type in ["sincos", "learned", "none"]
        assert (image_size // patch_size) * patch_size == image_size, "image size must be divisible by patch size"

        if positional_embedding_type == "sincos":
            positional_embedding = SinusoidalPositionalEncoding(
                dims=dims,
                max_len=num_patches + class_token,
            )

        elif positional_embedding_type == "learned":
            positional_embedding = LearnedPositionalEncoding(
                dims=dims,
                max_len=num_patches + class_token,
            )

        else:
            positional_embedding = None

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            output_dims=dims,
        )
        self.patch_embed_norm = nn.BatchNorm1d(dims)

        self.class_token = None

        if class_token:
            self.class_token = nn.Parameter(
                torch.normal(
                    mean=0,
                    std=init_std,
                    size=(1, 1, dims),
                    dtype=torch.float
                ),
                requires_grad=True,
            )
            trunc_normal_(self.class_token, std=init_std)

        self.transformer = Transformer(
            dims=dims,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            positional_embedding=positional_embedding,
        )

        if num_classes is not None:
            self.head = nn.Linear(dims, num_classes)

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        patch_embeddings = self.patch_embed(image)

        if self.class_token is not None:
            batch_size = patch_embeddings.shape[0]
            class_tokens = self.class_token.expand(batch_size, -1, -1)

            patch_embeddings = torch.cat((class_tokens, patch_embeddings), dim=1)

            if mask is not None:
                mask = torch.cat(
                    (
                        torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device),
                        mask
                    ),
                    dim=1
                )

        x = self.transformer(patch_embeddings, mask=mask)

        if self.class_token is not None:
            x = x[:, 0]

        if self.num_classes is not None:
            x = self.head(x)

        return x

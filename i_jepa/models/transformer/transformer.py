import math
from typing import Optional

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .modules import (
    TransformerLayer,
)


class Transformer(nn.Module):
    def __init__(
        self,
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
        positional_embedding: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.num_features = self.embed_dim = dims

        self.positional_embedding = positional_embedding

        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    dim=self.embed_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(self.embed_dim)
        self.token_norm = norm_layer(self.embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tokens

        x = self.pos_embed(x)

        if mask is not None:
            mask = mask.flatten(1)
            x = x[:, mask == 1, :]

        for block in self.blocks:
            x = block(x)

        return x

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attention.out.weight.data, layer_id + 1)
            rescale(layer.mlp.linear_2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

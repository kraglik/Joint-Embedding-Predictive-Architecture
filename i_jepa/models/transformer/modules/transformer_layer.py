from typing import Optional

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .mlp import MLP


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm_layer_1 = norm_layer(dim)
        self.attention = MultiHeadAttention(
            dim,
            heads=heads,
            bias=qkv_bias,
            dropout=attention_dropout,
            scale=qk_scale,
        )
        self.norm_layer_2 = norm_layer(dim)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dim,
            dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm_layer_1(x)
        att_x = self.attention(query=norm_x, key=norm_x, value=norm_x, mask=mask)

        x = x + att_x
        x = x + self.mlp(self.norm_layer_2(x))

        return x

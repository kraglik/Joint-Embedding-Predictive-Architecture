from typing import Optional

import torch
from torch import nn

from .softmax_one import SoftmaxOne


class MultiHeadLinearEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int,
        heads: int,
        head_dims: int,
        bias: bool,
    ) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.heads = heads
        self.head_dims = head_dims
        self.bias = bias

        self.linear = nn.Linear(input_dims, heads * head_dims, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.view(*x.shape[:-1], self.heads, self.head_dims)

        return x.transpose(-2, -3)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dims: int,
        heads: int = 4,
        dims: int = 256,
        dropout: float = 0.1,
        scale: Optional[float] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.head_dims = dims // heads
        self.heads = heads

        self.query = MultiHeadLinearEncoder(input_dims, heads, self.head_dims, bias)
        self.key = MultiHeadLinearEncoder(input_dims, heads, self.head_dims, bias)
        self.value = MultiHeadLinearEncoder(input_dims, heads, self.head_dims, bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = scale or self.head_dims ** -0.5

        self.out = nn.Linear(dims, input_dims)
        self.softmax = SoftmaxOne(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = self.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .view(
                query.shape[0],
                -1,
                self.heads * self.head_dims,
            )
        )

        return self.out(out)

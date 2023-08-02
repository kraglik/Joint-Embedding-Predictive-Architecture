import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        dropout: float,
        activation=nn.GELU,
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(input_dims, hidden_dims)
        self.activation = activation()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dims, output_dims)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x

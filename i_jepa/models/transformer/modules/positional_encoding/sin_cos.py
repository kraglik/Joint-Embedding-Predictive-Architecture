import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dims, max_len):
        super().__init__()

        pe = torch.zeros(max_len, dims)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dims, 2) * -(math.log(10000.0) / dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

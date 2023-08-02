import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dims, max_len):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, dims))
        nn.init.xavier_uniform_(self.pos_enc.data)

    def forward(self, x):
        return x + self.pos_enc

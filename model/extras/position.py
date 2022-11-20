"""
Sinusoidal position encoding

Copy-paste from https://github.com/pytorch/tutorials/blob/04e1ba925501c2195abb3f42d494202a187e3c46/beginner_source/transformer_tutorial.py#L106 with a modification.

"""

import torch
import math
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import pdb

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = rearrange(pe, 's b c -> b s c')
        self.register_buffer('pos_table', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pos_table[:, :x.size(0)]
        return self.dropout(x)



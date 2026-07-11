import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim) * -emb)  # [half_dim]
        self.register_buffer("emb", emb)

    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]  # [bsz, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [bsz, dim]
        return emb

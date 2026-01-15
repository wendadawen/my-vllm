import torch
from torch import nn
import torch.nn.functional as F

class SiLUAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return F.silu(x0) * x1
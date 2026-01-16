import torch
from torch import nn
from src.layers.cuda import cuda_ext_lib

class SiLUAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        output = torch.empty(x0.shape, dtype=x0.dtype, device=x0.device)
        cuda_ext_lib.silu_and_mul_bf16(output, x0, x1)
        return output
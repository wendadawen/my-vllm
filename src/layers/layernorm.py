import torch
from torch import nn
import torch.nn.functional as F
from src.layers.cuda import cuda_ext_lib

class RMSNorm(nn.Module):

    def __init__(self, N: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            cuda_ext_lib.rmsnorm_bf16(output, x, self.weight.data, self.eps)
            return output
        else:
            cuda_ext_lib.rmsnorm_fused_add_inplace_bf16(x, residual, self.weight.data, self.eps)
            return x, residual
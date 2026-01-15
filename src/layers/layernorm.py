import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, N: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N))
        self.eps = eps
        self.normalized_shape = [N]

    # @torch.compile
    # def rms_forward(
    #     self,
    #     x: torch.Tensor,
    # ) -> torch.Tensor:
    #     orig_dtype = x.dtype
    #     x = x.float()
    #     var = x.pow(2).mean(dim=-1, keepdim=True)
    #     x.mul_(torch.rsqrt(var + self.eps))
    #     x = x.to(orig_dtype).mul_(self.weight)
    #     return x

    # @torch.compile
    # def add_rms_forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     orig_dtype = x.dtype
    #     x = x.float().add_(residual.float())
    #     residual = x.to(orig_dtype)
    #     var = x.pow(2).mean(dim=-1, keepdim=True)
    #     x.mul_(torch.rsqrt(var + self.eps))
    #     x = x.to(orig_dtype).mul_(self.weight)
    #     return x, residual

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: torch.Tensor | None = None,
    # ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    #     if residual is None:
    #         return self.rms_forward(x)
    #     else:
    #         return self.add_rms_forward(x, residual)

    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    @torch.compile
    def rms_norm_with_residual(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + residual
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_norm(x)
        else:
            return self.rms_norm_with_residual(x, residual)
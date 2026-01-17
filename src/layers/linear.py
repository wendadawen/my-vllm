import torch
from torch import nn
import torch.nn.functional as F
from src.layers.cuda import cuda_ext_lib

class Linear(nn.Module):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        assert bias == False
        if bias is True:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)  # vllm未量化采用的就是PyTorch的linear实现

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     out_shape = (*x.shape[:-1], self.weight.shape[-2])

    #     output = torch.empty(
    #         out_shape,
    #         device=x.device,
    #         dtype=x.dtype
    #     )
    #     cuda_ext_lib.linear_bf16(x, self.weight.data, output)
    #     return output

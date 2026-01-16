import torch
from torch import nn
import torch.nn.functional as F
from src.layers.cuda import cuda_ext_lib

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        

    # def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     return F.embedding(input_ids, self.weight)  # vllm采用的就是PyTorch的embedding实现
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = torch.empty(input_ids.numel(), self.weight.data.size(-1), device=input_ids.device, dtype=self.weight.data.dtype)
        cuda_ext_lib.embedding_bf16(output, self.weight.data, input_ids)
        return output
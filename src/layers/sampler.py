import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # @torch.compile
    # def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    #     logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    #     probs = torch.softmax(logits, dim=-1)
    #     sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    #     return sample_tokens
    
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        gumbel_noise = torch.ones_like(probs)  # 固定值1，无任何随机
        sample_tokens = probs.div(gumbel_noise).argmax(dim=-1)
        return sample_tokens
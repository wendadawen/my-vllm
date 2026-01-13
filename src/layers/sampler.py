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
        
        # ========== 核心修改：固定噪声，移除随机 ==========
        # 方案1：用全1张量 → 最简单，效果好，推荐
        gumbel_noise = torch.ones_like(probs)  # 固定值1，无任何随机
        # 方案1备选：用极小固定值，和你原代码的clamp_min保持一致，效果等价
        # gumbel_noise = torch.full_like(probs, fill_value=1e-10)
        
        sample_tokens = probs.div(gumbel_noise).argmax(dim=-1)
        return sample_tokens
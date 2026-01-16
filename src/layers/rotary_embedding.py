from functools import lru_cache
import torch
from torch import nn
from src.layers.cuda import cuda_ext_lib

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)  # Qwen3的rope实现有点不同，不是通常的相邻对旋转，而是 x[0] 与 x[head_dim/2]旋转
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 挺复杂的一个函数，这里是外积
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # @torch.compile
    # def forward(
    #     self,
    #     positions: torch.Tensor,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     cos_sin = self.cos_sin_cache[positions]
    #     cos, sin = cos_sin.chunk(2, dim=-1)
    #     query = apply_rotary_emb(query, cos, sin)
    #     key = apply_rotary_emb(key, cos, sin)
    #     return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cuda_ext_lib.rotary_embedding_inplace_bf16(positions, query, key, self.cos_sin_cache.data)
        return query, key

@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

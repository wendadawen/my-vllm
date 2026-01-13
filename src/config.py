
from transformers import AutoConfig
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str
    max_num_batched_tokens: int = 16384  # 一批次的最大 token 数量
    max_num_seqs: int = 512  # 一批次的最大数量
    max_model_len: int = 4096  # 手动指定的 prefill + decode 的最长token数量
    gpu_memory_utilization: float = 0.9
    hf_config: AutoConfig | None = None
    kvcache_block_size: int = 256  # 必须是 256 的倍数，因为 flash-attn 的 block kvcache 是这样规定的
    num_kvcache_blocks: int = -1  # 先 warmup，再根据 warmup 结果来确定
    cuda_id: int = 4  # 运行在哪个 GPU 设备上

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
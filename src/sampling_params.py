from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 4096  # 最大生成 token 数
    ignore_eos: bool = False  # 是否忽略 eos
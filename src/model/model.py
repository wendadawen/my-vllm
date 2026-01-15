from src.model.qwen3 import Qwen3ForCausalLM
from src.model.qwen3_opt import Qwen3ForCausalLM_OPT


def get_model_class(model_name: str):
    """根据 model_name 返回对应的模型类"""
    model_mapping = {
        "Qwen3-0.6B-OPT": Qwen3ForCausalLM_OPT,
        "Qwen3": Qwen3ForCausalLM,
    }
    if model_name in model_mapping:
        return model_mapping[model_name]
    return Qwen3ForCausalLM

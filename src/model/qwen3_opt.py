import torch
from torch import nn
from transformers import Qwen3Config

from src.layers.attention import Attention
from src.layers.rotary_embedding import get_rope
from src.layers.linear import Linear
from src.layers.activation import SiLUAndMul
from src.layers.layernorm import RMSNorm
from src.layers.embedding import Embedding
from src.utils.context import get_context

cn = 0
def print_tensor(s: str, tensor: torch.Tensor):
    return
    # if get_context().warm_up:
    #     return
    # global cn
    # cn += 1
    # if cn >= 100:
    #     return 
    # print(s)
    # cnt = 0
    # for x in tensor.flatten().tolist():
    #     cnt += 1
    #     print(x, end=" ")
    # print()
    

class Qwen3MLP(nn.Module):

    def __init__(self, config: Qwen3Config, layer_id: int):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        assert config.hidden_act == "silu"
        self.act_fn = SiLUAndMul()
        self.layer_id = str(layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor  # [num_batched_tokens, hidden_size]
    ) -> torch.Tensor:  # [num_bacthed_tokens, hidden_size]
        x0 = self.gate_proj(hidden_states)
        x1 = self.up_proj(hidden_states)
        print_tensor("layer: " + self.layer_id + " mlp:gate_proj", x0)
        print_tensor("layer: " + self.layer_id + " mlp:up_proj", x1)
        hidden_states = self.act_fn(x0, x1)
        print_tensor("layer: " + self.layer_id + " mlp:silu+mul", hidden_states)
        hidden_states = self.down_proj(hidden_states)
        print_tensor("layer: " + self.layer_id + " mlp:down_proj", hidden_states)
        return hidden_states


class Qwen3Attention(nn.Module):

    def __init__(self, config: Qwen3Config, layer_id: int):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = Linear(config.num_attention_heads * config.head_dim,config.hidden_size, bias=False)

        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        self.attn = Attention(config.num_attention_heads, config.head_dim, config.head_dim ** -0.5, config.num_key_value_heads)
        self.rotary_emb = get_rope(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.layer_id = str(layer_id)
    
    def forward(
        self,
        positions: torch.Tensor,  # [num_batched_tokens]
        hidden_states: torch.Tensor  # [num_batched_tokens, hidden_size]
    ) -> torch.tensor:  # [num_batched_tokens, hidden_size]
        k = self.k_proj(hidden_states)
        q = self.q_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        print_tensor("layer: " + self.layer_id + " self_attn:q_proj", q)
        print_tensor("layer: " + self.layer_id + " self_attn:k_proj", k)
        print_tensor("layer: " + self.layer_id + " self_attn:v_proj", v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        print_tensor("layer: " + self.layer_id + " self_attn:q_norm", q)
        print_tensor("layer: " + self.layer_id + " self_attn:k_norm", k)
        q, k = self.rotary_emb(positions, q, k)
        print_tensor("layer: " + self.layer_id + " self_attn:rotary_emb q", q)
        print_tensor("layer: " + self.layer_id + " self_attn:rotary_emb k", k)
        o = self.attn(q, k, v)
        print_tensor("layer: " + self.layer_id + " self_attn:attn", o)
        output = self.o_proj(o.flatten(1, -1))  # [num_batched_tokens, hidden_size]
        print_tensor("layer: " + self.layer_id + " self_attn:o_proj", output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, layer_id: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config, layer_id)
        self.layer_id = str(layer_id)
    
    def forward(
        self,
        positions: torch.Tensor,  # [num_batched_tokens]
        hidden_states: torch.Tensor,  # [num_batched_tokens, hidden_size]
        residual: torch.Tensor | None  # [num_batched_tokens, hidden_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # [num_batched_tokens, hidden_size], [num_batched_tokens, hidden_size]
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        print_tensor("layer: " + self.layer_id + " input_layernorm", hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        print_tensor("layer: " + self.layer_id + " self_attn", hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        print_tensor("layer: " + self.layer_id + " post_attention_layernorm", hidden_states)
        hidden_states = self.mlp(hidden_states)
        print_tensor("layer: " + self.layer_id + " mlp", hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [num_batched_tokens]
        positions: torch.Tensor   # [num_batched_tokens]
    ) -> torch.Tensor:  # [num_batched_tokens, hidden_size]
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM_OPT(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [num_batched_tokens]
        positions: torch.Tensor,  # [num_batched_tokens]
    ) -> torch.Tensor:  # [num_batched_tokens, hidden_size]
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor  # [num_batched_tokens, hidden_size]
    ) -> torch.Tensor:  # [batch_size, vocab_size]
        context = get_context()
        if context.is_prefill:  # 如果是 prefill，只需要最后一个 token 的输出，也就是生成下一个token说需要的 hidden_dim
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return self.lm_head(hidden_states)

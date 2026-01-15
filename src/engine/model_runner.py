import torch

from src.engine.sequence import Sequence
from src.layers.sampler import Sampler
from src.utils.context import set_context, reset_context, get_context
from src.utils.loader import load_model
from src.config import Config
from src.model.model import get_model_class


class ModelRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.dtype)
        torch.torch.set_default_device("cuda")
        torch.cuda.set_device(config.cuda_id)
        self.sampler = Sampler()
        model_class = get_model_class(config.model_name)
        self.model = model_class(config.hf_config)
        load_model(self.model, config.model_path)
        self.warmup()
        self.allocate_kv_cache_block()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def warmup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        get_context().warm_up = True
        self.run(seqs, True)
        get_context().warm_up = False
        torch.cuda.empty_cache()

    def allocate_kv_cache_block(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * config.kvcache_block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, config.kvcache_block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():  # 给模型适配上
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> list[list[int]]:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return block_tables
    
    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """
            return: 
                input_ids：一维，vLLM 的算子采用了特殊的设计，可以把所有的多个输入当做一个 batch 考虑，不需要考虑每个 batch 具体的序列长度，因此并不需要padding
                positions：一维
            需要准备这一批次的一些辅助数据，包括
                - input_ids: [seqLen_0 + seqLen_1 + .... + seqLen_n]
                - positions: 同 input_ids 维度
                - cu_seqlens_q: [batch_size + 1]，Query 序列的累积长度数组
                - cu_seqlens_k
                - max_seqlen_q: scale, 批量中单个序列本次需要 Prefill 的最大 token 长度
                - max_seqlen_k
                - slot_mapping: 同 input_ids 维度，大模型的 KV Cache 是一块连续的显存空间，被切分成固定大小的block，每个 block 有block_size个槽位，slot_mapping就是告诉模型：当前序列的某个 token，该从显存的哪个槽位读取/写入 KV 值
                - block_table：[batch_size, max_num_blocks_per_seq]
        """
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q, max_seqlen_k = 0, 0
        slot_mapping, block_tables = [], None
        context = get_context()
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if context.warm_up:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * Sequence.block_size
                if i == seq.num_blocks - 1:
                    end = start + seq.num_last_block_tokens
                else:
                    end = start + Sequence.block_size
                slot_mapping.extend(list(range(start, end)))
        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """
            return: 
                input_ids：一维
                positions：一维
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * Sequence.block_size + seq.num_last_block_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
    
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
            input_ids: 一维
            position：一维
            return：[batch_size, hidden_size]
        """
        context = get_context()
        return self.model.compute_logits(self.model(input_ids, positions))
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
            对于一批次的 seqs
            根据是否是 prefill 进行数据处理
            调用真正的 LLM 大模型进行一次推理
            并且将输出 logits 放入 sampler 选出这一批次生成的 token_ids
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions)
        temperatures = self.prepare_sample(seqs)
        token_ids = self.sampler(logits, temperatures).tolist()  # tensor 转 list，不然出错
        reset_context()
        return token_ids
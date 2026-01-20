from collections import deque
from transformers import AutoTokenizer

from src.sampling_params import SamplingParams
from src.engine.sequence import Sequence, SequenceStatus
from src.config import Config
from src.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: Config) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast = True)
        self.eos = self.tokenizer.eos_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # prefill or 换出
        self.running: deque[Sequence] = deque()  # decode

    def add_request(self, token_ids: list[int], sampling_params: SamplingParams) -> None:
        assert len(token_ids) < self.max_model_len
        seq = Sequence(token_ids, sampling_params)
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
            优先处理 prefill，再处理 decode
        """
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            seqlen = len(seq)
            if num_batched_tokens + seqlen > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            num_batched_tokens += seqlen - seq.num_cached_tokens
            self.block_manager.allocate(seq)
            scheduled_seqs.append(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                num_batched_tokens += 1
                self.block_manager.append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """
            在 LLMEngine 每执行完毕一批次的推理任务就需要调用这个函数
            判断该批次中的哪些 seq 已经完成了完整的推理
            完成了的 seq 需要在该函数释放资源
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or (seq.num_completion_tokens == seq.max_tokens) or (self.max_model_len == seq.num_tokens):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        
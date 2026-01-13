from transformers import AutoTokenizer

from src.sampling_params import SamplingParams
from src.engine.scheduler import Scheduler
from src.engine.sequence import Sequence
from src.engine.model_runner import ModelRunner
from src.config import Config


class LLMEngine:

    def __init__(
            self, 
            model_path: str
        ) -> None:
        config = Config(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        Sequence.block_size = config.kvcache_block_size
        self.model_runner = ModelRunner(config)
        self.scheduler = Scheduler(config)

    def step(self) -> list[tuple[int, list[int]]]:
        """
        - 调度器给出本批次需要执行的 seqs
        - 调用 ModelRunner 调用模型执行一批次的推理
        - 判断该批次有哪些 seqs 完成了 Prefill+Decode 的完整推理
            完成了推理的 seq 需要调用 scheduler.postprocess 来释放资源
            原因：使用的是 Continuous Batching，因此每一批次不会等到执行完毕，而是迭代级别的 Batching
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finised()]
        return outputs

    def generate(
        self, 
        prompts: list[str], 
        sampling_params: SamplingParams
    ) -> list[dict]:
        """
        - 将 prompts 的每条 prompt 形成调度器的调度的单位 request
        - 直到所有 requests 都被执行完毕
            - 调用 step 执行 request
        - 将结果按 seq_id 排序整理输出
        """
        outputs = {}
        for prompt in prompts:
            self.scheduler.add_request(prompt, sampling_params)
        while not self.scheduler.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        return outputs
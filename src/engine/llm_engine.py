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
        prompts: list[list[int] | str], 
        sampling_params: SamplingParams | list[SamplingParams]
    ) -> list[dict]:
        """
        - 将 prompts 的每条 prompt 形成调度器的调度的单位 request
        - 直到所有 requests 都被执行完毕
            - 调用 step 执行 request
        - 将结果按 seq_id 排序整理输出
        
        Args:
            prompts: 可以是 token id 列表的列表，或字符串列表（会自动转换为 token ids）
            sampling_params: 可以是单个 SamplingParams 或 SamplingParams 列表（每个请求一个）
        """
        # 处理 prompts：如果是字符串，转换为 token ids
        prompt_token_ids = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_token_ids.append(self.tokenizer.encode(prompt))
            else:
                prompt_token_ids.append(prompt)
        
        # 处理 sampling_params：如果是单个，转换为列表
        if isinstance(sampling_params, SamplingParams):
            sampling_params_list = [sampling_params] * len(prompt_token_ids)
        else:
            sampling_params_list = sampling_params
            assert len(sampling_params_list) == len(prompt_token_ids), \
                f"sampling_params 列表长度 ({len(sampling_params_list)}) 必须与 prompts 长度 ({len(prompt_token_ids)}) 相同"
        
        outputs = {}
        for token_ids, sp in zip(prompt_token_ids, sampling_params_list):
            self.scheduler.add_request(token_ids, sp)
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
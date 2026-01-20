import os
import time
from random import randint, seed
from src import LLM, SamplingParams
import sys

def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    DIRPATH = os.path.dirname(os.path.abspath(__file__))
    sys.stdout = open(os.path.join(DIRPATH, "output.txt"), "w", encoding="utf-8")
    model_path = os.path.join(DIRPATH, "huggingface/Qwen3-0.6B")
    llm = LLM(model_path)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
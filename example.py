from transformers import AutoTokenizer
from src import LLM, SamplingParams
import sys
import os

def main():
    # 打开文件，将标准输出重定向到文件
    DIRPATH = os.path.dirname(os.path.abspath(__file__))
    sys.stdout = open(os.path.join(DIRPATH, "output.txt"), "w", encoding="utf-8")
    model_path = os.path.join(DIRPATH, "huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True)
    llm = LLM(model_path)
    sampling_params = SamplingParams()

    prompts = [
        "asad" * 1,
        "asad" * 1 + "abcdefgh"
    ]

    prompts = [
        tokenizer.apply_chat_template(  # '<|im_start|>user\nasad<|im_end|>\n<|im_start|>assistant\n'
            [{"role": "user", "content": prompt}],
            tokenize = False,  # 不要转化为 token id
            add_generation_prompt = True  # 末尾加上assistant
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}") # !r：保留「特殊控制字符」，避免格式错乱
        print(f"Completion: {output["text"]!r}")

if __name__ == "__main__":
    main()
# my-vllm

一个自定义的大语言模型推理引擎实现，支持高效的批量生成和推理加速。

## 项目简介

本项目是一个轻量级的 LLM 推理引擎，参考 vLLM 的设计理念，实现了核心的推理功能，包括：

- **批量推理**：支持多序列并行生成
- **KV Cache 管理**：高效的显存块管理机制
- **CUDA 加速**：自定义 CUDA 内核实现关键操作
- **Qwen3 模型支持**：针对 Qwen3 模型进行了优化

## 项目结构

```
my-vllm/
├── src/
│   ├── engine/          # 推理引擎核心
│   │   ├── llm_engine.py      # LLM 引擎主类
│   │   ├── scheduler.py        # 调度器
│   │   ├── model_runner.py     # 模型运行器
│   │   ├── block_manager.py    # KV Cache 块管理器
│   │   └── sequence.py         # 序列管理
│   ├── layers/          # 模型层实现
│   │   ├── attention.py        # 注意力机制
│   │   ├── linear.py           # 线性层
│   │   ├── layernorm.py        # LayerNorm
│   │   ├── embedding.py        # 嵌入层
│   │   ├── rotary_embedding.py # 旋转位置编码
│   │   ├── sampler.py          # 采样器
│   │   └── cuda/               # CUDA 内核实现
│   │       ├── __init__.py
│   │       ├── activation.cu   # 激活函数 CUDA 内核
│   │       ├── binding.cu      # CUDA 绑定
│   │       ├── embedding.cu    # 嵌入层 CUDA 内核
│   │       ├── layernorm.cu    # LayerNorm CUDA 内核
│   │       ├── linear.cu       # 线性层 CUDA 内核
│   │       └── rotary_embedding.cu  # 旋转位置编码 CUDA 内核
│   ├── model/           # 模型定义
│   │   ├── qwen3.py            # Qwen3 模型
│   │   └── qwen3_opt.py        # Qwen3 优化版本
│   ├── utils/           # 工具函数
│   ├── config.py        # 配置类
│   ├── sampling_params.py      # 采样参数
│   └── llm.py           # LLM 接口
├── example.py           # 使用示例
├── bench.py             # 性能基准测试
└── download_model.py    # 模型下载脚本
```

## 安装要求

- Python 3.8+
- CUDA 11.0+ (需要支持 CUDA 的 GPU)
- PyTorch
- transformers
- huggingface_hub

## 快速开始

### 1. 下载模型

首先需要下载 Qwen3 模型：

```bash
python download_model.py
```

模型将下载到 `./huggingface/Qwen3-0.6B/` 目录。

### 2. 运行示例

运行基础示例代码：

```bash
python example.py
```

示例代码会展示如何使用 LLM 接口进行文本生成。

### 3. 性能测试

运行基准测试来评估性能：

```bash
python bench.py
```

基准测试会测试批量推理的吞吐量。

#### 基准测试结果

运行 `bench.py` 的性能测试结果：

```
Total: 133966tok, Time: 28.62s, Throughput: 4680.66tok/s
```

测试配置：
- 序列数量：256
- 最大输入长度：1024 tokens
- 最大输出长度：1024 tokens

## 配置选项

可以通过修改 `src/config.py` 中的 `Config` 类来调整配置：

- `max_num_batched_tokens`: 一批次的最大 token 数量（默认：16384）
- `max_num_seqs`: 一批次的最大序列数（默认：512）
- `max_model_len`: 模型支持的最大长度（默认：4096）
- `gpu_memory_utilization`: GPU 显存利用率（默认：0.9）
- `kvcache_block_size`: KV Cache 块大小（默认：256，必须是 256 的倍数）
- `cuda_id`: 使用的 GPU 设备 ID（默认：4）

## 核心特性

### 1. 高效的调度器

实现了基于 PagedAttention 的调度算法，支持动态批处理和高效的序列管理。

### 2. KV Cache 块管理

使用块级别的 KV Cache 管理，减少显存碎片，提高显存利用率。

### 3. CUDA 内核优化

针对关键操作（如注意力计算、LayerNorm、线性层等）实现了自定义 CUDA 内核，提升推理速度。

### 4. 批量推理

支持多序列并行生成，充分利用 GPU 并行计算能力。

## 注意事项

- 确保有足够的 GPU 显存来加载模型和 KV Cache
- KV Cache 块大小必须是 256 的倍数（Flash Attention 的要求）
- 当前主要支持 Qwen3 模型，其他模型可能需要适配

## 开发说明

本项目是一个学习项目，用于理解 LLM 推理引擎的实现原理。代码结构清晰，便于学习和扩展。

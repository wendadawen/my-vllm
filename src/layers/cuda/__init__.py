import os
from torch.utils.cpp_extension import load

# ===================== 1. 批量收集所有CUDA文件 =====================
# 算子根目录
CUDA_DIR = os.path.dirname(os.path.abspath(__file__))

# 批量收集所有 .cu 文件，自动遍历目录，无需手动添加
cuda_sources = [os.path.join(CUDA_DIR, f) for f in os.listdir(CUDA_DIR) if f.endswith(".cu")]

# ===================== 2. 集中编译所有算子【整个项目只编译一次】 =====================
# 编译参数
EXTRA_CUDA_FLAGS = [
    "-O3",        # 最高优化级别，算子速度拉满
    "-arch=sm_86",# A6000: 86
    # "-lineinfo",   # 可选，调试时保留行号，不影响速度
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math"
]

EXTRA_CFLAGS = [
    "-std=c++17"
]

# 核心：全局唯一的编译入口，编译后得到算子模块
cuda_ext_lib = load(
    name="project_cuda_ops",          # 自定义算子名称，项目唯一即可
    sources=cuda_sources,              # 批量收集的所有cu文件，无需手动维护！
    # verbose=True,                     # 编译时打印日志
    extra_cuda_cflags=EXTRA_CUDA_FLAGS,
    extra_cflags=EXTRA_CFLAGS,
    build_directory=os.path.join(CUDA_DIR, "build"), # 编译产物统一存放在cuda/build下，管理方便
    with_cuda=True
)

# ===================== 3. 导出算子模块，供项目全局调用 =====================
__all__ = ["cuda_ext_lib"]
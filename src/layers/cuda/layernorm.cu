#include <iostream>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }


__device__ __forceinline__ float warp_reduce(float val) {
    for(int i = 16; i >= 1; i >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }
    return val;
}
// y = (x / RMS(x)) * weight
// RMS(x) = sqrt (sum x*x / d + eps)
// 除了 ()*weight 是原数据类型计算，其他计算都是float计算，对齐vllm
// 一个 block 处理 一行（hidden_size）
__global__ void rmsnorm_bf16_kernel(bfloat16* output, bfloat16* input, bfloat16* weight, int hidden_size, float eps) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    output = output + row * hidden_size;
    input  = input  + row * hidden_size;
    // rms acc
    float sum = 0.f;
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[i]);
        sum += x * x;
    }
    sum = warp_reduce(sum);

    const int NUM_WARPS = (blockDim.x + 31) / 32;
    const int wid = tid >> 5;
    const int lid = tid & 31;
    __shared__ float sdata[32];
    if(lid == 0) sdata[wid] = sum;
    __syncthreads();
    if(wid == 0) {
        sum = lid < NUM_WARPS ? sdata[lid]: 0.f;
        sum = warp_reduce(sum);
        if(lid == 0) {
            sum = rsqrtf(sum / hidden_size + eps);
            sdata[0] = sum;
        }
    }
    __syncthreads();
    sum = sdata[0];
    // norm
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[i]);
        output[i] = (bfloat16)(x * sum) * weight[i];
    }
}

__global__ void rmsnorm_fused_add_inplace_bf16_kernel(bfloat16* input, bfloat16* residual, bfloat16* weight, int hidden_size, float eps) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    input = input + row * hidden_size;
    residual = residual + row * hidden_size;

    float var = 0.f;
    for(int i = tid; i < hidden_size; i += blockDim.x) {

        // vllm
        // bfloat16 temp = input[i];
        // temp = residual[i] + temp;
        // float x = static_cast<float>(temp);
        // residual[i] = temp;
        // var += x * x;

        // nano-vllm
        float x = static_cast<float>(input[i]);
        float r = static_cast<float>(residual[i]);
        x = x + r;
        residual[i] = (bfloat16)x;
        var += x * x;
    }
    var = warp_reduce(var);
    const int NUM_WARPS = (blockDim.x + 31) / 32;
    __shared__ float sdata[32];
    const int wid = tid >> 5;
    const int lid = tid & 31;
    if(lid == 0) sdata[wid] = var;
    __syncthreads();
    if(wid == 0) {
        var = lid < NUM_WARPS ? sdata[lid]: 0.f;
        var = warp_reduce(var);
        if(lid == 0) {
            var = rsqrtf(var / hidden_size + eps);
            sdata[0] = var;
        }
    }
    __syncthreads();
    var = sdata[0];
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(residual[i]);
        input[i] = (bfloat16)(x * var) * weight[i];
    }
}

void rmsnorm_bf16(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kBFloat16);
    int64_t hidden_size = input.size(-1);
    int64_t rows = input.numel() / hidden_size;
    dim3 block(256);
    dim3 grid(rows);
    rmsnorm_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()),
        reinterpret_cast<bfloat16*>(input.data_ptr()),
        reinterpret_cast<bfloat16*>(weight.data_ptr()),
        hidden_size, eps
    );
}

void rmsnorm_fused_add_inplace_bf16(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(residual, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kBFloat16);
    int64_t hidden_size = input.size(-1);
    int64_t rows = input.numel() / hidden_size;
    dim3 block(256);
    dim3 grid(rows);
    rmsnorm_fused_add_inplace_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(input.data_ptr()),
        reinterpret_cast<bfloat16*>(residual.data_ptr()),
        reinterpret_cast<bfloat16*>(weight.data_ptr()),
        hidden_size, eps
    );
}


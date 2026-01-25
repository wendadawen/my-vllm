#include <iostream>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }


// silu(x) = x / (1.f + exp(-x))
template<typename T>
__device__ __forceinline__ T silu(const T& x) {
    return (T)(((float)x) / (1.f + expf((float)(-x))));
}

// block(256)  grid((N + 255) / 256)
__global__ void silu_and_mul_bf16_kernel(bfloat16* output, bfloat16* x, bfloat16* y, int N) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int id = tx + bx * blockDim.x;
    if(id < N) {
        output[id] = silu<bfloat16>(x[id]) * y[id];
    }
}

#define BF168(x) (reinterpret_cast<float4*>(&(x))[0])

__global__ void silu_and_mul_bf16x4_kernel(
    bfloat16* output, 
    bfloat16* x, 
    bfloat16* y, 
    int N
) {
    const int idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx = 8 * idx_thread;
    const int remain = N & 7;
    if (idx + 7 < N) {
        bfloat16 reg_x[8];
        bfloat16 reg_y[8];
        BF168(reg_x) = BF168(x[idx]);
        BF168(reg_y) = BF168(y[idx]);
        bfloat16 reg_out[8];
        reg_out[0] = silu<bfloat16>(reg_x[0]) * reg_y[0];
        reg_out[1] = silu<bfloat16>(reg_x[1]) * reg_y[1];
        reg_out[2] = silu<bfloat16>(reg_x[2]) * reg_y[2];
        reg_out[3] = silu<bfloat16>(reg_x[3]) * reg_y[3];
        reg_out[4] = silu<bfloat16>(reg_x[4]) * reg_y[4];
        reg_out[5] = silu<bfloat16>(reg_x[5]) * reg_y[5];
        reg_out[6] = silu<bfloat16>(reg_x[6]) * reg_y[6];
        reg_out[7] = silu<bfloat16>(reg_x[7]) * reg_y[7];
        BF168(output[idx]) = BF168(reg_out);
    }

    if (idx_thread < remain) {
        const int idx_remain = N - 1 - idx_thread;
        bfloat16 reg_x = x[idx_remain];
        bfloat16 reg_y = y[idx_remain];
        output[idx_remain] = silu<bfloat16>(reg_x) * reg_y;
    }
}

// silu(x) * y
void silu_and_mul_bf16(torch::Tensor& output, torch::Tensor& x, torch::Tensor& y) {
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kBFloat16);
    int64_t N = x.numel();
    dim3 block(256);
    dim3 grid((N + 2047) / 2048);
    silu_and_mul_bf16x4_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()), 
        reinterpret_cast<bfloat16*>(x.data_ptr()), 
        reinterpret_cast<bfloat16*>(y.data_ptr()), N
    );
}
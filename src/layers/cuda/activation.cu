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

// silu(x) * y
void silu_and_mul_bf16(torch::Tensor& output, torch::Tensor& x, torch::Tensor& y) {
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kBFloat16);
    int64_t N = x.numel();
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    silu_and_mul_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()), 
        reinterpret_cast<bfloat16*>(x.data_ptr()), 
        reinterpret_cast<bfloat16*>(y.data_ptr()), N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul_bf16", &silu_and_mul_bf16, "silu_and_mul_bf16");
}

#include <torch/extension.h>

void silu_and_mul_bf16(torch::Tensor& output, torch::Tensor& x, torch::Tensor& y);
void rmsnorm_bf16(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, float eps);
void rmsnorm_fused_add_inplace_bf16(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float eps);
void embedding_bf16(torch::Tensor& output, torch::Tensor& weight, torch::Tensor& token_ids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul_bf16", &silu_and_mul_bf16, "silu_and_mul_bf16");
    m.def("rmsnorm_bf16", &rmsnorm_bf16, "rmsnorm_bf16");
    m.def("rmsnorm_fused_add_inplace_bf16", &rmsnorm_fused_add_inplace_bf16, "rmsnorm_fused_add_inplace_bf16");
    m.def("embedding_bf16", &embedding_bf16, "embedding_bf16");
}

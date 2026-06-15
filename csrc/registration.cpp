#include <torch/extension.h>

void launch_fused_rmsnorm(at::Tensor& input, at::Tensor& weight, at::Tensor& output, float epsilon);
void launch_paged_attention(at::Tensor& q, at::Tensor& k_buffer, at::Tensor& block_table, at::Tensor& context_lens, at::Tensor& out_scores, int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &launch_fused_rmsnorm, "Fused RMSNorm");
    // 👈 暴露出我们崭新的 PagedAttention 符号
    m.def("paged_attention", &launch_paged_attention, "Custom PagedAttention mock kernel");
}
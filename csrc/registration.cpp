#include <torch/extension.h>

void launch_fused_rmsnorm(at::Tensor& input, at::Tensor& weight, at::Tensor& output, float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &launch_fused_rmsnorm, "Fused RMSNorm kernel using Warp Shuffle");
}
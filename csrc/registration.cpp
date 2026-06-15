#include <torch/extension.h>

// 1. 声明我们即将在 CUDA 文件里实现的 Fused 激活算子前向函数
void launch_fused_add_sign_kernel(at::Tensor& x, at::Tensor& y, at::Tensor& out);

// 2. 使用 PyBind11 进行模块注册
// 注意：PYBIND11_MODULE 后面的名字（_C）必须和 setup.py 中定义的扩展库名称完全一致
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_add_sign",          // Python 端调用的函数名
        &launch_fused_add_sign_kernel, // 对应的 C++ 函数指针
        "A custom fused add and sign kernel for vLLM plugin demo" // 文档说明
    );
}
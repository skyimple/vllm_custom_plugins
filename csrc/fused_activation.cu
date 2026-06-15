#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 经典的 Element-wise CUDA 核函数
__global__ void fused_add_sign_cuda_kernel(const float* x, const float* y, float* out, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float sum = x[idx] + y[idx];
        // 简单的门控/激活模拟：如果大于0取1.0，小于0取-1.0
        out[idx] = (sum > 0.0f) ? 1.0f : -1.0f;
    }
}

// C++ 包装层：负责解析 PyTorch 张量并配置线程网格
void launch_fused_add_sign_kernel(at::Tensor& x, at::Tensor& y, at::Tensor& out) {
    // 工业级安全检查：确保所有张量都在 GPU 上且内存连续
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    
    int numel = x.numel();
    
    // 配置线程块（针对 3080Ti，256 是极其稳妥且高效的 BlockSize）
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    
    // 发射 Kernel
    fused_add_sign_cuda_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        numel
    );
}
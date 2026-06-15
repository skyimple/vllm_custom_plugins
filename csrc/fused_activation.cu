#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// 底层硬核工具：让一个 Warp 内的 32 个线程通过寄存器飞速求和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// 核心 Fused RMSNorm CUDA 核函数
// 假设每个 Block 处理大模型中的一个 Token 向量（长度为 hidden_size=1024）
__global__ void fused_rmsnorm_kernel(const float* __restrict__ input, 
                                     const float* __restrict__ weight, 
                                     float* __restrict__ output, 
                                     float epsilon, 
                                     int hidden_size) {
    // 一个 Block 启动 32 个线程（刚好一个 Warp）
    int tid = threadIdx.x; 
    int bid = blockIdx.x; // 代表当前处理的是 Batch * SeqLen 中的第几个 Token
    
    // 指针偏移到当前 Token 的起始地址
    const float* x_input = input + bid * hidden_size;
    float* x_output = output + bid * hidden_size;
    
    // 1. 每个线程负责累加 1024/32 = 32 个元素的平方和到本地寄存器
    float thread_sq_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += 32) {
        float val = x_input[i];
        thread_sq_sum += val * val;
    }
    
    // 2. 调用 Warp Shuffle，让 32 个线程的寄存器数据在片上直接合流
    float total_sq_sum = warp_reduce_sum(thread_sq_sum);
    
    // 3. 计算均方根分母的倒数（所有线程共享这个结果，因为全都在寄存器里算好了）
    float rms_reciprocal = rsqrtf(total_sq_sum / hidden_size + epsilon);
    
    // 4. 第二次循环：直接从输入写到输出，中途乘上分母倒数和权重 Gamma
    // 没有任何中间变量落盘显存！
    for (int i = tid; i < hidden_size; i += 32) {
        x_output[i] = x_input[i] * rms_reciprocal * weight[i];
    }
}

// C++ 包装层
void launch_fused_rmsnorm(at::Tensor& input, at::Tensor& weight, at::Tensor& output, float epsilon) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(), "Inputs must be CUDA tensors");
    
    int num_tokens = input.numel() / input.size(-1); // BatchSize * SeqLen
    int hidden_size = input.size(-1);               // 必须是 1024
    
    // 每个 Token 分配一个 Block (内含 32 个线程，即 1 个 Warp)
    dim3 grid(num_tokens);
    dim3 block(32);
    
    fused_rmsnorm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        epsilon,
        hidden_size
    );
}
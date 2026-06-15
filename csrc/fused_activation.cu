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


// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// 工业级 PagedAttention 极简模拟核函数
// 每个 Thread 负责处理一个特定的 Head 的一个维度
__global__ void paged_attention_mock_kernel(
    const float* __restrict__ q,             // 当前推理的 Query 张量: [BatchSize, NumHeads, HeadSize]
    const float* __restrict__ k_buffer,      // 全局非连续 Key 显存池: [NumBlocks, BlockSize, NumHeads, HeadSize]
    const int* __restrict__ block_table,     // 块映射路由表: [BatchSize, MaxBlocksPerSeq]
    const int* __restrict__ context_lens,    // 每个句子的真实 Token 长度: [BatchSize]
    float* __restrict__ out_scores,          // 输出的注意力得分 Raw Scores: [BatchSize, NumHeads, MaxContextLen]
    int num_heads,
    int head_size,
    int block_size,
    int max_blocks_per_seq) {
    
    // 建立 3D 硬件网格映射
    int head_idx = blockIdx.y;               // 当前是第几个 Attention Head
    int seq_idx = blockIdx.x;                // 当前是 Batch 里的第几条句子（Sequence）
    int token_idx = threadIdx.x;             // 当前线程负责比对历史中的第几个 Token
    
    int current_len = context_lens[seq_idx];
    
    // 防御性边界：如果当前线程的序号超过了这句子的历史总长度，直接挂起，不参与计算
    if (token_idx >= current_len) return;
    
    // 💡 核心跨界寻址：利用我们对齐的公式，找出这个历史 Token 躺在哪个物理房间
    int logical_block_idx = token_idx / block_size;
    int block_slot_idx = token_idx % block_size;
    
    // 从路由表（Block Table）中取出真正的物理 Block 编号
    int physical_block_id = block_table[seq_idx * max_blocks_per_seq + logical_block_idx];
    
    // ⚡ 物理内存精确制导：算出该 Token 的 Key 向量在四维显存池中的绝对起始首地址
    // 寻址步长：物理块号 * 块大小 * 步长 + 块内偏移 * 步长 + 头序号 * 维度
    const float* k_ptr = k_buffer + 
                         physical_block_id * (block_size * num_heads * head_size) +
                         block_slot_idx * (num_heads * head_size) +
                         head_idx * head_size;
                         
    // 定位当前正在处理的 Query 向量的起始地址
    const float* q_ptr = q + seq_idx * (num_heads * head_size) + head_idx * head_size;
    
    // 3. 执行点积计算 (Dot-Product: Q 乘以散落的 K)
    float score = 0.0f;
    for (int d = 0; d < head_size; ++d) {
        score += q_ptr[d] * k_ptr[d];
    }
    
    // 4. 将计算好的原始注意力得分写入输出矩阵
    out_scores[seq_idx * (num_heads * current_len) + head_idx * current_len + token_idx] = score;
}

// C++ 转发包装层
void launch_paged_attention(
    at::Tensor& q, at::Tensor& k_buffer, at::Tensor& block_table, 
    at::Tensor& context_lens, at::Tensor& out_scores, int block_size) {
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int head_size = q.size(2);
    int max_blocks_per_seq = block_table.size(1);
    
    // 配置网格：x 轴处理 Batch，y 轴处理各个 Head
    dim3 grid(batch_size, num_heads);
    // 为了简单，我们让一个 Block 启动 128 个线程，最大支持单句 128 长度的历史比对
    dim3 block(128); 
    
    paged_attention_mock_kernel<<<grid, block>>>(
        q.data_ptr<float>(), k_buffer.data_ptr<float>(), block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(), out_scores.data_ptr<float>(),
        num_heads, head_size, block_size, max_blocks_per_seq
    );
}
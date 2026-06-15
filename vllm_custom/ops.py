import torch
import vllm_custom._C as custom_ops

def fused_rmsnorm(input_tensor: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    assert input_tensor.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert input_tensor.size(-1) == 1024, "This optimized kernel strictly requires hidden_size=1024"
    
    output = torch.empty_like(input_tensor)
    # 调用全新的 C++ 符号
    custom_ops.fused_rmsnorm(input_tensor, weight, output, epsilon)
    return output


# 保持之前的 fused_rmsnorm 不变...

def paged_attention(q: torch.Tensor, k_buffer: torch.Tensor, block_table: torch.Tensor, context_lens: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    batch_size = q.size(0)
    num_heads = q.size(1)
    
    # 为每个 Batch 分配独立的历史得分空间
    max_len = context_lens.max().item()
    out_scores = torch.zeros(batch_size, num_heads, max_len, device="cuda", dtype=torch.float32)
    
    # 跨界调用二进制核心
    custom_ops.paged_attention(q, k_buffer, block_table, context_lens, out_scores, block_size)
    return out_scores
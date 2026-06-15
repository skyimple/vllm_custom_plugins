import torch
import vllm_custom._C as custom_ops

def fused_rmsnorm(input_tensor: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    assert input_tensor.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert input_tensor.size(-1) == 1024, "This optimized kernel strictly requires hidden_size=1024"
    
    output = torch.empty_like(input_tensor)
    # 调用全新的 C++ 符号
    custom_ops.fused_rmsnorm(input_tensor, weight, output, epsilon)
    return output
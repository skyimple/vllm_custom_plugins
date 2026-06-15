import torch
# 导入我们上半场编译成功的二进制扩展
import vllm_custom._C as custom_ops

def fused_add_sign(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    工业级包装器：在把数据喂给底层的 C++ 之前，进行严苛的输入防御检查。
    大模型推理中，任何一个不合法的 Tensor 漏进 CUDA 底层都会直接导致整台机器 Core Dump。
    """
    # 1. 基础防御性检查
    assert x.is_cuda and y.is_cuda, "vLLM Plugin Error: Inputs must be on CUDA devices."
    assert x.shape == y.shape, f"vLLM Plugin Error: Shape mismatch between X {x.shape} and Y {y.shape}."
    assert x.is_contiguous() and y.is_contiguous(), "vLLM Plugin Error: Tensors must be contiguous."
    
    # 2. 动态开辟输出显存空间
    out = torch.empty_like(x)
    
    # 3. 跨界调用 C++ 导出的核心算子
    custom_ops.fused_add_sign(x, y, out)
    
    return out
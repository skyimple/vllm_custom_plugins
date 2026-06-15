import torch
import torch.nn as nn
from vllm_custom.ops import fused_add_sign

class MockLlamaMLPWithPlugin(nn.Module):
    """
    模拟 vLLM 内部的 MLP 层。
    真实的 vLLM 会在这里调用 PagedAttention 或 FusedRMSNorm。
    我们在这里演示如何将自定义的 Fused 算子强行热插拔进 Forward 流水线。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # 模拟模型权重
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False, device="cuda")
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False, device="cuda")
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 模拟常规的矩阵乘法投影
        x = self.gate_proj(hidden_states)
        y = self.up_proj(hidden_states)
        
        # 2. 【核心热插拔点】
        # 原生 PyTorch 做法：out = torch.sign(x + y) -> 会在显存中开辟中间变量，极慢！
        # 工业优化做法：直接调用我们的定制二进制算子
        print(" [vLLM Hook] 成功拦截前向传播，正在调用自定义 Fused 激活算子...")
        output = fused_add_sign(x, y)
        
        return output
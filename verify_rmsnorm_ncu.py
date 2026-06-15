import torch
from vllm_custom.ops import fused_rmsnorm

# 1. 准备超大规模的 Token 数据，把显存撑开，方便观察带宽
# 4096 个 Token，每个 Token 1024 维度
input_tensor = torch.randn(16, 256, 1024, device="cuda")
weight = torch.ones(1024, device="cuda")

# 预热两边算子，排除冷启动干扰
for _ in range(10):
    _ = fused_rmsnorm(input_tensor, weight)
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    _ = input_tensor * torch.rsqrt(variance + 1e-6) * weight

# --- 采样点 1：测试我们手写的 Fused RMSNorm ---
torch.cuda.cudart().cudaProfilerStart()
custom_out = fused_rmsnorm(input_tensor, weight)
torch.cuda.cudart().cudaProfilerStop()

# --- 采样点 2：测试 PyTorch 原生的未融合 RMSNorm ---
torch.cuda.cudart().cudaProfilerStart()
variance = input_tensor.pow(2).mean(-1, keepdim=True)
torch_out = input_tensor * torch.rsqrt(variance + 1e-6) * weight
torch.cuda.cudart().cudaProfilerStop()

print("⚡ NCU 性能采样点埋设完毕！")
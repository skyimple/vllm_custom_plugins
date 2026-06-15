import torch
from vllm_custom.ops import fused_rmsnorm

print("🚀 开始验证工业级 Fused RMSNorm 算子...")

# 1. 模拟标准大模型输入：Batch=4, SeqLen=256, HiddenSize=1024 (共 1024 个 Token)
input_tensor = torch.randn(4, 256, 1024, device="cuda")
# 模拟可学习的权重 Gamma
weight = torch.ones(1024, device="cuda")

# 2. 运行我们的定制 Fused 算子
custom_out = fused_rmsnorm(input_tensor, weight)

# 3. 运行 PyTorch 原生的标准 RMSNorm 算法作为真值对照（Ground Truth）
variance = input_tensor.pow(2).mean(-1, keepdim=True)
torch_out = input_tensor * torch.rsqrt(variance + 1e-6) * weight

# 4. 精度绝对大比对
max_diff = torch.max(torch.abs(custom_out - torch_out)).item()
print(f"📐 算子最大绝对误差 (Max Absolute Error): {max_diff}")

assert max_diff < 1e-4, "❌ 精度对齐失败！请检查寄存器规约逻辑。"
print("🔥 精度完美对齐！你亲手写出的 Fused RMSNorm 算子完全正确！")
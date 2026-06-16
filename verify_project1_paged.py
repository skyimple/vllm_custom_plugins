import torch
import math
from vllm_custom.paged_attn_mock import PagedCacheManager
from vllm_custom.ops import paged_attention_v1

print("🚀 【项目一阶段测试】：开始执行大厂生产级 PagedAttention 全功能对齐...")

# 1. 硬件参数就位
batch_size = 2
num_heads = 16
head_size = 64
block_size = 4
manager = PagedCacheManager(num_blocks=512, num_heads=num_heads, head_size=head_size, block_size=block_size)
block_table, context_lens = manager.get_mock_block_table()

# 2. 构造标准的输入
q = torch.randn(batch_size, num_heads, head_size, device="cuda")

# 🌟 为两个句子分别构造完整、连续的历史 K, V 矩阵，用于 PyTorch 原生计算
max_len = context_lens.max().item()
raw_k_seq = [torch.randn(context_lens[i].item(), num_heads, head_size, device="cuda") for i in range(batch_size)]
raw_v_seq = [torch.randn(context_lens[i].item(), num_heads, head_size, device="cuda") for i in range(batch_size)]

# 🚨 【核心映射】：人肉将这些连续的 KV 数据，打碎、散落进 Paged 显存池的离散块里！
for seq_idx in range(batch_size):
    c_len = context_lens[seq_idx].item()
    for t_idx in range(c_len):
        logical_block = t_idx // block_size
        slot_idx = t_idx % block_size
        p_block = block_table[seq_idx, logical_block].item()
        
        # 塞入离散池
        manager.k_buffer[p_block, slot_idx, :, :] = raw_k_seq[seq_idx][t_idx]
        manager.v_buffer[p_block, slot_idx, :, :] = raw_v_seq[seq_idx][t_idx]

# 3. 运行你的全功能 Paged 算子
cuda_out = paged_attention_v1(q, manager.k_buffer, manager.v_buffer, block_table, context_lens, block_size)

# 4. 用标准 PyTorch 矩阵乘法计算 Ground Truth (针对每一条独立句子)
torch_outputs = []
for i in range(batch_size):
    c_len = context_lens[i].item()
    # [1, num_heads, 1, head_size] — 增加 token 维度，确保 4D @ 4D 正确广播
    qi = q[i].unsqueeze(0).unsqueeze(-2)
    # [1, c_len, num_heads, head_size] -> transpose -> [1, num_heads, c_len, head_size]
    ki = raw_k_seq[i].unsqueeze(0).transpose(1, 2)
    vi = raw_v_seq[i].unsqueeze(0).transpose(1, 2)

    # 算原生 Attention
    scores = torch.matmul(qi, ki.transpose(-2, -1)) / math.sqrt(head_size)
    probs = torch.softmax(scores, dim=-1)
    context_i = torch.matmul(probs, vi) # [1, num_heads, 1, head_size]
    torch_outputs.append(context_i.squeeze(-2).squeeze(0))

torch_out = torch.stack(torch_outputs, dim=0)

# 5. 绝对精度审判
max_diff = torch.max(torch.abs(cuda_out - torch_out)).item()
print(f"\n📊 准入级对齐报告：")
print(f"   最大绝对误差 (Max Absolute Error): {max_diff}")

if max_diff < 1e-4:
    print("\n🎉 完美大胜！项目一第一阶段通关！你独立实现的完整版 PagedAttention 算子已经具备大厂在职交付水准！")
else:
    print("\n❌ 精度溃败！检查共享内存同步或 Softmax 分母分子的规约逻辑！")
    exit(1)
import torch
from vllm_custom.paged_attn_mock import PagedCacheManager
from vllm_custom.ops import paged_attention

print("🔥 正在启动 vLLM 核心 PagedAttention 联合寻址验证...")

# 1. 初始化显存池：512个物理块，16个Head，每Head 64维，BlockSize=4
manager = PagedCacheManager(num_blocks=512, num_heads=16, head_size=64, block_size=4)
block_table, context_lens = manager.get_mock_block_table()

# 2. 构造模拟的 Query 向量 (BatchSize=2, NumHeads=16, HeadSize=64)
q = torch.randn(2, 16, 64, device="cuda")

# 3. 【制造人工已知特征数据】：
# 故意在我们推演的“句子1第7个Token”所躺的物理坑位上填入全 1.0
# 物理坐标：Block 9, Slot 2
manager.k_buffer[9, 2, :, :] = 1.0

# 4. 运行我们魔改的 PagedAttention 算子
scores = paged_attention(q, manager.k_buffer, block_table, context_lens, block_size=4)

# 5. 【真值检验】：
# 因为 K 向量全被我们刷成了 1.0，所以点积结果（Q * 1.0）必须严格等于 Q 向量沿维度的求和！
# 我们来检查 句子1 (index 1) 的 第 7 个 Token (index 6) 的计算结果
cuda_calculated_score = scores[1, 0, 6].item() # 句子1，Head 0，Token 6
gt_score = q[1, 0, :].sum().item()             # 对应的 Query 沿维度求和

print(f"\n📊 芯片动态寻址点对齐报告：")
print(f"   CUDA 核函数寻址计算出的得分: {cuda_calculated_score}")
print(f"   数学理论预期参考得分 (GT):   {gt_score}")

assert abs(cuda_calculated_score - gt_score) < 1e-4, "❌ 动态寻址断层！某个指针计算偏移算错了位置！"
print("\n🎉 苍天在上！动态分页寻址完美打通！你已经彻底掌握了 vLLM 引擎最核心的立身法宝！")
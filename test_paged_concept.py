from vllm_custom.paged_attn_mock import PagedCacheManager

manager = PagedCacheManager(num_blocks=512, num_heads=16, head_size=64)
block_table, context_lens = manager.get_mock_block_table()

print("\n💡 【思考题核对】：")
print(f"句子 1 的长度是 {context_lens[1].item()} 个 Token。")
print(f"它在物理显存里分布在哪些 Block 编号中？ {block_table[1].tolist()}")
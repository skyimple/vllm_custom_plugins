import torch

class PagedCacheManager:
    """
    模拟 vLLM 的物理显存池管理器
    负责在 GPU 上开辟非连续的 KV 缓存块，并管理 Block Table 路由。
    """
    def __init__(self, num_blocks: int, num_heads: int, head_size: int, block_size: int = 4):
        self.num_blocks = num_blocks   # 物理块的总数
        self.num_heads = num_heads     # Attention Head 的数量
        self.head_size = head_size     # 每个 Head 的维度 (e.g., 64)
        self.block_size = block_size   # 每个 Block 可以容纳多少个 Token (e.g., 4)
        
        # 🚨 模拟 vLLM 的冷酷现实：
        # 真实的 vLLM 在服务启动时，就会把剩下的全部显存一次性初始化为一个巨大的四维张量池
        # 形状：[物理块总数, 物理块容量(4), Head数, Head维度]
        self.k_buffer = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda", dtype=torch.float32)
        self.v_buffer = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda", dtype=torch.float32)
        
        print(f"📦 vLLM 虚拟显存池初始化成功！")
        print(f"   Key 物理缓存池形状: {self.k_buffer.shape}")

    def get_mock_block_table(self):
        """
        构造一个模拟的 Block Table（映射路由表）
        假设当前有 2 个句子（BatchSize = 2）
        - 句子 0 比较短，占用了物理块 3 和 物理块 5
        - 句子 1 比较长，占用了物理块 1、物理块 9 和 物理块 2
        """
        # 在真实 vLLM 中，这个表是由 CPU 侧的显存调度器（Scheduler）动态分配并传给 CUDA 的
        block_table = torch.tensor([
            [3, 5, -1],  # -1 代表未启用的空白块
            [1, 9,  2]
        ], device="cuda", dtype=torch.int32)
        
        # 记录每个句子的真实 Token 长度
        context_lens = torch.tensor([6, 11], device="cuda", dtype=torch.int32)
        
        return block_table, context_lens
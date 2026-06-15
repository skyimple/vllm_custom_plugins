import torch
from vllm_custom.model_mock import MockLlamaMLPWithPlugin


def main():
    print("🚀 正在初始化模拟 vLLM 插件流水线...")
    
    # 构造输入：模拟 BatchSize = 2, SequenceLength = 4, HiddenSize = 8 的推理请求
    hidden_states = torch.randn(2, 4, 8, device="cuda")
    
    # 实例化我们注入了插件的自定义模型层
    model_layer = MockLlamaMLPWithPlugin(hidden_size=8)
    
    # 执行前向传播
    with torch.inference_mode(): # 推理模式，关闭梯度计算
        final_output = model_layer(hidden_states)
        
    print("\n 网络前向传播顺利通过！")
    print("输出张量的形状 (应为 [2, 4, 8]):", final_output.shape)
    print("输出张量的部分数值（应全为 1.0 或 -1.0）:\n", final_output[0, 0])

if __name__ == "__main__":
    main()
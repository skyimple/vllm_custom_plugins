import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前绝对路径
ext_directory = os.path.abspath(os.path.dirname(__file__))

setup(
    name="vllm_custom_plugins",
    version="0.1",
    packages=["vllm_custom"],
    ext_modules=[
        CUDAExtension(
            name="vllm_custom._C", # 编译生成的 .so 会躺在 vllm_custom 包下面
            sources=[
                "csrc/registration.cpp",
                "csrc/fused_activation.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # 针对你的 3080Ti (Ampere 架构) 必须锁死 sm_86 开启极限优化
                "nvcc": ["-O3", "-arch=sm_86", "--use_fast_math"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
setup(
    name="blackflash",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="blackflash_cuda",
            sources=[
                "binding/blackflash_ops.cu",
                "kernel/flash_attn_mma.cu",
            ],
            include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",
                    "-std=c++17",
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
            libraries=["cuda"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
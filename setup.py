from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform
import torch
import shutil
ROOT = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = ["--use_fast_math"]
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc", "/openmp"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }
else:
    # Match PyTorch's CXX11 ABI setting
    cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17", "-fopenmp", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }

setup(
    name="flex_gemm",
    packages=[
        "flex_gemm",
        "flex_gemm.utils",
        "flex_gemm.ops",
        "flex_gemm.ops.spconv",
        "flex_gemm.ops.grid_sample",
        "flex_gemm.kernels",
        "flex_gemm.kernels.triton",
        "flex_gemm.kernels.triton.spconv",
        "flex_gemm.kernels.triton.grid_sample",
    ],
    ext_modules=[
        CUDAExtension(
            name="flex_gemm.kernels.cuda",
            sources=[
                # Hashmap functions
                "flex_gemm/kernels/cuda/hash/hash.cu",
                # Serialization functions
                "flex_gemm/kernels/cuda/serialize/api.cu",
                # Grid sample functions
                "flex_gemm/kernels/cuda/grid_sample/grid_sample.cu",
                # Convolution functions
                "flex_gemm/kernels/cuda/spconv/subm_neighbor_map.cu",
                "flex_gemm/kernels/cuda/spconv/sparse_neighbor_map.cu",
                "flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu",
                # main
                "flex_gemm/kernels/cuda/ext.cpp",
            ],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
    ]
)

# copy cache to tmp dir
os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
shutil.copyfile(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
    os.path.expanduser('~/.flex_gemm/autotune_cache.json'),
)


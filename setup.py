from setuptools import setup
import os
import platform
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_cuda_enabled():
    return _is_truthy(os.environ.get("FLEX_GEMM_BUILD_CUDA")) or _is_truthy(
        os.environ.get("BUILD_CUDA")
    )


def _get_cuda_extension_config():
    if not _build_cuda_enabled():
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import (
            CUDAExtension,
            BuildExtension,
            IS_HIP_EXTENSION,
        )
    except Exception as exc:
        raise RuntimeError(
            "CUDA build requested but torch is not available. "
            "Install torch or use --no-build-isolation so the build env can see it."
        ) from exc

    build_target = os.environ.get("BUILD_TARGET", "auto")

    if build_target == "auto":
        is_hip = bool(IS_HIP_EXTENSION)
    elif build_target == "cuda":
        is_hip = False
    elif build_target == "rocm":
        is_hip = True
    else:
        raise ValueError(f"Unsupported BUILD_TARGET: {build_target}")

    if not is_hip:
        cc_flag = ["--use_fast_math"]
    else:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        cc_flag = [f"--offload-arch={arch}" for arch in archs]

    if platform.system() == "Windows":
        extra_compile_args = {
            "cxx": ["/O2", "/std:c++17", "/EHsc"],
            "nvcc": ["-O3", "-std=c++17"] + cc_flag,
        }
    else:
        # Match PyTorch's CXX11 ABI setting
        cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
            "nvcc": ["-O3", "-std=c++17"] + cc_flag,
        }

    ext_modules = [
        CUDAExtension(
            name="flex_gemm.kernels.cuda",
            sources=[
                # Hashmap functions
                "flex_gemm/kernels/cuda/hash/hash.cu",
                # Grid sample functions
                "flex_gemm/kernels/cuda/grid_sample/grid_sample.cu",
                # Convolution functions
                "flex_gemm/kernels/cuda/spconv/neighbor_map.cu",
                # main
                "flex_gemm/kernels/cuda/ext.cpp",
            ],
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = _get_cuda_extension_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

# copy cache to tmp dir
os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
shutil.copyfile(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
    os.path.expanduser('~/.flex_gemm/autotune_cache.json'),
)


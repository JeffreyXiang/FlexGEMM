class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"

class Backend:
    """Backend choices for neighbor map computation."""
    TRITON = "triton"
    CUDA = "cuda"
    TORCH = "torch"

ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  
"Default algorithm"
HASHMAP_RATIO = 2.0         
"Ratio of hashmap size to input size"
BACKEND = Backend.TRITON  
"Default backend for neighbor map computation"


def set_algorithm(algorithm: Algorithm):
    global ALGORITHM
    assert algorithm in (
        Algorithm.EXPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM,
        Algorithm.IMPLICIT_GEMM_SPLITK,
        Algorithm.MASKED_IMPLICIT_GEMM,
        Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    ), f"Unsupported algorithm {algorithm}"
    ALGORITHM = algorithm


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


def set_backend(backend: Backend):
    global BACKEND
    assert backend in (Backend.TRITON, Backend.CUDA), f"Unsupported backend {backend}"  
    BACKEND = backend

from .submanifold_conv3d import SubMConv3dFunction, sparse_submanifold_conv3d

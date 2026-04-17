class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"


ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  # Default algorithm
HASHMAP_RATIO = 2.0         # Ratio of hashmap size to input size
USE_TRITON_NEIGHBOR_MAP = True  # Use Triton implementation for neighbor_map (can switch to CUDA for verification)


def set_algorithm(algorithm: Algorithm):
    global ALGORITHM
    ALGORITHM = algorithm


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


def set_use_triton_neighbor_map(use_triton: bool):
    """Switch between Triton and CUDA neighbor_map implementations for verification."""
    global USE_TRITON_NEIGHBOR_MAP
    USE_TRITON_NEIGHBOR_MAP = use_triton


from .submanifold_conv3d import SubMConv3dFunction, sparse_submanifold_conv3d

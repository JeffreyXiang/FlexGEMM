class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"


ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  
"Default algorithm"
HASHMAP_RATIO = 2.0         
"Ratio of hashmap size to input size"

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


from .submanifold_conv import *
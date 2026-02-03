class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"
    

class SerializationMode:
    """Serialization mode when constructing a key from 3D coordinates."""
    BXYZ = 0
    Z_ORDER = 1
    HILBERT = 2
    

class SparseConv3dOutCoordAlgorithm:
    """Algorithm choices for generating output coordinates."""
    HASHMAP = 0
    EXPAND_UNIQUE = 1


ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  # Default algorithm
HASHMAP_RATIO = 2.0                  # Ratio of hashmap size to input size
OUT_COORD_HASHMAP_RATIO = 1.1        # Ratio of hashmap size to max possible output coordinates
OUT_COORD_ALGO = SparseConv3dOutCoordAlgorithm.HASHMAP
SERIALIZATION_MODE = SerializationMode.BXYZ


def set_algorithm(algorithm: Algorithm):
    global ALGORITHM
    ALGORITHM = algorithm


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


from .submanifold_conv3d import SubMConv3dFunction, sparse_submanifold_conv3d
from .sparse_conv3d import SparseConv3dFunction, sparse_conv3d

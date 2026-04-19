HASHMAP_RATIO = 2.0         # Ratio of hashmap size to input size


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


from .grid_sample_torch import grid_sample_3d_torch
from .grid_sample import grid_sample_3d

__all__ = [
    "grid_sample_3d",
    "grid_sample_3d_torch",
]
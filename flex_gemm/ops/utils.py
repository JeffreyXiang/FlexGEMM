import itertools
from numbers import Number

import torch
from torch import Tensor


def init_hashmap(spatial_size, hashmap_size, device):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D
        
    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)
    
    return hashmap_keys, hashmap_vals


def make_conv_neighbor_offsets(kernel_size: tuple[int, ...], dilation: tuple[int, ...], batch_dims: int = 0, dtype=torch.int32, device: torch.device = None) -> Tensor:
    spatial_ranges = [
        range(-(k // 2) * l, (k // 2 + 1) * l, l)
        for k, l in zip(kernel_size, dilation)
    ]
    offsets = torch.tensor(list(itertools.product(*[
        *itertools.repeat((0,), batch_dims),
        *spatial_ranges,
    ])), dtype=dtype, device=device)
    return offsets


def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0.) -> Tensor:
    "Pad the specified dimension of the tensor to the next power of two with zeros."
    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(size, int):
        size = (size,)
    if len(dim) == 1 and len(size) > 1:
        size = size * len(dim)
    if len(dim) > 1 and len(size) == 1:
        size = size * len(dim)
    assert len(dim) == len(size), f"dim and size must have the same length. Got {len(dim)} and {len(size)} respectively."
    
    pad_size = [0] * x.dim()
    for d, s in zip(dim, size):
        pad_size[d] = max(0, s - x.shape[d])
    if any(p > 0 for p in pad_size):
        x = torch.nn.functional.pad(
            x, 
            tuple(itertools.chain.from_iterable((0, p) for p in reversed(pad_size))), 
            value=value
        )
    return x
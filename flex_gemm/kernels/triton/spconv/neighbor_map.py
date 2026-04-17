from typing import *
import itertools

import torch
from torch import Tensor
import triton
import triton.language as tl

from ..hashmap import hashmap_build_triton, _hashmap_lookup_inline_32bit, _vec_load, pad_to_size_along_dim


__all__ = [
    "build_neighbour_map_triton",
    "build_submanifold_convnd_neighbour_map_triton",
    "build_submanifold_conv3d_neighbour_map_triton",
    "neighbor_map_post_process_for_masked_implicit_gemm_1",
]



# Loop over neighbors. Compatible with arbitrary number of neighbors, but less efficient.
@triton.jit
def _hashmap_build_neighbour_map_loop_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.pointer_type,
    n_coords: int,
    neightbor_offsets_ptr: tl.pointer_type,
    n_neighbors: int,
    neighbor_map_ptr: tl.pointer_type,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_coords

    coord_vec = _vec_load(coords_ptr + offs * D, mask, D)
    arange_D = tl.arange(0, D)
    for i in range(n_neighbors):
        neighbor_offset_vec = tl.load(neightbor_offsets_ptr + i * D + arange_D)
        neighbor_coord_vec = coord_vec + neighbor_offset_vec[None, :]
        found_idx = _hashmap_lookup_inline_32bit(
            hashmap_ptr, hashmap_size, 
            coords_ptr, 
            neighbor_coord_vec, 
            mask=mask, 
            D=D
        )
        tl.store(neighbor_map_ptr + offs * n_neighbors + i, found_idx, mask=mask)


# Vectorized neighbor search. More efficient but requires more registers. Maybe neighbors at most 256
@triton.jit
def _hashmap_build_neighbour_map_vectorized_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.pointer_type,
    n_coords: int,
    neightbor_offsets_ptr: tl.pointer_type,
    N_NEIGHBORS: tl.constexpr,
    neighbor_map_ptr: tl.pointer_type,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_coords

    coord_vec = _vec_load(coords_ptr + offs * D, mask, D)
    neighbor_offset_vec = tl.load(neightbor_offsets_ptr + tl.arange(0, N_NEIGHBORS * D)).reshape(N_NEIGHBORS, D)

    neighbor_coord_vec = coord_vec[:, None, :] + neighbor_offset_vec[None, :, :]
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size, 
        coords_ptr, 
        neighbor_coord_vec,
        mask=mask[:, None],
        D=D
    )
    tl.store(neighbor_map_ptr + offs[:, None] * N_NEIGHBORS + tl.arange(0, N_NEIGHBORS)[None, :], found_idx, mask=mask[:, None])


def build_neighbour_map_triton(
    coords: torch.Tensor,
    offsets: torch.Tensor,
    hashmap: torch.Tensor | None = None,
):
    """Build neighbor map given coords and neighbor offsets.
    
    Args:
        coords: (N, D) int32 tensor, tag & slot hashmap. If None, it will be built from coords.
            Prefix dimensions will be viewed as batch dimensions.
        offsets: (V, D) int32 tensor, the relative offsets of neighbors, where V is the size of the kernel.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.
    """
    assert coords.dtype == torch.int32 and offsets.dtype == torch.int32, "coords and offsets must be int32 tensors."
    assert coords.shape[1] == offsets.shape[1], f"coords and offsets must have the same number of dimensions. Got {coords.shape[1]} and {offsets.shape[1]} respectively."

    coords = pad_to_size_along_dim(coords, dim=1, size=triton.next_power_of_2(coords.shape[1]), value=0)
    coords = coords.contiguous()
    n_coords, D = coords.shape
    n_neighbors = offsets.shape[0]
    
    if hashmap is None:
        hashmap = hashmap_build_triton(coords)
    
    if n_neighbors <= 256:
        # For small number of neighbors, load all neighbor offsets into registers search in one shot.
        offsets = pad_to_size_along_dim(offsets, dim=(0, 1), size=(triton.next_power_of_2(offsets.shape[0]), D), value=0)
        offsets = offsets.contiguous()
        n_neighbors_pad = offsets.shape[0]
        neighbor_map = torch.full((n_coords, n_neighbors_pad), -1, dtype=torch.int32, device=coords.device)
        BLOCK_SIZE = triton.cdiv(256, n_neighbors_pad)  # Try to keep the number of neighbors per block constant
        grid = (triton.cdiv(n_coords, BLOCK_SIZE), )
        _hashmap_build_neighbour_map_vectorized_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neightbor_offsets_ptr=offsets,
            N_NEIGHBORS=n_neighbors_pad,
            neighbor_map_ptr=neighbor_map,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE
        )
        neighbor_map = neighbor_map[:, :n_neighbors].contiguous()
    else:
        # For large number of neighbors, loop over neighbors in the kernel to save registers.
        offsets = pad_to_size_along_dim(offsets, dim=1, size=D, value=0)  
        neighbor_map = torch.full((n_coords, n_neighbors), -1, dtype=torch.int32, device=coords.device)
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_coords, BLOCK_SIZE), )
        _hashmap_build_neighbour_map_loop_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neightbor_offsets_ptr=offsets,
            n_neighbors=n_neighbors,
            neighbor_map_ptr=neighbor_map,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return neighbor_map


def build_submanifold_convnd_neighbour_map_triton(
    coords: torch.Tensor,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    hashmap: torch.Tensor | None = None
) -> torch.Tensor:
    """Make neighbor map for N-dimensional submanifold convolution.
    
    Args:
        coords: (N, D or more) int32 tensor, where D is the dimension of coordinates (e.g., 4 for (..., x_1, x_2, ..., x_D)).
            Prefix dimensions will be viewed as batch dimensions.
        kernel_size: (K_1, K_2, ..., K_D), tuple of kernel sizes in each dimension.
        dilation: (L_1, L_2, ..., L_D), tuple of dilation factors in each dimension.
        hashmap: (N,) int32 tensor, tag & slot hashmap. If None, it will be built from coords.
    
    Returns:
        neighbor_map: (N, K_1 * K_2 * ... * K_D) uint32 tensor, where V is the size of the kernel. Each entry is the index of the neighbor in coords, or -1 if not found.
    """
    assert len(kernel_size) == len(dilation) <= coords.shape[1], f"kernel_size and dilation must have the same length, and cannot be longer than the coordinate dimension. Got {len(kernel_size)} and {len(dilation)} respectively."
    offsets = torch.tensor(
        list(itertools.chain(
            itertools.repeat(0, coords.shape[1] - len(kernel_size)),  # For batch dimensions
            itertools.product(*[range(-(k // 2) * l, (k // 2 + 1) * l, l) for k, l in zip(kernel_size, dilation)])
        )), 
        dtype=torch.int32,
        device=coords.device
    )
    return build_neighbour_map_triton(coords, offsets, hashmap)


def build_submanifold_conv3d_neighbour_map_triton(
    coords: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
    hashmap: torch.Tensor | None = None
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    assert len(kernel_size) == 3 and len(dilation) == 3, "kernel_size and dilation must be either int or tuple of length 3."

    return build_submanifold_convnd_neighbour_map_triton(
        coords, kernel_size, dilation, hashmap
    )


@triton.jit
def _mask_gray_binary_kernel(
    mask_ptr: tl.pointer_type,
    gray_ptr: tl.pointer_type,
    binary_ptr: tl.pointer_type,
    N: int,
    V: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    mask_vals = tl.load(
        mask_ptr + offs_n[:, None] * stride_n + offs_v[None, :],
        mask=mask_n[:, None] & mask_v[None, :],
        other=0
    )
    valid = mask_vals != 0

    bit_weights = tl.full((BLOCK_V,), 1, dtype=tl.uint32) << offs_v.to(tl.uint32)
    zeros = tl.zeros((BLOCK_V,), dtype=tl.uint32)
    gray = tl.sum(tl.where(valid, bit_weights[None, :], zeros[None, :]), axis=1)

    binary = gray
    binary ^= binary >> 1
    binary ^= binary >> 2
    binary ^= binary >> 4
    binary ^= binary >> 8
    binary ^= binary >> 16

    tl.store(gray_ptr + offs_n, gray, mask=mask_n)
    tl.store(binary_ptr + offs_n, binary, mask=mask_n)


def neighbor_map_post_process_for_masked_implicit_gemm_1(
    neighbor_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post-process the neighbor map for masked implicit GEMM.

    Returns:
        gray_code: (N,) uint32 tensor of per-row kernel masks (bitset up to 32).
        sorted_idx: (N,) int64 tensor sorting rows by binary code.
        valid_signal_i: (L,) int32 tensor of input indices for valid signals.
        valid_signal_o: (L,) int32 tensor of output indices for valid signals.
        valid_signal_seg: (V + 1,) int32 tensor of segment boundaries per kernel idx.
    """
    if neighbor_map.dim() != 2:
        raise ValueError("neighbor_map must be a 2D tensor")

    neighbor_map = neighbor_map.contiguous()
    N, V = neighbor_map.shape

    if V > 32:
        raise ValueError("Kernel volume must be <= 32 for bitset encoding")

    if neighbor_map.numel() == 0:
        gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_map.device)
        sorted_idx = torch.empty((N,), dtype=torch.int64, device=neighbor_map.device)
        valid_signal_i = torch.empty((0,), dtype=torch.int32, device=neighbor_map.device)
        valid_signal_o = torch.empty((0,), dtype=torch.int32, device=neighbor_map.device)
        valid_signal_seg = torch.zeros((V + 1,), dtype=torch.int32, device=neighbor_map.device)
        return gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg

    if neighbor_map.dtype not in (torch.int32, torch.uint32):
        raise ValueError("neighbor_map must be int32 or uint32")

    neighbor_mask = neighbor_map.view(torch.int32) != -1
    neighbor_mask_u8 = neighbor_mask.to(torch.uint8).contiguous()

    gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_map.device)
    binary_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_map.device)
    BLOCK_N = 256
    BLOCK_V = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    _mask_gray_binary_kernel[grid](
        mask_ptr=neighbor_mask_u8,
        gray_ptr=gray_code,
        binary_ptr=binary_code,
        N=N,
        V=V,
        stride_n=neighbor_mask_u8.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_V=BLOCK_V,
    )
    
    sorted_idx = torch.argsort(binary_code)

    neighbor_map_T = neighbor_map.transpose(0, 1).contiguous()
    neighbor_mask_T = neighbor_mask.transpose(0, 1).contiguous()

    neighbor_mask_flat = neighbor_mask_T.reshape(-1)
    mask_flat_indices = torch.where(neighbor_mask_flat)[0]

    valid_signal_i = neighbor_map_T.reshape(-1).index_select(0, mask_flat_indices)
    valid_signal_o = torch.remainder(mask_flat_indices, N).to(torch.uint32)
    valid_signal_seg = torch.empty((V + 1,), dtype=torch.uint32, device=neighbor_map.device)
    valid_signal_seg[0] = 0
    neighbor_mask_T.sum(dim=1, dtype=torch.uint32).cumsum(dim=0, out=valid_signal_seg[1:])

    return gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg
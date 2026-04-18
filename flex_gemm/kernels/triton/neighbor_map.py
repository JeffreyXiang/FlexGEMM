from typing import *
import itertools

import torch
from torch import Tensor
import triton
import triton.language as tl

from .hashmap import hashmap_build_triton, _hashmap_lookup_inline_32bit, _vec_load, pad_to_size_along_dim


__all__ = [
    "build_neighbor_map_triton",
    "neighbor_map_post_process_for_masked_implicit_gemm_1",
    "neighbor_map_post_process_for_masked_implicit_gemm_2",
]


# Loop over neighbors. Compatible with arbitrary number of neighbors, but less efficient.
@triton.jit
def _hashmap_build_neighbor_map_loop_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.pointer_type,
    n_coords: int,
    neighbor_offsets_ptr: tl.pointer_type,
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
        neighbor_offset_vec = tl.load(neighbor_offsets_ptr + i * D + arange_D)
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
def _hashmap_build_neighbor_map_vectorized_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.pointer_type,
    n_coords: int,
    neighbor_offsets_ptr: tl.pointer_type,
    N_NEIGHBORS: tl.constexpr,
    neighbor_map_ptr: tl.pointer_type,
    D: tl.constexpr,
    BLOCK_N_COORDS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N_COORDS + tl.arange(0, BLOCK_N_COORDS)
    mask_coords = offs < n_coords

    coord_vec = _vec_load(coords_ptr + offs * D, mask_coords, D)
    neighbor_offset_vec = tl.load(neighbor_offsets_ptr + tl.arange(0, N_NEIGHBORS * D)).reshape(N_NEIGHBORS, D)

    neighbor_coord_vec = coord_vec[:, None, :] + neighbor_offset_vec[None, :, :]
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size,
        coords_ptr, 
        neighbor_coord_vec,
        mask=mask_coords[:, None],
        D=D
    )
    tl.store(
        neighbor_map_ptr + offs[:, None] * N_NEIGHBORS + tl.arange(0, N_NEIGHBORS)[None, :], 
        found_idx, 
        mask=mask_coords[:, None]
    )


def build_neighbor_map_triton(
    coords: Tensor,
    offsets: Tensor,
    hashmap: Tensor | None = None,
):
    """Build neighbor map given coords and neighbor offsets.
    
    Args:
        coords: (N, D) int8 / int16 / int32 tensor, tag & slot hashmap. If None, it will be built from coords.
            Prefix dimensions will be viewed as batch dimensions.
        offsets: (V, D) tensor of the same dtype as coords, the relative offsets of neighbors, where V is the size of the kernel.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.

    Returns:
        neighbor_map: (N, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    assert coords.dtype in (torch.int8, torch.int16, torch.int32), f"coords must be int8, int16 or int32. Got {coords.dtype}."
    assert coords.dtype == offsets.dtype, f"coords and offsets must have the same dtype. Got {coords.dtype} and {offsets.dtype} respectively."
    assert coords.shape[1] == offsets.shape[1], f"coords and offsets must have the same number of dimensions. Got {coords.shape[1]} and {offsets.shape[1]} respectively."

    D = triton.next_power_of_2(coords.shape[1])
    coords = pad_to_size_along_dim(coords, dim=1, size=D, value=0)
    coords = coords.contiguous()
    n_coords = coords.shape[0]
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
        _hashmap_build_neighbor_map_vectorized_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neighbor_offsets_ptr=offsets,
            N_NEIGHBORS=n_neighbors_pad,
            neighbor_map_ptr=neighbor_map,
            D=D,
            BLOCK_N_COORDS=BLOCK_SIZE
        )
        neighbor_map = neighbor_map[:, :n_neighbors].contiguous()
    else:
        # For large number of neighbors, loop over neighbors in the kernel to save registers.
        offsets = pad_to_size_along_dim(offsets, dim=1, size=D, value=0)  
        neighbor_map = torch.full((n_coords, n_neighbors), -1, dtype=torch.int32, device=coords.device)
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_coords, BLOCK_SIZE), )
        _hashmap_build_neighbor_map_loop_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neighbor_offsets_ptr=offsets,
            n_neighbors=n_neighbors,
            neighbor_map_ptr=neighbor_map,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return neighbor_map



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
    neighbor_map: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        return gray_code, sorted_idx, valid_signal_i, valid_signal_o

    if neighbor_map.dtype not in (torch.int32, torch.uint32):
        raise ValueError("neighbor_map must be int32 or uint32")

    neighbor_mask = neighbor_map.view(torch.int32) != -1

    gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_map.device)
    binary_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_map.device)
    BLOCK_N = 64
    BLOCK_V = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    _mask_gray_binary_kernel[grid](
        mask_ptr=neighbor_mask,
        gray_ptr=gray_code,
        binary_ptr=binary_code,
        N=N,
        V=V,
        stride_n=neighbor_mask.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_V=BLOCK_V,
    )
    
    sorted_idx = torch.argsort(binary_code.view(torch.int32))

    neighbor_map_T = neighbor_map.transpose(0, 1)
    neighbor_mask_T = neighbor_mask.transpose(0, 1)

    mask_flat_indices = torch.where(neighbor_mask_T.reshape(-1))[0]

    valid_signal_i = neighbor_map_T.reshape(-1).index_select(0, mask_flat_indices).to(torch.uint32)
    valid_signal_o = torch.remainder(mask_flat_indices.to(torch.int32), N).to(torch.uint32)

    return gray_code, sorted_idx, valid_signal_i, valid_signal_o


@triton.jit
def _reduce_gray_code_kernel(
    gray_code_ptr: tl.pointer_type,
    sorted_idx_ptr: tl.pointer_type,
    reduced_code_ptr: tl.pointer_type,
    seglen_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs < N
    sorted_offs = tl.load(sorted_idx_ptr + offs, mask=mask_n, other=0)
    gray = tl.load(gray_code_ptr + sorted_offs, mask=mask_n, other=0).to(tl.uint32)

    acc = tl.reduce_or(gray, axis=0)
    seglen = tl.sum((acc >> tl.arange(0, 32)) & 1, axis=0).to(tl.int32)

    tl.store(reduced_code_ptr + pid, acc)
    tl.store(seglen_ptr + pid + 1, seglen)


@triton.jit
def _scatter_reduced_code_kernel(
    reduced_code_ptr: tl.pointer_type,
    seg_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    num_blocks: int,
    BLOCK_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid < num_blocks
    code = tl.load(reduced_code_ptr + pid, mask=mask, other=0).to(tl.uint32)
    seg_start = tl.load(seg_ptr + pid, mask=mask, other=0).to(tl.int32)
    bits = tl.arange(0, BLOCK_BITS)
    bit_set = (code >> bits.to(tl.uint32)) & 1
    write_pos = tl.cumsum(bit_set.to(tl.int32), axis=0) - 1
    do_store = (bit_set != 0) & mask
    pos = seg_start + write_pos
    tl.store(out_ptr + pos, bits, mask=do_store)


def neighbor_map_post_process_for_masked_implicit_gemm_2(
    gray_code: Tensor,
    sorted_idx: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Build valid kernel indices for masked implicit GEMM.

    Returns:
        valid_kernel_idx: (L,) int32 tensor containing valid kernel indices.
        valid_kernel_seg: (num_blocks + 1,) int32 tensor containing segment boundaries.
    """
    if gray_code.dim() != 1 or sorted_idx.dim() != 1:
        raise ValueError("gray_code and sorted_idx must be 1D tensors")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of 2")
    if gray_code.dtype not in (torch.int32, torch.uint32):
        raise ValueError("gray_code must be int32 or uint32")

    gray_code = gray_code.contiguous()
    sorted_idx = sorted_idx.contiguous()
    N = gray_code.numel()

    num_blocks = triton.cdiv(N, block_size)
    valid_kernel_seg = torch.zeros((num_blocks + 1,), dtype=torch.int32, device=gray_code.device)

    if N == 0 or num_blocks == 0:
        valid_kernel_idx = torch.empty((0,), dtype=torch.int32, device=gray_code.device)
        return valid_kernel_idx, valid_kernel_seg

    reduced_code = torch.empty((num_blocks,), dtype=torch.int32, device=gray_code.device)
    grid = (num_blocks,)
    _reduce_gray_code_kernel[grid](
        gray_code_ptr=gray_code,
        sorted_idx_ptr=sorted_idx,
        reduced_code_ptr=reduced_code,
        seglen_ptr=valid_kernel_seg,
        N=N,
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size >= 128 else 2,
    )

    torch.cumsum(valid_kernel_seg, dim=0, out=valid_kernel_seg)
    total_valid = valid_kernel_seg[-1].item()
    if total_valid == 0:
        valid_kernel_idx = torch.empty((0,), dtype=torch.int32, device=gray_code.device)
        return valid_kernel_idx, valid_kernel_seg

    valid_kernel_idx = torch.empty((total_valid,), dtype=torch.int32, device=gray_code.device)
    _scatter_reduced_code_kernel[grid](
        reduced_code_ptr=reduced_code,
        seg_ptr=valid_kernel_seg,
        out_ptr=valid_kernel_idx,
        num_blocks=num_blocks,
        BLOCK_BITS=32,
        num_warps=1,
    )

    return valid_kernel_idx, valid_kernel_seg
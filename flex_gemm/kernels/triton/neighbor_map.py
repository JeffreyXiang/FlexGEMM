from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from .hashmap import hashmap_build_triton, _hashmap_lookup_inline_32bit, _vec_load, pad_to_size_along_dim


__all__ = [
    "build_neighbor_map_from_offsets_triton",
    "build_neighbor_map_from_kernel_dilation_triton",
    "neighbor_map_post_process_for_masked_implicit_gemm_1",
    "neighbor_map_post_process_for_masked_implicit_gemm_2",
]



@triton.jit
def _hashmap_find_store_neighbor_map_inline(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.const,
    offs_coords: tl.tensor,
    mask_coords: tl.tensor,
    offs_neighbors: tl.tensor,
    mask_neighbors: tl.tensor,
    neighbor_offset_vec: tl.tensor,
    n_neighbors: int,
    neighbor_map_ptr: tl.tensor,
    D: tl.constexpr,
):  
    mask_coords_x_neighbors = mask_coords[:, None] & mask_neighbors[None, :]
    coord_vec = _vec_load(coords_ptr + offs_coords * D, mask_coords, D)
    neighbor_coord_vec = coord_vec[:, None, :] + neighbor_offset_vec[None, :, :].to(coord_vec.dtype)
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size,
        coords_ptr, 
        neighbor_coord_vec,
        mask=mask_coords_x_neighbors,
        D=D
    )
    tl.store(
        neighbor_map_ptr + offs_coords[:, None] * n_neighbors + offs_neighbors[None, :], 
        found_idx, 
        mask=mask_coords_x_neighbors
    )


@triton.jit
def _hashmap_prepare_offs_masks_inline(
    pid_coords: int,
    pid_neighbors: int,
    n_coords: int,
    n_neighbors: int,
    BLOCK_N_COORDS: tl.constexpr,
    BLOCK_N_NEIGHBORS: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    offs_coords = pid_coords * BLOCK_N_COORDS + tl.arange(0, BLOCK_N_COORDS)
    mask_coords = offs_coords < n_coords
    offs_neighbors = pid_neighbors * BLOCK_N_NEIGHBORS + tl.arange(0, BLOCK_N_NEIGHBORS)
    mask_neighbors = offs_neighbors < n_neighbors
    return offs_coords, mask_coords, offs_neighbors, mask_neighbors


# ===== Arbitrary neighbor offsets ======
@triton.jit
def _hashmap_build_neighbor_map_from_offsets_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.const,
    n_coords: int,
    neighbor_offsets_ptr: tl.const,
    n_neighbors: int,
    neighbor_map_ptr: tl.pointer_type,
    D: tl.constexpr,
    BLOCK_N_NEIGHBORS: tl.constexpr,
    BLOCK_N_COORDS: tl.constexpr
):
    pid_coords, pid_neighbors = tl.program_id(0), tl.program_id(1)
    offs_coords, mask_coords, offs_neighbors, mask_neighbors = _hashmap_prepare_offs_masks_inline(
        pid_coords, pid_neighbors, n_coords, n_neighbors, BLOCK_N_COORDS, BLOCK_N_NEIGHBORS
    )

    neighbor_offset_vec = tl.load(neighbor_offsets_ptr + offs_neighbors[:, None] * D + tl.arange(0, D)[None, :], mask=mask_neighbors[:, None], other=0)

    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_ptr=coords_ptr,
        offs_coords=offs_coords,
        mask_coords=mask_coords,
        offs_neighbors=offs_neighbors,
        mask_neighbors=mask_neighbors,
        neighbor_offset_vec=neighbor_offset_vec,
        n_neighbors=n_neighbors,
        neighbor_map_ptr=neighbor_map_ptr,
        D=D
    )


# =========== 4D ============
@triton.jit
def _make_4d_vec_inline(x0, x1, x2, x3, dtype=tl.int32) -> tl.tensor:
    idx = tl.arange(0, 4)
    vec = tl.full((4,), x0, dtype=dtype)
    vec = tl.where(idx == 1, x1, vec)
    vec = tl.where(idx == 2, x2, vec)
    vec = tl.where(idx == 3, x3, vec)
    return vec


@triton.jit
def _make_conv_neighbor_offsets_4d_inline(
    idx: tl.tensor,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, 
    L0: int, L1: int, L2: int, L3: int
) -> tl.tensor:
    strides_vec = _make_4d_vec_inline(K1 * K2 * K3, K2 * K3, K3, 1)
    kernel_sizes_vec = _make_4d_vec_inline(K0, K1, K2, K3)
    dilation_vec = _make_4d_vec_inline(L0, L1, L2, L3)

    idx = idx.to(tl.int32)
    neighbor_delta = ((idx[:, None] // strides_vec) % kernel_sizes_vec - (kernel_sizes_vec - 1) // 2) * dilation_vec

    return neighbor_delta


@triton.jit
def _hashmap_build_neighbor_map_conv4d_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.const,
    n_coords: int,
    neighbor_map_ptr: tl.pointer_type,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
    L0: int, L1: int, L2: int, L3: int,
    BLOCK_N_NEIGHBORS: tl.constexpr,
    BLOCK_N_COORDS: tl.constexpr,
):
    n_neighbors: tl.constexpr = K0 * K1 * K2 * K3
    pid_coords = tl.program_id(0) 
    pid_neighbors = tl.program_id(1)

    offs_coords, mask_coords, offs_neighbors, mask_neighbors = _hashmap_prepare_offs_masks_inline(
        pid_coords, pid_neighbors, n_coords, n_neighbors, BLOCK_N_COORDS, BLOCK_N_NEIGHBORS
    )

    # Make constexpr neighbor deltas.
    neighbor_offset_vec = _make_conv_neighbor_offsets_4d_inline(
        offs_neighbors, 
        K0, K1, K2, K3, 
        L0, L1, L2, L3
    ) # (BLOCK_N_NEIGHBORS, D)
    
    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_ptr=coords_ptr,
        offs_coords=offs_coords,
        mask_coords=mask_coords,
        offs_neighbors=offs_neighbors,
        mask_neighbors=mask_neighbors,
        neighbor_offset_vec=neighbor_offset_vec,
        n_neighbors=n_neighbors,
        neighbor_map_ptr=neighbor_map_ptr,
        D=4
    )



# ========= 8-D =========
@triton.jit
def _make_8d_vec_inline(x0, x1, x2, x3, x4, x5, x6, x7, dtype=tl.int32) -> tl.tensor:
    idx = tl.arange(0, 8)
    vec = tl.full((8,), x0, dtype=dtype)
    vec = tl.where(idx == 1, x1, vec)
    vec = tl.where(idx == 2, x2, vec)
    vec = tl.where(idx == 3, x3, vec)
    vec = tl.where(idx == 4, x4, vec)
    vec = tl.where(idx == 5, x5, vec)
    vec = tl.where(idx == 6, x6, vec)
    vec = tl.where(idx == 7, x7, vec)
    return vec


@triton.jit
def _make_conv_neighbor_offsets_8d_inline(
    idx: tl.tensor,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, K4: tl.constexpr, K5: tl.constexpr, K6: tl.constexpr, K7: tl.constexpr,
    L0: int, L1: int, L2: int, L3: int, L4: int, L5: int, L6: int, L7: int
) -> tl.tensor:
    strides_vec = _make_8d_vec_inline(
        K1 * K2 * K3 * K4 * K5 * K6 * K7,
        K2 * K3 * K4 * K5 * K6 * K7,
        K3 * K4 * K5 * K6 * K7,
        K4 * K5 * K6 * K7,
        K5 * K6 * K7,
        K6 * K7,
        K7,
        1,
    )
    kernel_sizes_vec = _make_8d_vec_inline(K0, K1, K2, K3, K4, K5, K6, K7)
    dilation_vec = _make_8d_vec_inline(L0, L1, L2, L3, L4, L5, L6, L7)

    idx = idx.to(tl.int32)
    neighbor_delta = ((idx[:, None] // strides_vec) % kernel_sizes_vec - (kernel_sizes_vec - 1) // 2) * dilation_vec

    return neighbor_delta


@triton.jit
def _hashmap_build_neighbor_map_conv8d_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_ptr: tl.pointer_type,
    n_coords: int,
    neighbor_map_ptr: tl.pointer_type,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, K4: tl.constexpr, K5: tl.constexpr, K6: tl.constexpr, K7: tl.constexpr,
    L0: int, L1: int, L2: int, L3: int, L4: int, L5: int, L6: int, L7: int,
    BLOCK_N_NEIGHBORS: tl.constexpr,
    BLOCK_N_COORDS: tl.constexpr,
):
    n_neighbors: tl.constexpr = K0 * K1 * K2 * K3 * K4 * K5 * K6 * K7
    pid_coords, pid_neighbors = tl.program_id(0), tl.program_id(1)

    offs_coords, mask_coords, offs_neighbors, mask_neighbors = _hashmap_prepare_offs_masks_inline(
        pid_coords, pid_neighbors, n_coords, n_neighbors, BLOCK_N_COORDS, BLOCK_N_NEIGHBORS
    )

    # Make constexpr neighbor deltas.
    neighbor_offset_vec = _make_conv_neighbor_offsets_8d_inline(
        offs_neighbors, 
        K0, K1, K2, K3, K4, K5, K6, K7,
        L0, L1, L2, L3, L4, L5, L6, L7
    ) # (BLOCK_N_NEIGHBORS, D)
    
    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_ptr=coords_ptr,
        offs_coords=offs_coords,
        mask_coords=mask_coords,
        offs_neighbors=offs_neighbors,
        mask_neighbors=mask_neighbors,
        neighbor_offset_vec=neighbor_offset_vec,
        n_neighbors=n_neighbors,
        neighbor_map_ptr=neighbor_map_ptr,
        D=8
    )


def build_neighbor_map_from_kernel_dilation_triton(
    coords: Tensor,
    *,
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    hashmap: Tensor | None = None,
) -> Tensor:
    """Build neighbor map for constexpr kernel defined by `kernel_size` and `dilation`.
    Supports up to 8D.
    
    Args:
        coords: (N, D) int8 / int16 / int32 tensor, tag & slot hashmap. If None, it will be built from coords.
            Prefix dimensions will be viewed as batch dimensions.
        kernel_size: (<=D,) tuple of integers, the size of the convolution kernel.
        dilation: (<=D,) tuple of integers, the dilation of the convolution kernel.
            If kernel_size or dilation is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.
    
    Returns:
        neighbor_map: (N, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    assert coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"coords must be int8, int16 or int32. Got {coords.dtype}."
    )

    orig_D = coords.shape[1]
    if orig_D > 8:
        raise ValueError(f"Too many coordinate dimensions: {orig_D}. Max supported is 8.")
    # Pad prefix batch dimensions of kernel size and dilation to match coordinate dimensions. 
    if len(kernel_size) > orig_D or len(dilation) > orig_D:
        raise ValueError(f"kernel_size and dilation cannot have more dimensions than coords. Got kernel_size with {len(kernel_size)} dims, dilation with {len(dilation)} dims, but coords has {orig_D} dims.")
    kernel_size = (1,) * (orig_D - len(kernel_size)) + kernel_size
    dilation = (1,) * (orig_D - len(dilation)) + dilation

    # Pad to 4D or 8D (at tail)
    D = 4 if orig_D <= 4 else 8
    coords = pad_to_size_along_dim(coords, dim=1, size=D, value=0, side='right')
    n_coords = coords.shape[0]
    kernel_size_D = tuple(kernel_size) + (1,) * (D - orig_D)
    dilation_D = tuple(dilation) + (1,) * (D - orig_D)
    n_neighbors = math.prod(kernel_size_D)

    # Build hashmap if not provided
    if hashmap is None:
        hashmap = hashmap_build_triton(coords)
    
    # Build neighbor map
    neighbor_map = torch.empty((n_coords, n_neighbors), dtype=torch.int32, device=coords.device)
    BLOCK_N_NEIGHBORS = min(32, triton.next_power_of_2(n_neighbors))
    BLOCK_N_COORDS = max(1, 256 // BLOCK_N_NEIGHBORS)
    grid = (triton.cdiv(n_coords, BLOCK_N_COORDS), triton.cdiv(n_neighbors, BLOCK_N_NEIGHBORS))
    if D == 4:
        _hashmap_build_neighbor_map_conv4d_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neighbor_map_ptr=neighbor_map,
            K0=kernel_size_D[0], K1=kernel_size_D[1], K2=kernel_size_D[2], K3=kernel_size_D[3],
            L0=dilation_D[0], L1=dilation_D[1], L2=dilation_D[2], L3=dilation_D[3],
            BLOCK_N_NEIGHBORS=BLOCK_N_NEIGHBORS,
            BLOCK_N_COORDS=BLOCK_N_COORDS
        )
    elif D == 8:
        _hashmap_build_neighbor_map_conv8d_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_ptr=coords,
            n_coords=n_coords,
            neighbor_map_ptr=neighbor_map,
            K0=kernel_size_D[0], K1=kernel_size_D[1], K2=kernel_size_D[2], K3=kernel_size_D[3], K4=kernel_size_D[4], K5=kernel_size_D[5], K6=kernel_size_D[6], K7=kernel_size_D[7],
            L0=dilation_D[0], L1=dilation_D[1], L2=dilation_D[2], L3=dilation_D[3], L4=dilation_D[4], L5=dilation_D[5], L6=dilation_D[6], L7=dilation_D[7],
            BLOCK_N_NEIGHBORS=BLOCK_N_NEIGHBORS,
            BLOCK_N_COORDS=BLOCK_N_COORDS
        )
    return neighbor_map


def build_neighbor_map_from_offsets_triton(
    coords: Tensor,
    offsets: Tensor,
    hashmap: Tensor | None = None,
):
    """Build neighbor map given coords and neighbor offsets.
    
    Args:
        coords: (N, D) int8 / int16 / int32 tensor, tag & slot hashmap. If None, it will be built from coords.
            Prefix dimensions will be viewed as batch dimensions.
        offsets: (V, D) tensor of the same dtype as coords, the relative offsets of neighbors, where V is the size of the kernel.
        kernel_size: (D,) tuple of integers, the size of the convolution kernel.
        dilation: (D,) tuple of integers, the dilation of the convolution kernel.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.

    Returns:
        neighbor_map: (N, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    assert coords.dtype in (torch.int8, torch.int16, torch.int32), f"coords must be int8, int16 or int32. Got {coords.dtype}."
    assert coords.dtype == offsets.dtype, f"coords and offsets must have the same dtype. Got {coords.dtype} and {offsets.dtype} respectively."
    if offsets.shape[1] > coords.shape[1]:
        raise ValueError(f"Offsets cannot have more dimensions than coords. Got offsets with {offsets.shape[1]} dims, but coords has {coords.shape[1]} dims.")
    if offsets.shape[1] < coords.shape[1]:
        offsets = pad_to_size_along_dim(offsets, dim=1, size=coords.shape[1], value=0, side='left')

    # General neighbor map construction with arbitrary offsets.
    orig_D = coords.shape[1]
    # Pad to next power of 2 in int32 (4 bytes) words
    D = triton.cdiv(triton.next_power_of_2(triton.cdiv(orig_D * coords.dtype.itemsize, 4)) * 4, coords.dtype.itemsize)
    coords = pad_to_size_along_dim(coords, dim=1, size=D, value=0, side='right')
    n_coords = coords.shape[0]
    if hashmap is None:
        hashmap = hashmap_build_triton(coords)

    offsets = pad_to_size_along_dim(offsets, dim=1, size=D, value=0, side='right')
    n_neighbors = offsets.shape[0]
    neighbor_map = torch.empty((n_coords, n_neighbors), dtype=torch.int32, device=coords.device)

    offsets = offsets.contiguous()
    
    BLOCK_N_NEIGHBORS = 32
    BLOCK_N_COORDS = 256 // BLOCK_N_NEIGHBORS 
    grid = (triton.cdiv(n_coords, BLOCK_N_COORDS), triton.cdiv(n_neighbors, BLOCK_N_NEIGHBORS))
    _hashmap_build_neighbor_map_from_offsets_kernel[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap.shape[0],
        coords_ptr=coords,
        n_coords=n_coords,
        neighbor_offsets_ptr=offsets,
        n_neighbors=n_neighbors,
        neighbor_map_ptr=neighbor_map,
        D=D,
        BLOCK_N_NEIGHBORS=BLOCK_N_NEIGHBORS,
        BLOCK_N_COORDS=BLOCK_N_COORDS
    )

    return neighbor_map


# ========================================================================================
# ================= neighbor map post-processing for masked implicit GEMM ================
# ========================================================================================
@triton.jit
def _mask_gray_binary_kernel(
    mask_ptr: tl.pointer_type,
    gray_ptr: tl.pointer_type,
    binary_ptr: tl.pointer_type,
    N: int,
    V: int,
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
    valid = mask_vals > 0

    bit_weights = tl.full((BLOCK_V,), 1, dtype=tl.uint32) << offs_v
    gray = tl.sum(tl.where(valid, bit_weights[None, :], 0), axis=1)

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
        raise ValueError(f"Masked implicit GEMM with more than 32 neighbors is not supported. Got V={V}.")

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
    binary_code = torch.empty((N,), dtype=torch.int32, device=neighbor_map.device)
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
    
    sorted_idx = torch.argsort(binary_code)

    neighbor_map_T = neighbor_map.transpose(0, 1)
    neighbor_mask_T = neighbor_mask.transpose(0, 1)

    mask_flat_indices = torch.argwhere(neighbor_mask_T.reshape(-1)).squeeze(1)

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
    seglen = tl.sum((acc >> tl.arange(0, 32)) & 1, axis=0).to(tl.int32) # popcount of acc. Inline ASM does not improve speed.

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
    assert gray_code.is_contiguous() and sorted_idx.is_contiguous(), "gray_code and sorted_idx must be contiguous"

    N = gray_code.numel()

    num_blocks: int = triton.cdiv(N, block_size)
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

    valid_kernel_seg.cumsum_(dim=0)
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
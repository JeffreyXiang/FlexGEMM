import itertools
from typing import Optional, Tuple, Literal
from numbers import Number
import torch
from torch import Tensor

import triton
import triton.language as tl

__all__ = [
    'hashmap_build_triton',
    'hashmap_lookup_triton',
    'hashmap_build_lookup_triton',
]


def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0., side: Literal['left', 'right'] = 'right') -> Tensor:
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
            tuple(itertools.chain.from_iterable((0, p) if side == 'right' else (p, 0) for p in reversed(pad_size))), 
            value=value
        )
    return x


@triton.jit
def _vec_load(ptr: tl.pointer_type, mask: tl.tensor, D: tl.constexpr) -> tl.tensor:
    "Load a vector key from memory given a pointer."
    vec = tl.load(tl.expand_dims(ptr, -1) + tl.arange(0, D), mask=tl.expand_dims(mask, -1), other=0)
    return vec


@triton.jit
def _vec_hash_32bit(vec: tl.tensor, D: tl.constexpr) -> tl.tensor:
    # Per-index multipliers and a single accumulator keep mixing strong with fewer ops.
    idx = tl.arange(0, D)
    seed = idx.to(tl.uint32) + 0x9E3779B9
    seed = (seed ^ (seed >> 16)) * 0x7FEB352D
    seed = (seed ^ (seed >> 15)) * 0x846CA68B
    seed = seed ^ (seed >> 16)
    mult = seed | 1

    v = vec.to(tl.uint32)
    v = v + seed
    v ^= v >> 15
    v *= 0x2C1B3C6D
    v ^= v >> 12

    h = tl.sum(v * mult, axis=-1)
    h ^= h >> 16
    h *= 0x7FEB352D
    h ^= h >> 15
    h *= 0x846CA68B
    h ^= h >> 16
    return h.to(tl.int32)


@triton.jit
def _vec_pack_little_endian_to_int32(vec: tl.tensor) -> tl.tensor:
    """Pack a little-endian integer vector into int32 words."""
    tl.static_assert(
        vec.dtype.itemsize == 1 or vec.dtype.itemsize == 2 or vec.dtype.itemsize == 4,
        "Unsupported query_vec element width",
    )
    if vec.dtype.itemsize == 4:
        return vec.to(tl.int32)

    if vec.dtype.itemsize == 2:
        vec_u16 = tl.reshape(tl.cast(vec, tl.uint16, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 2, 2)
        return tl.sum(vec_u16.to(tl.uint32) << (tl.arange(0, 2) << 4), axis=-1).to(tl.int32)

    if vec.dtype.itemsize == 1:
        vec_u8 = tl.reshape(tl.cast(vec, tl.uint8, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 4, 4)
        return tl.sum(vec_u8.to(tl.uint32) << (tl.arange(0, 4) << 3), axis=-1).to(tl.int32)


@triton.jit
def _hashmap_build_kernel_32bit(
    hashmap_ptr: tl.tensor, 
    hashmap_size: int,
    keys_ptr: tl.const,
    n_keys: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(D * keys_ptr.dtype.element_ty.itemsize % 4 == 0, "keys byte width must be divisible by 4")
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32))
    D_32: tl.constexpr = D * keys_ptr.dtype.element_ty.itemsize // 4

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_keys   

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    # Compute hash value
    # Load key vectors once, then hash from registers.
    key_vec = _vec_load(keys_ptr_32 + idx * D_32, mask=mask, D=D_32)
    hash_val = _vec_hash_32bit(key_vec, D=D_32)
    # Upper tag bits, lower index bits. (index must be smaller than hashmap_size)
    store_val = (hash_val & TAG_BIT_MASK) | idx

    # Probing loop
    to_be_inserted = mask
    target_slot = hash_val & SLOT_BIT_MASK
    while tl.sum(to_be_inserted) > 0:
        # Try to insert the key index into the hash table
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(to_be_inserted, -1, -2), store_val)
        # Update mask: keep only those that failed to insert
        to_be_inserted = to_be_inserted & (prev >= 0)

        # Update target_slot for next attempt
        target_slot += tl.where(to_be_inserted, 1, 0)
        target_slot &= SLOT_BIT_MASK


@triton.jit
def _hashmap_lookup_inline_32bit(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    keys_ptr: tl.const,
    query_vec: tl.tensor,   
    mask: tl.tensor,
    D: tl.constexpr
):
    """Lookup the query_vec in the hash map and return the found index or -1 if not found.
    NOTE: keys_ptr must be 4-byte aligned and D must be divisible by 4.
    """
    keys_ptr_32 = tl.cast(keys_ptr, tl.pointer_type(tl.int32))
    query_vec_32 = _vec_pack_little_endian_to_int32(query_vec)
    D_32: tl.constexpr = D * query_vec.dtype.itemsize // 4
    tl.static_assert(D_32 == query_vec_32.shape[-1], "Invalid query_vec shape after packing to int32. Check D and input dtype.")

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    hash_val = _vec_hash_32bit(query_vec_32, D=D_32)
    query_tag = hash_val & TAG_BIT_MASK

    is_active = tl.broadcast_to(mask, query_vec_32.shape[:-1])
    found_idx = tl.full(query_vec_32.shape[:-1], -1, tl.int32)

    # Probing loop
    curr_slot = hash_val & SLOT_BIT_MASK
    while tl.sum(is_active) > 0:
        # Compute current slot to probe
        stored_val = tl.load(hashmap_ptr + curr_slot, mask=is_active, other=-1)

        # Drop queries that hit empty slots
        is_active &= (stored_val >= 0)
        
        # Extract stored index & tag
        stored_idx = stored_val & SLOT_BIT_MASK
        stored_tag = stored_val & TAG_BIT_MASK
        # First compare tags
        is_match = is_active & (stored_tag == query_tag)
        # Then compare full keys
        key_vec = _vec_load(keys_ptr_32 + stored_idx * D_32, mask=is_match, D=D_32)
        is_match &= tl.min(key_vec == query_vec_32, axis=-1) > 0

        # Update found indices
        success = is_match & is_active
        found_idx = tl.where(success, stored_idx, found_idx)
        is_active &= ~success
        
        # Update current slot
        curr_slot += 1
        curr_slot &= SLOT_BIT_MASK
    return found_idx
    

@triton.jit
def _hashmap_lookup_kernel_32bit(
    queries_ptr: tl.const,
    keys_ptr: tl.const,
    hashmap_ptr: tl.pointer_type,
    results_ptr: tl.pointer_type,
    hashmap_size: int,
    n_queries: int,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_queries

    # Compute hash value for queries
    query_vec = _vec_load(queries_ptr + offs * D, mask=mask, D=D)
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size, 
        keys_ptr, query_vec, 
        mask=mask, 
        D=D
    )

    # Store results
    tl.store(results_ptr + offs, found_idx, mask=mask)


def hashmap_build_triton(keys: Tensor) -> Tensor:
    """
    Build a hash map from the given keys using Triton.
    
    Args:
        keys (Tensor): A tensor of shape `(n_keys, D)` representing the keys.

    Returns:
        Tensor: A 1D tensor representing the hash map.

    Notes
    -----
        The hash map stores a combination of a hash tag and the index of each key.
        See `hashmap_lookup_triton` for querying the hash map.
        Use `hashmap_build_lookup_triton` for a combined build and lookup operation.
    """
    # Determine hash map size (next power of two greater than 2x number of elements)
    n_keys = keys.shape[0]
    hashmap_size = triton.next_power_of_2(n_keys * 2)

    # Pad keys to a byte width that is a power of two in int32 words.
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes)
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE), )
    
    _hashmap_build_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return hashmap


def hashmap_lookup_triton(hashmap: Tensor, keys: Tensor, queries: Tensor) -> Tensor:
    """
    Lookup the indices of the given queries in the provided hash map.

    Args:
        hashmap (Tensor): A 1D tensor representing the hash map built using `hashmap_build_triton`.
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys used to build the hash map.
        queries (Tensor): A tensor of shape `(n_queries, *key_dims)` representing the queries to look up.
    
    Returns:
        Tensor: A 1D int32 tensor of shape `(n_queries,)` containing the indices of the queries in the keys.
                If a query is not found, its index will be -1.
    """
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)

    n_queries = queries.shape[0]
    hashmap_size = hashmap.shape[0]

    # Pad and convert keys and queries to appropriate dtype.
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes) 
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)
    queries_i32 = pad_to_size_along_dim(queries, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    results = torch.empty((n_queries,), dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_queries, BLOCK_SIZE), )
    _hashmap_lookup_kernel_32bit[grid](
        queries_ptr=queries_i32,
        keys_ptr=keys_i32,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results


def hashmap_build_lookup_triton(keys: Tensor, queries: Tensor) -> Tensor:
    """
    Build a hash map from the given keys and lookup the indices of the given queries in a single operation.
    Args:
        keys (Tensor): A tensor of shape `(n_keys, *key_dims)` representing the keys.
        queries (Tensor): A tensor of shape `(n_queries, *key_dims)` representing the queries to look up.
    
    Returns:
        Tensor: A 1D int32 tensor of shape `(n_queries,)` containing the indices of the queries in the keys.
                If a query is not found, its index will be -1.
    """
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view.
    n_keys = keys.shape[0]
    n_queries = queries.shape[0]

    # Determine hash map size (next power of two greater than 2x number of elements)
    hashmap_size = triton.next_power_of_2(n_keys * 2)

    # Pad keys and queries to a byte width that is a power of two in int32 words.
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)
    D_32 = triton.next_power_of_2(triton.cdiv(keys.shape[1], 4))   # pad to power of two by int32 (4 bytes)
    keys_i32 = pad_to_size_along_dim(keys, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)
    queries_i32 = pad_to_size_along_dim(queries, dim=1, size=D_32 * 4, value=0, side='right').view(torch.int32)

    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=keys.device)
    results = torch.empty((n_queries,), dtype=torch.int32, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_keys, BLOCK_SIZE), )
    
    _hashmap_build_kernel_32bit[grid](
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        keys_ptr=keys_i32,
        n_keys=n_keys,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    grid = (triton.cdiv(n_queries, BLOCK_SIZE), )
    _hashmap_lookup_kernel_32bit[grid](
        queries_ptr=queries_i32,
        keys_ptr=keys_i32,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        D=D_32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results


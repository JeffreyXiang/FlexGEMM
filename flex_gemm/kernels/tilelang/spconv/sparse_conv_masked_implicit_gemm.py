from typing import *
import math
import torch
import tilelang
import tilelang.language as T
from ....utils.autotuner import tilelang_autotune
from . import config


def sparse_conv_fwd_implicit_gemm_kernel_key(compile_kwargs, kernel_args):
    input, weight, bias, neighbor, output = list(kernel_args)[:5]
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    return f'(2^{int(math.log2(N))}, 2^{int(math.log2(M))}, {Ci}, {Co}, {V}, {compile_kwargs["dtype"]})'


def sparse_conv_fwd_implicit_gemm_kernel_heuristic(compile_kwargs, kernel_args):
    input, weight, bias, neighbor, sorted_idx, valid_kernel_fn, valid_kernel_seg_fn, output = list(kernel_args)
    valid_kernel = valid_kernel_fn(compile_kwargs['B1'])
    valid_kernel_seg = valid_kernel_seg_fn(compile_kwargs['B1'])
    return input, weight, bias, neighbor, sorted_idx, valid_kernel, valid_kernel_seg, output


@tilelang_autotune(
    configs=config.autotune_config,
    key_fn=sparse_conv_fwd_implicit_gemm_kernel_key,
    heuristic_fn=sparse_conv_fwd_implicit_gemm_kernel_heuristic,
)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True
    }
)
def sparse_conv_fwd_masked_implicit_gemm_kernel(
    Ci, Co, V,
    B1,                         # Block size for M dimension
    B2,                         # Block size for Co dimension
    BK,                         # Block size for K dimension (V * Ci)
    num_stages,                 # Number of stages for pipelining
    num_warps,                  # Number of warps per block
    has_bias,                   # Whether bias is present
    transpose_weight = False,   # Whether to transpose the weight matrix
    smem_neighbor = False,      # Whether to use shared memory for neighbor offset
    dtype = T.float16,
):
    N = T.dynamic(name='N')
    M = T.dynamic(name='M')
    L = T.dynamic(name='L')

    weight_shape = (Ci, V, Co) if transpose_weight else (Co, V, Ci)
    weight_shared_shape = (BK, B2) if transpose_weight else (B2, BK)

    @T.prim_func
    def main(
        input: T.Tensor((N, Ci), dtype),
        weight: T.Tensor(weight_shape, dtype),
        bias: T.Tensor((Co), dtype),
        neighbor: T.Tensor((M, V), T.uint32),
        sorted_idx: T.Tensor((M,), T.int64),
        valid_kernel: T.Tensor((L,), T.int32),
        valid_kernel_seg: T.Tensor((T.ceildiv(M, B1) + 1,), T.int32),
        output: T.Tensor((M, Co), dtype),
    ):
        """
        Sparse convolution forward kernel using masked implicit GEMM.
        
        Args:
            input (pointer): A pointer to the input tensor of shape (N, Ci)
            weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
            bias (pointer): A pointer to the bias tensor of shape (Co)
            neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
            sorted_idx (pointer): A pointer to the sorted index tensor of shape (M)
            valid_kernel (pointer): A pointer to the valid neighbor index tensor of shape (L)
            valid_kernel_seg (pointer): A pointer to the valid neighbor index segment tensor of shape (BLOCK_M + 1)
            output (pointer): A pointer to the output tensor of shape (M, Co)
        """
        with T.Kernel(T.ceildiv(M, B1), T.ceildiv(Co, B2), threads=num_warps*32) as (block_id_m, block_id_co):
            num_threads = num_warps * 32
            tid = T.get_thread_binding(0)
            segs_1 = T.ceildiv(B1, num_threads)
            num_segs = T.ceildiv(num_threads, B1)
            seglenth_K = T.ceildiv(BK, num_segs)
            seg_id = tid // num_segs
            seg_iid = tid % num_segs
            
            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            input_shared = T.alloc_shared((B1, BK), dtype)
            weight_shared = T.alloc_shared(weight_shared_shape, dtype)
            output_shared = T.alloc_shared((B1, B2), dtype)
            if smem_neighbor:
                neighbor_onchip = T.alloc_shared((B1, V), T.uint32)
            else:
                neighbor_onchip = T.alloc_local((segs_1, V), T.uint32)
            
            valid_kernel_onchip = T.alloc_local((V,), T.int32)

            T.annotate_layout({
                input_shared: tilelang.layout.make_swizzled_layout(input_shared),
                weight_shared: tilelang.layout.make_swizzled_layout(weight_shared),
            })

            accumulator = T.alloc_fragment((B1, B2), T.float32)
            T.clear(accumulator)
            
            valid_kernel_start = valid_kernel_seg[block_id_m]
            valid_kernel_end = valid_kernel_seg[block_id_m + 1]
            valid_kernel_seglen = valid_kernel_end - valid_kernel_start
            
            for neign_idx in T.Vectorized(valid_kernel_seglen):
                valid_kernel_onchip[neign_idx] = valid_kernel[valid_kernel_start + neign_idx]

            # Load neighbor offset
            for seg_1 in T.Unroll(segs_1):
                m_idx = sorted_idx[block_id_m * B1 + seg_1 * num_threads + seg_id]
                for neign_idx in T.Serial(valid_kernel_seglen):
                    v = valid_kernel_onchip[neign_idx]
                    if smem_neighbor:
                        neighbor_onchip[seg_1 * num_threads + seg_id, neign_idx] = neighbor[m_idx, v]
                    else:
                        neighbor_onchip[seg_1, neign_idx] = neighbor[m_idx, v]

            num_k = T.ceildiv(Ci, BK)
            for k in T.Pipelined(num_k * valid_kernel_seglen, num_stages=num_stages):
                v = k // num_k
                bk = k % num_k

                # Load input block
                if smem_neighbor:
                    for i, j in T.Parallel(B1, BK):
                        input_shared[i, j] = input[neighbor_onchip[i, v], bk * BK + j]
                else:
                    for seg_1 in T.Unroll(segs_1):
                        for i in T.Vectorized(seglenth_K):       
                            input_shared[seg_1 * num_threads + seg_id, seg_iid * seglenth_K + i] = input[neighbor_onchip[seg_1, v], bk * BK + seg_iid * seglenth_K + i]

                # Load weight block
                if not transpose_weight:
                    for i, j in T.Parallel(B2, BK):
                        weight_shared[i, j] = weight[block_id_co * B2 + i, valid_kernel_onchip[v], bk * BK + j]
                else:
                    for i, j in T.Parallel(BK, B2):
                        weight_shared[i, j] = weight[bk * BK + i, valid_kernel_onchip[v], block_id_co * B2 + j]

                T.gemm(input_shared, weight_shared, accumulator, transpose_A=False, transpose_B=not transpose_weight)

            # Add bias
            if has_bias:
                for i, j in T.Parallel(B1, B2):
                    accumulator[i, j] += bias[block_id_co * B2 + j]

            # Store output block
            T.copy(accumulator, output_shared)
            for i, j in T.Parallel(B1, B2):
                output[sorted_idx[block_id_m * B1 + i], block_id_co * B2 + j] = output_shared[i, j]

    return main


def sparse_conv_fwd_masked_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    sorted_idx: torch.Tensor,
    valid_kernel: Callable[[int], torch.Tensor],
    valid_kernel_seg: Callable[[int], torch.Tensor],
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    sparse_conv_fwd_masked_implicit_gemm_kernel(
        Ci, Co, V,
        has_bias = bias is not None,
        dtype = weight.dtype,
    ) (
        input, weight, bias, neighbor, sorted_idx, valid_kernel, valid_kernel_seg, output
    )
    return output

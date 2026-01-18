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
    return f'(2^{int(math.log2(N))}, 2^{int(math.log2(M))}, {Ci}, {Co}, {V}, {compile_kwargs["dtype"]}'


@tilelang_autotune(
    configs=config.autotune_config,
    key_fn=sparse_conv_fwd_implicit_gemm_kernel_key,
)
@tilelang.jit
def sparse_conv_fwd_implicit_gemm_kernel(
    Ci, Co, V,
    B1,                         # Block size for M dimension
    B2,                         # Block size for Co dimension
    BK,                         # Block size for K dimension (V * Ci)
    num_stages,                 # Number of stages for pipelining
    num_warps,                  # Number of warps per block
    has_bias,                   # Whether bias is present
    transpose_weight = False,   # Whether to transpose the weight matrix
    dtype = T.float16,
):
    N = T.dynamic(name='N')
    M = T.dynamic(name='M')

    weight_shape = (Ci, V, Co) if transpose_weight else (Co, V, Ci)
    weight_shared_shape = (BK, B2) if transpose_weight else (B2, BK)
    weight_flat_shape = (Ci, V * Co) if transpose_weight else (Co, V * Ci)

    @T.prim_func
    def main(
        input: T.Tensor((N, Ci), dtype),
        weight: T.Tensor(weight_shape, dtype),
        bias: T.Tensor((Co), dtype),
        neighbor: T.Tensor((M, V), T.uint32),
        output: T.Tensor((M, Co), dtype),
    ):
        """
        Sparse convolution forward kernel using implicit GEMM.
        
        Args:
            input (pointer): A pointer to the input tensor of shape (N, Ci)
            weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
            bias (pointer): A pointer to the bias tensor of shape (Co)
            neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
            output (pointer): A pointer to the output tensor of shape (M, Co)
        """
        with T.Kernel(T.ceildiv(M, B1), T.ceildiv(Co, B2), threads=num_warps*32) as (block_id_m, block_id_co):
            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            weight_flat = T.Tensor(weight_flat_shape, dtype, weight.data)

            input_shared = T.alloc_shared((B1, BK), dtype)
            weight_shared = T.alloc_shared(weight_shared_shape, dtype)
            neighbor_shared = T.alloc_shared((B1, V), T.uint32)

            T.annotate_layout({
                input_shared: tilelang.layout.make_swizzled_layout(input_shared),
                weight_shared: tilelang.layout.make_swizzled_layout(weight_shared),
            })

            accumulator = T.alloc_fragment((B1, B2), T.float32)
            T.clear(accumulator)

            T.copy(neighbor[block_id_m * B1, 0], neighbor_shared)

            num_k = T.ceildiv(Ci, BK)
            for k in T.Pipelined(num_k * V, num_stages=num_stages):
                v = k // num_k
                bk = k % num_k

                # Load input block
                for i, j in T.Parallel(B1, BK):
                    neighbor_offset = neighbor_shared[i, v]
                    input_shared[i, j] = input[neighbor_offset, bk * BK + j]

                # Load weight block
                if not transpose_weight:
                    T.copy(weight_flat[block_id_co * B2, v * Ci + bk * BK], weight_shared)
                else:
                    T.copy(weight_flat[bk * BK, v * Ci + block_id_co * B2], weight_shared)

                T.gemm(input_shared, weight_shared, accumulator, transpose_A=False, transpose_B=not transpose_weight)

            # Add bias
            if has_bias:
                for i, j in T.Parallel(B1, B2):
                    accumulator[i, j] += bias[block_id_co * B2 + j]

            T.copy(accumulator, output[block_id_m * B1, block_id_co * B2])

    return main


def sparse_conv_fwd_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    sparse_conv_fwd_implicit_gemm_kernel(
        Ci, Co, V,
        has_bias = bias is not None,
        dtype = weight.dtype,
    ) (
        input, weight, bias, neighbor, output,
    )
    return output


# def sparse_conv_bwd_implicit_gemm(
#     grad_output: torch.Tensor,
#     input: torch.Tensor,
#     weight: torch.Tensor,
#     bias: torch.Tensor,
#     neighbor: torch.Tensor,
#     neighbor_bwd: torch.Tensor,
# ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#     assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
#     assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
#     assert input.is_contiguous(), "Matrix input must be contiguous"
#     assert weight.is_contiguous(), "Matrix weight must be contiguous"
#     assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
#     assert neighbor_bwd.is_contiguous(), "Matrix neighbor_bwd must be contiguous"
#     N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
#     LOGN = int(math.log2(N))
#     LOGM = int(math.log2(M))
    
#     grad_input, grad_weight, grad_bias = None, None, None
    
#     # Grad for input
#     if input.requires_grad:
#         # Allocate output matrix output.
#         grad_input = torch.empty((N, Ci), device=input.device, dtype=input.dtype)
#         # Launch the kernel.
#         grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']),)
#         weight_bwd = weight if config.USE_ON_THE_FLY_WEIGHT_TRANSPOSE else weight.transpose(0, 2).contiguous()
#         sparse_conv_fwd_implicit_gemm_kernel[grid](
#             grad_output, weight_bwd, None, neighbor_bwd, grad_input,
#             N, LOGM, LOGN, Co, Ci, V,
#             allow_tf32=config.allow_tf32,
#             TRANSPOSE_WEIGHT=config.USE_ON_THE_FLY_WEIGHT_TRANSPOSE,
#         )
        
#     # Grad for weight
#     if weight.requires_grad:
#         # Allocate output matrix output.
#         grad_weight = torch.empty((Co, V, Ci), device=weight.device, dtype=weight.dtype)
#         # Launch the kernel.
#         grid = lambda META: (triton.cdiv(Co, META['B1']), triton.cdiv(V * Ci, META['BV'] * META['BCi']))
#         sparse_conv_bwd_weight_implicit_gemm_kernel[grid](
#             grad_output, input, neighbor, grad_weight,
#             M, LOGN, LOGM, Ci, Co, V,
#             allow_tf32=config.allow_tf32,
#         )
        
#     # Grad for bias
#     if bias is not None and bias.requires_grad:
#         grad_bias = grad_output.sum(0)

#     return grad_input, grad_weight, grad_bias


from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config
from .sparse_conv_implicit_gemm import sparse_conv_fwd_implicit_gemm_kernel


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics({
    'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
    'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
    'HAS_BIAS': lambda args: args['bias'] is not None,
})
@triton.jit
def sparse_conv_fwd_masked_implicit_gemm_kernel(
    input,
    weight,
    bias,
    neighbor,
    sorted_idx,
    output,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for M dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    HAS_BIAS: tl.constexpr,  # Whether bias is present
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Huristic parameters
    valid_kernel,
    valid_kernel_seg,
    # Specialize
    TRANSPOSE_WEIGHT: tl.constexpr = False,  # Whether to transpose the weight matrix
):
    """
    Indice convolution forward kernel using masked implicit GEMM.
    
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
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    valid_kernel_start = tl.load(valid_kernel_seg + block_id_m)
    valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_m + 1) - valid_kernel_start
    offset_m = block_id_m * B1 + tl.arange(0, B1)
    m_mask = offset_m < M
    offset_sorted_m = tl.load(sorted_idx + offset_m, mask=m_mask, other=0)  # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co                  # (B2,)
    offset_k = tl.arange(0, BK)                                             # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(num_k * valid_kernel_seglen):
        v = k // num_k
        bk = k % num_k
        v = tl.load(valid_kernel + valid_kernel_start + v)
        # Calculate pointers to weight matrix.
        if not TRANSPOSE_WEIGHT:
            weight_ptr = weight + (offset_co[None, :] * V * Ci) + (v * Ci) + (bk * BK + offset_k[:, None])      # (BK, B2)
        else:
            weight_ptr = weight + (offset_co[None, :]) + (v * Co) + ((bk * BK + offset_k[:, None]) * V * Co)    # (BK, B2)
        # Calculate pointers to input matrix.
        neighbor_offset = tl.load(neighbor + offset_sorted_m * V + v)                             # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset[:, None].to(tl.int64) * Ci + offset_k[None, :])         # (B1, BK)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
    c = accumulator.to(input.type.element_ty)
            
    # add bias
    if HAS_BIAS:
        bias_block = tl.load(bias + offset_co)
        c += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_m = offset_sorted_m
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_m[:, None] * Co + out_offset_co[None, :])
    out_mask = m_mask[:, None] & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, c, mask=out_mask)
    

@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.jit
def sparse_conv_bwd_weight_masked_implicit_gemm_kernel(
    grad_output,
    input,
    valid_signal_i,
    valid_signal_o,
    valid_signal_seg,
    grad_weight,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (M)
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Indice convolution backward to weight kernel using masked implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (M, Co)
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        valid_signal_i (pointer): A pointer to the valid input signal tensor of shape (L)
        valid_signal_o (pointer): A pointer to the valid output signal tensor of shape (L)
        valid_signal_seg (pointer): A pointer to the valid signal index segment tensor of shape (V + 1)
        grad_weight (pointer): A pointer to the gradient of the weight tensor of shape (Co, V, Ci)
    """
    num_blocks_co = tl.cdiv(Co, B1)
    num_blocks_ci = tl.cdiv(Ci, B2)
    block_id = tl.program_id(axis=0)
    block_id_co = block_id % num_blocks_co
    block_id_ci = block_id // num_blocks_co % num_blocks_ci
    block_id_v = block_id // (num_blocks_co * num_blocks_ci)
    
    # Create pointers for submatrices of A and B.
    valid_signal_start = tl.load(valid_signal_seg + block_id_v)
    valid_signal_seglen = tl.load(valid_signal_seg + block_id_v + 1) - valid_signal_start
    num_k = tl.cdiv(valid_signal_seglen, BK)  # Number of blocks in K dimension
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co                          # (B1,)
    offset_ci = (block_id_ci * B2 + tl.arange(0, B2)) % Ci                          # (B2,)
    offset_k = tl.arange(0, BK)                                                     # (BK,)
    
    valid_signal_i_ptr = valid_signal_i + valid_signal_start + offset_k
    valid_signal_o_ptr = valid_signal_o + valid_signal_start + offset_k
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)   
    
    # Iterate along V*Ci dimension.
    for k in range(num_k):
        # Calculate pointers to input and grad_output matrix.
        mask = offset_k < valid_signal_seglen - k * BK
        input_offset_n = tl.load(valid_signal_i_ptr, mask=mask, other=0)                            # (BK,)
        grad_output_offset_n = tl.load(valid_signal_o_ptr, mask=mask, other=0)                      # (BK,)
        input_ptr = input + (input_offset_n[:, None].to(tl.int64) * Ci + offset_ci[None, :])                     # (BK, B2)
        grad_output_ptr = grad_output + grad_output_offset_n[None, :].to(tl.int64) * Co + offset_co[:, None]     # (B1, BK)
        # Load the next block of input and grad_output.
        input_block = tl.load(input_ptr, mask=mask[:, None], other=0.0)
        grad_output_block = tl.load(grad_output_ptr, mask=mask[None, :], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
        # Advance pointers.
        valid_signal_i_ptr += BK
        valid_signal_o_ptr += BK
    c = accumulator.to(grad_output.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    grad_weight_offset_co = block_id_co * B1 + tl.arange(0, B1)
    grad_weight_offset_ci = block_id_ci * B2 + tl.arange(0, B2)
    grad_weight_ptr = grad_weight + (grad_weight_offset_co[:, None] * V * Ci + block_id_v * Ci + grad_weight_offset_ci[None, :])
    grad_weight_mask = (grad_weight_offset_co[:, None] < Co) & (grad_weight_offset_ci[None, :] < Ci)
    tl.store(grad_weight_ptr, c, mask=grad_weight_mask)


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
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
    sparse_conv_fwd_masked_implicit_gemm_kernel[grid](
        input, weight, bias, neighbor, sorted_idx, output,
        M, LOGN, LOGM, Ci, Co, V,
        valid_kernel=valid_kernel,
        valid_kernel_seg=valid_kernel_seg,
        allow_tf32=config.allow_tf32,
    )
    return output


def sparse_conv_bwd_masked_implicit_gemm(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    neighbor_bwd: torch.Tensor,
    valid_signal_i: torch.Tensor,
    valid_signal_o: torch.Tensor,
    valid_signal_seg: torch.Tensor,
    sorted_idx_bwd: Optional[torch.Tensor],
    valid_kernel_bwd: Callable[[int], torch.Tensor],
    valid_kernel_bwd_seg: Callable[[int], torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    assert neighbor_bwd.is_contiguous(), "Matrix neighbor_bwd must be contiguous"
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    
    grad_input, grad_weight, grad_bias = None, None, None
    
    # Grad for input
    if input.requires_grad:
        # Allocate output matrix output.
        grad_input = torch.empty((N, Ci), device=input.device, dtype=input.dtype)
        # Launch the kernel.
        grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']),)
        weight_bwd = weight if config.USE_ON_THE_FLY_WEIGHT_TRANSPOSE else weight.transpose(0, 2).contiguous()
        if sorted_idx_bwd is None:
            sparse_conv_fwd_implicit_gemm_kernel[grid](
                grad_output, weight_bwd, None, neighbor_bwd, grad_input,
                N, LOGM, LOGN, Co, Ci, V,
                allow_tf32=config.allow_tf32,
                TRANSPOSE_WEIGHT=config.USE_ON_THE_FLY_WEIGHT_TRANSPOSE,
            )
        else:
            sparse_conv_fwd_masked_implicit_gemm_kernel[grid](
                grad_output, weight_bwd, None, neighbor_bwd, sorted_idx_bwd, grad_input,
                N, LOGM, LOGN, Co, Ci, V,
                valid_kernel=valid_kernel_bwd,
                valid_kernel_seg=valid_kernel_bwd_seg,
                allow_tf32=config.allow_tf32,
                TRANSPOSE_WEIGHT=config.USE_ON_THE_FLY_WEIGHT_TRANSPOSE,
            )
        
    # Grad for weight
    if weight.requires_grad:
        # Allocate output matrix output.
        grad_weight = torch.empty((Co, V, Ci), device=weight.device, dtype=weight.dtype)
        # Launch the kernel.
        grid = lambda META: (triton.cdiv(Co, META['B1']) * triton.cdiv(Ci, META['B2']) * V,)
        sparse_conv_bwd_weight_masked_implicit_gemm_kernel[grid](
            grad_output, input,
            valid_signal_i,
            valid_signal_o,
            valid_signal_seg,
            grad_weight,
            M, LOGN, LOGM, Ci, Co, V,
            allow_tf32=config.allow_tf32,
        )
        
    # Grad for bias
    if bias is not None and bias.requires_grad:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias

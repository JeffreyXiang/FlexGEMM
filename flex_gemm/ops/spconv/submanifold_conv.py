from typing import *
import itertools
import torch
from torch import Tensor
from torch.autograd import Function
from . import Algorithm
from .. import spconv
from ..utils import make_conv_neighbor_offsets
from ... import kernels


__all__ = [
    "SubMConvFunction",
    "sparse_submanifold_conv",
    "sparse_submanifold_conv3d",
]


def _lookup_pytorch(key: Tensor, query: Tensor) -> Tensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    Parameters
    ----
    - `key` (Tensor): shape `(K, *key_shape)`, the array to search in
    - `query` (Tensor): shape `(..., *key_shape)`, the array to search for. `...` represents any number of batch dimensions.

    Returns
    ----
    - `indices` (Tensor): shape `(...,)` shape `(...,)` indices in `key` for each `query`. If a query is not found in key, the corresponding index will be -1.

    Notes
    ----
    `O((Q + K) * log(Q + K))` complexity, where `Q` is the number of queries and `K` is the number of keys.
    """
    num_keys, *key_shape = key.shape
    query_batch_shape = query.shape[:query.ndim - key.ndim + 1]

    unique, inverse = torch.unique(
        torch.cat([key, query.reshape(-1, *key_shape)], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:num_keys], torch.arange(num_keys, device=key.device))
    result = index.index_select(0, inverse[num_keys:]).reshape(query_batch_shape)
    return torch.where(result < num_keys, result, -1)


class SubMConvNeighborCache:
    neighbor_map: torch.Tensor

    def __init__(self, neighbor_map: torch.Tensor):
        self.neighbor_map = neighbor_map
        if spconv.ALGORITHM in [Algorithm.MASKED_IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM_SPLITK]:
            self.neighbor_map_post_process_for_masked_implicit_gemm_1()

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)

    def neighbor_map_post_process_for_masked_implicit_gemm_1(self):
        neighbor_map = self.neighbor_map

        V = self.neighbor_map.shape[1] 
        assert V <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"

        gray_code, sorted_idx, valid_signal_i, valid_signal_o = \
            kernels.triton.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
        self['gray_code'] = gray_code
        self['sorted_idx'] = sorted_idx
        self['valid_signal_i'] = valid_signal_i
        self['valid_signal_o'] = valid_signal_o

    def neighbor_map_post_process_for_masked_implicit_gemm_2(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.triton.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
        
    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_{block_size}' not in self or f'valid_kernel_seg_{block_size}' not in self:
            self.neighbor_map_post_process_for_masked_implicit_gemm_2(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_{block_size}' not in self or f'valid_kernel_seg_{block_size}' not in self:
            self.neighbor_map_post_process_for_masked_implicit_gemm_2(block_size)
        return self[f'valid_kernel_seg_{block_size}']


class SubMConvFunction(Function):
    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        kernel_offsets: torch.Tensor,
    ) -> SubMConvNeighborCache:
        coords = coords.contiguous()
        kernel_offsets = kernel_offsets.contiguous()

        if kernel_offsets.shape[1] < coords.shape[1]:
            # add batch dims to neighbor offsets if not already included
            batch_dims = coords.shape[1] - kernel_offsets.shape[1]
            kernel_offsets = torch.cat([
                torch.zeros((kernel_offsets.shape[0], batch_dims), dtype=kernel_offsets.dtype, device=kernel_offsets.device),
                kernel_offsets
            ], dim=1)

        # Compute neighbor map
        if spconv.BACKEND == spconv.Backend.TRITON:
            neighbor_map = kernels.triton.build_neighbor_map_triton(
                coords,
                kernel_offsets,
            )
        elif spconv.BACKEND == spconv.Backend.TORCH:
            neighbor_coords = coords[:, None, :] + kernel_offsets[None, :, :]          # [N, V, 4]
            neighbor_map = _lookup_pytorch(coords, neighbor_coords).to(torch.int32)
        
        return SubMConvNeighborCache(neighbor_map)
        
    @staticmethod
    def _sparse_submanifold_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous"
        N = feats.shape[0]
        Co, V, Ci = weight.shape
        
        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:        
            neighbor_map = neighbor_cache['neighbor_map']
            
            im2col = feats.index_select(0, neighbor_map.view(-1).view(dtype=torch.int32).clamp_min(0))\
                            .masked_fill((neighbor_map == -1).view(-1, 1), 0).view(N, V * Ci)
            
            # addmm
            weight = weight.view(Co, V * Ci).transpose(0, 1)
            if bias is not None:
                output = torch.addmm(bias, im2col, weight)
            else:
                output = torch.mm(im2col, weight)
        
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm(
                feats,
                weight,
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm_splitk(
                feats,
                weight,
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm(
                feats,
                weight,
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
                feats,
                weight,
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return output

    @staticmethod
    def _sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        N = feats.shape[0]
        Co, V, Ci = weight.shape

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache['neighbor_map']
            
            if feats.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
                inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != -1
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)
                
                # addmm
                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
                
            if weight.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != -1
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)
                
                # addmm
                grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.flatten(1, -2),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.flatten(1, -2),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.flatten(1, -2),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
        
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.flatten(1, -2),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        kernel_offsets: torch.Tensor,
        neighbor_cache: Optional[SubMConvNeighborCache],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SubMConvNeighborCache]:
        Co, V, Ci = weight.shape
        assert weight.shape[1] == kernel_offsets.shape[0], f"Kernel offsets length ({kernel_offsets.shape[0]}) should match weight kernel volume ({weight.shape[1]})"
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        
        # check if neighbor map is already computed
        if neighbor_cache is None:
            neighbor_cache = SubMConvFunction._compute_neighbor_cache(coords, kernel_offsets)
            
        # compute output
        output = SubMConvFunction._sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias)
        
        # save for backward
        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        
        return output, neighbor_cache
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        
        grad_input, grad_weight, grad_bias = SubMConvFunction._sparse_submanifold_conv_backward(grad_output, feats, neighbor_cache, weight, bias)
        
        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None


def sparse_submanifold_conv(
    feats: torch.Tensor,
    coords: torch.Tensor,
    kernel_offsets: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SubMConvNeighborCache] = None,
) -> Tuple[torch.Tensor, SubMConvNeighborCache]:
    """
    Sparse submanifold convolution function.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, B + D] tensor of input coordinates.
            Each row represents a coordinate, where the first B dimensions are batch indices, and the last D dimensions are spatial coordinates.
        kernel_offsets (torch.Tensor): [V, D] tensor of kernel offsets.
            V is the kernel volume, and D is the spatial dimension.
        weight (torch.Tensor): [Co, V, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConvNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.

    Returns:
        Tuple[torch.Tensor, SubMConvNeighborCache]:
            - output (torch.Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConvNeighborCache): neighbor cache for backward.
    """
    return SubMConvFunction.apply(feats, coords, kernel_offsets, neighbor_cache, weight, bias)


def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SubMConvNeighborCache] = None,
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[torch.Tensor, SubMConvNeighborCache]:
    """
    Sparse submanifold convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConv3dNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (Tuple[int, int, int]): dilation rate.

    Returns:
        Tuple[torch.Tensor, SubMConv3dNeighborCache]:
            - output (torch.Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward.
    """
    kernel_size = weight.shape[1:4]
    kernel_offsets = make_conv_neighbor_offsets(kernel_size, dilation, batch_dims=coords.shape[1] - 3, dtype=torch.int32, device=feats.device)
    return SubMConvFunction.apply(feats, coords, kernel_offsets, neighbor_cache, weight.flatten(1, 3), bias)
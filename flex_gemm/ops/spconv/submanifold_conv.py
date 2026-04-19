from typing import *
import itertools
import torch
from torch import Tensor
from torch.autograd import Function
from .. import spconv
from ... import config
from ..utils import make_conv_neighbor_offsets, init_hashmap, lookup_pytorch
from ... import kernels


__all__ = [
    "SubMConvExplicitGemmFunction",
    "SubMConvImplicitGemmFunction",
    "SubMConvImplicitGemmSplitKFunction",
    "SubMConvMaskedImplicitGemmFunction",
    "SubMConvMaskedImplicitGemmSplitKFunction",
    "sparse_submanifold_conv3d",
    "sparse_submanifold_conv",
    "sparse_submanifold_conv_any_offset",
]





class SubMConvNeighborCache:
    neighbor_map: Tensor

    def __init__(self, neighbor_map: Tensor):
        self.neighbor_map = neighbor_map

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

        if config.USE_CUDA_EXTENSION:
            gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
                kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
        else:
            gray_code, sorted_idx, valid_signal_i, valid_signal_o = \
                kernels.triton.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
        self['gray_code'] = gray_code
        self['sorted_idx'] = sorted_idx
        self['valid_signal_i'] = valid_signal_i
        self['valid_signal_o'] = valid_signal_o

    def neighbor_map_post_process_for_masked_implicit_gemm_2(self, block_size: int):
        if config.USE_CUDA_EXTENSION:
            valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        else:
            valid_kernel, valid_kernel_seg = kernels.triton.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
    
    def valid_kernel_callback(self, block_size: int) -> Tensor:
        if f'valid_kernel_{block_size}' not in self or f'valid_kernel_seg_{block_size}' not in self:
            self.neighbor_map_post_process_for_masked_implicit_gemm_2(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> Tensor:
        if f'valid_kernel_{block_size}' not in self or f'valid_kernel_seg_{block_size}' not in self:
            self.neighbor_map_post_process_for_masked_implicit_gemm_2(block_size)
        return self[f'valid_kernel_seg_{block_size}']


class SubMConvExplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SubMConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_map = neighbor_cache['neighbor_map']
        N = feats.shape[0]
        im2col = feats.index_select(0, neighbor_map.view(-1).view(dtype=torch.int32).clamp_min(0))\
                        .masked_fill((neighbor_map == -1).view(-1, 1), 0).view(N, V * Ci)

        weight_mat = weight.view(Co, V * Ci).transpose(0, 1)
        if bias is not None:
            output = torch.addmm(bias, im2col, weight_mat)
        else:
            output = torch.mm(im2col, weight_mat)

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        neighbor_map = neighbor_cache['neighbor_map']
        N = feats.shape[0]
        Co, V, Ci = weight.shape

        if feats.requires_grad:
            im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
            inv_neighbor_map = torch.flip(neighbor_map, [1])
            mask = inv_neighbor_map.view(-1) != -1
            im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Co)
            grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
        else:
            grad_input = None

        if weight.requires_grad:
            im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
            mask = neighbor_map.view(-1) != -1
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)
            grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous()
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None


class SubMConvImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SubMConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map']
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map']
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias, None


class SubMConvImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SubMConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map']
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map']
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias, None


class SubMConvMaskedImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SubMConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1()

        output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache.valid_kernel_callback,
            neighbor_cache.valid_kernel_seg_callback
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache['valid_kernel_callback'],
            neighbor_cache['valid_kernel_seg_callback'],
            neighbor_cache['valid_signal_i'],
            neighbor_cache['valid_signal_o'],
            neighbor_cache['valid_signal_seg']
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias, None


class SubMConvMaskedImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SubMConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SubMConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1()

        output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache.valid_kernel_callback,
            neighbor_cache.valid_kernel_seg_callback
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache['valid_kernel_callback'],
            neighbor_cache['valid_kernel_seg_callback'],
            neighbor_cache['valid_signal_i'],
            neighbor_cache['valid_signal_o'],
            neighbor_cache['valid_signal_seg']
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias, None


def _compute_neighbor_cache_any_offset(
    coords: Tensor,
    offsets: Tensor,
) -> SubMConvNeighborCache:
    coords = coords.contiguous()
    offsets = offsets.contiguous()

    if offsets.shape[1] < coords.shape[1]:
        # add batch dims to neighbor offsets if not already included
        batch_dims = coords.shape[1] - offsets.shape[1]
        offsets = torch.cat([
            torch.zeros((offsets.shape[0], batch_dims), dtype=offsets.dtype, device=offsets.device),
            offsets
        ], dim=1)

    # Compute neighbor map
    if config._USE_PYTORCH_FOR_TEST:
        neighbor_coords = coords[:, None, :] + offsets[None, :, :]          # [N, V, 4]
        neighbor_map = lookup_pytorch(coords, neighbor_coords).to(torch.int32)
    else:
        neighbor_map = kernels.triton.build_neighbor_map_triton(
            coords,
            offsets=offsets,
        )
        
    return SubMConvNeighborCache(neighbor_map)


def _compute_neighbor_cache_kernel_dilation(
    coords: Tensor,
    shape: Optional[torch.Size],
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...]
) -> SubMConvNeighborCache:
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert len(kernel_size) == len(dilation), "Kernel size and dilation should have the same length"

    # CUDA extension is specially optimized for 3D convolution with int32 coords.
    use_cuda_extension = config.USE_CUDA_EXTENSION \
        and shape is not None \
        and len(kernel_size) == len(dilation) == 3 \
        and coords.shape[1] == 4 \

    if config._USE_PYTORCH_FOR_TEST:
        # Debug only
        offsets = make_conv_neighbor_offsets(kernel_size, dilation, batch_dims=coords.shape[1] - len(kernel_size), dtype=torch.int32, device=coords.device)
        neighbor_coords = coords[:, None, :] + offsets[None, :, :]          # [N, V, D]
        neighbor_map = lookup_pytorch(coords, neighbor_coords).to(torch.int32)

    elif use_cuda_extension:
        # Use the CUDA extension if possible
        assert coords.dtype in [torch.int32], "Coords should be int32 for CUDA backend"

        N, C, W, H, D = shape
        hashmap_keys, hashmap_vals = init_hashmap(shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device)
        neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
            hashmap_keys, hashmap_vals, coords,
            W, H, D,
            kernel_size[0], kernel_size[1], kernel_size[2],
            dilation[0], dilation[1], dilation[2],
        )
    else:
        # Triton kernels for neighbor map construction. 
        if coords.shape[1] <= 4:
            # If no more than 4 dimensions
            neighbor_map = kernels.triton.build_neighbor_map_conv4d_triton(
                coords,
                kernel_size=(1,) * (coords.shape[1] - len(kernel_size)) + kernel_size,
                dilation=(1,) * (coords.shape[1] - len(dilation)) + dilation,
            )
        else:
            # For higher dimensions, fall back to the general kernel with offsets.
            offsets = make_conv_neighbor_offsets(kernel_size, dilation, batch_dims=coords.shape[1] - len(kernel_size), dtype=torch.int32, device=coords.device)
            neighbor_map = kernels.triton.build_neighbor_map_triton(
                coords,
                offsets=offsets
            )
            
    return SubMConvNeighborCache(neighbor_map)


def _select_submconv_function(algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] | None = None) -> Type[Function]:
    if algorithm is None:
        # Default to the global config algorithm if not specified.
        algorithm = spconv.ALGORITHM
        
    if algorithm == "explicit_gemm":
        return SubMConvExplicitGemmFunction
    if algorithm == "implicit_gemm":
        return SubMConvImplicitGemmFunction
    if algorithm == "implicit_gemm_splitk":
        return SubMConvImplicitGemmSplitKFunction
    if algorithm == "masked_implicit_gemm":
        return SubMConvMaskedImplicitGemmFunction
    if algorithm == "masked_implicit_gemm_splitk":
        return SubMConvMaskedImplicitGemmSplitKFunction
    raise ValueError(f"Invalid algorithm {algorithm}")


def sparse_submanifold_conv(
    feats: Tensor,
    coords: Tensor,
    shape: Optional[torch.Size],
    weight: Tensor,
    bias: Optional[Tensor] = None,
    neighbor_cache: Optional[SubMConvNeighborCache] = None,
    dilation: int | tuple[int, int, int] = 1,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> Tuple[Tensor, SubMConvNeighborCache]:
    """
    Sparse submanifold convolution.

    Args:
        feats (Tensor): [N, C] tensor of input features.
        coords (Tensor): [N, B + D] tensor of input coordinates.
            Each row represents a coordinate, where the first B dimensions are batch indices, and the last D dimensions are spatial coordinates.
        shape (Optional[torch.Size]): shape of the input tensor in NCWHD order. Only required when using CUDA extension.
        weight (Tensor): [Co, K1, ..., KD, Ci] tensor of weights.
        bias (Optional[Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConv3dNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (Tuple[int, int, int]): dilation rate.
        algorithm (Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"]): algorithm to use for convolution.

    Returns:
        Tuple[Tensor, SubMConv3dNeighborCache]:
            - output (Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward or future reuse of shared structures.
    """
    if isinstance(dilation, int):
        dilation = (dilation,) * (weight.ndim - 2)
    if neighbor_cache is None:
        neighbor_cache = _compute_neighbor_cache_kernel_dilation(coords, shape, weight.shape[1:-1], dilation)
    
    SubMConvFunc = _select_submconv_function(algorithm)
    output, neighbor_cache = SubMConvFunc.apply(feats, neighbor_cache, weight.flatten(1, -2), bias)
    return output, neighbor_cache


def sparse_submanifold_conv3d(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    neighbor_cache: Optional[SubMConvNeighborCache] = None,
    dilation: int | tuple[int, int, int] = (1, 1, 1),
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> tuple[Tensor, SubMConvNeighborCache]:
    """
    Sparse submanifold convolution for 3D input.

    Args:
        feats (Tensor): [N, C] tensor of input features.
        coords (Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConv3dNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (Tuple[int, int, int]): dilation rate.
        algorithm (Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"]): algorithm to use for convolution.

    Returns:
        Tuple[Tensor, SubMConv3dNeighborCache]:
            - output (Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward or future reuse of shared structures.
    """
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    if neighbor_cache is None:
        neighbor_cache = _compute_neighbor_cache_kernel_dilation(coords, shape, weight.shape[1:4], dilation)
    
    SubMConvFunc = _select_submconv_function(algorithm)
    output, neighbor_cache = SubMConvFunc.apply(feats, neighbor_cache, weight.flatten(1, -2), bias)
    return output, neighbor_cache


def sparse_submanifold_conv_any_offset(
    feats: Tensor,
    coords: Tensor,
    offsets: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    neighbor_cache: Optional[SubMConvNeighborCache] = None,
    algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] = None,
) -> Tuple[Tensor, SubMConvNeighborCache]:
    """
    Sparse submanifold convolution function with general kernel offsets.

    Args:
        feats (Tensor): [N, C] tensor of input features.
        coords (Tensor): [N, B + D] tensor of input coordinates.
            Each row represents a coordinate, where the first B dimensions are batch indices, and the last D dimensions are spatial coordinates.
        offsets (Tensor): [V, D] tensor of kernel offsets.
            V is the kernel volume, and D is the spatial dimension.
        weight (Tensor): [Co, V, Ci] tensor of weights.
        bias (Optional[Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConvNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        algorithm (Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"]): algorithm to use for convolution.

    Returns:
        Tuple[Tensor, SubMConvNeighborCache]:
            - output (Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConvNeighborCache): neighbor cache for backward or future reuse of shared structures.
    """
    # A current limitation: the gemm backward relies on the symmetry of neighbors.
    assert torch.equal(offsets, (-offsets).flip(0)), "Offsets must be symmetric."

    if neighbor_cache is None:
        neighbor_cache = _compute_neighbor_cache_any_offset(coords, offsets)
    
    SubMConvFunc = _select_submconv_function(algorithm)
    output, neighbor_cache = SubMConvFunc.apply(feats, neighbor_cache, weight.flatten(1, -2), bias)
    return output, neighbor_cache
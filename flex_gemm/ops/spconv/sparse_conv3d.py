from typing import *
import torch
from torch.autograd import Function
from . import Algorithm, SparseConv3dOutCoordAlgorithm
from .. import spconv
from ... import kernels


class SparseConv3dNeighborCache:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def compute_kernel_idx(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
        
    # NOTE:
    # valid_kernel and valid_kernel_seg are block-size dependent because
    # Triton kernels use different block-sizes during autotuning.
    #
    # We lazily compute and cache them here to:
    #   1. Avoid recomputation across multiple kernel launches
    #   2. Support multiple Triton specializations with the same neighbor cache
        
    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_seg_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_seg_{block_size}']
    
    def compute_kernel_idx_bwd(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code_bwd'], self['sorted_idx_bwd'], block_size)
        self[f'valid_kernel_bwd_{block_size}'] = valid_kernel
        self[f'valid_kernel_bwd_seg_{block_size}'] = valid_kernel_seg
    
    def valid_kernel_bwd_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_bwd_{block_size}'):
            self.compute_kernel_idx_bwd(block_size)
        return self[f'valid_kernel_bwd_{block_size}']
    
    def valid_kernel_bwd_seg_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_bwd_seg_{block_size}'):
            self.compute_kernel_idx_bwd(block_size)
        return self[f'valid_kernel_bwd_seg_{block_size}']


class SparseConv3dFunction(Function):
    @staticmethod
    def _get_output_coords(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> torch.Tensor:
        assert coords.is_contiguous(), "Coords should be contiguous"
        assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
        N, C, W, H, D = shape
        
        if coords.is_cuda:
            if spconv.OUT_COORD_ALGO == SparseConv3dOutCoordAlgorithm.HASHMAP:
                output_coords = kernels.cuda.hashmap_build_sparse_conv_out_coords(
                    coords, spconv.OUT_COORD_HASHMAP_RATIO, spconv.SERIALIZATION_MODE,
                    N, W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    stride[0], stride[1], stride[2],
                    padding[0], padding[1], padding[2],
                    dilation[0], dilation[1], dilation[2],
                )
            elif spconv.OUT_COORD_ALGO == SparseConv3dOutCoordAlgorithm.EXPAND_UNIQUE:
                output_coords = kernels.cuda.expand_unique_build_sparse_conv_out_coords(
                    coords, spconv.SERIALIZATION_MODE,
                    N, W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    stride[0], stride[1], stride[2],
                    padding[0], padding[1], padding[2],
                    dilation[0], dilation[1], dilation[2],
                )
        else:
            raise NotImplementedError("CPU version is not implemented")
        return output_coords
    
    @staticmethod
    def _get_output_coords_torch(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> torch.Tensor:
        N, C, W, H, D = shape
        Wo = (W + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        Ho = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        Do = (D + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
        
        offset = torch.meshgrid(
            -dilation[0] * torch.arange(kernel_size[0]),
            -dilation[1] * torch.arange(kernel_size[1]),
            -dilation[2] * torch.arange(kernel_size[2]),
            indexing='ij'
        )
        offset = torch.stack(offset, dim=-1).reshape(-1, 3).int().to(coords.device)
        all_potentials_out_coords = (coords + torch.tensor([0, padding[0], padding[1], padding[2]], device=coords.device).int()) \
                                     .unsqueeze(1).repeat(1, kernel_size[0] * kernel_size[1] * kernel_size[2], 1)
        all_potentials_out_coords[:, :, 1:] += offset.unsqueeze(0)                          # [N, kernel_vol, 4]
        all_potentials_out_coords = all_potentials_out_coords.reshape(-1, 4)                # [N * kernel_vol, 4]
        t_stride = torch.tensor([1, stride[0], stride[1], stride[2]], device=coords.device).int()
        valid_strided_out_coords = torch.all(all_potentials_out_coords % t_stride == 0, dim=-1)
        all_potentials_out_coords = all_potentials_out_coords[valid_strided_out_coords] // t_stride
        t_out_size = torch.tensor([Wo, Ho, Do], device=coords.device).int()
        valid_out_coords = torch.all((all_potentials_out_coords[:, 1:] >= 0) * (all_potentials_out_coords[:, 1:] < t_out_size), dim=-1)
        all_potentials_out_coords = all_potentials_out_coords[valid_out_coords]
        out_coords = torch.unique(all_potentials_out_coords, dim=0)
        
        return out_coords
    
    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        out_coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        needs_grad: bool,
        is_out_coords_given: bool = False,
    ) -> SparseConv3dNeighborCache:
        assert coords.is_contiguous(), "Coords should be contiguous"
        assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
        N, C, W, H, D = shape
        
        if spconv.ALGORITHM in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map, neighbor_map_bwd = kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
                    coords, out_coords, spconv.HASHMAP_RATIO, needs_grad,
                    N, W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    stride[0], stride[1], stride[2],
                    padding[0], padding[1], padding[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            return SparseConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
                'neighbor_map_bwd': neighbor_map_bwd,
            })
        
        elif spconv.ALGORITHM in [Algorithm.MASKED_IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map, neighbor_map_bwd = kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
                    coords, out_coords, spconv.HASHMAP_RATIO, needs_grad,
                    N, W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    stride[0], stride[1], stride[2],
                    padding[0], padding[1], padding[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            V = kernel_size[0] * kernel_size[1] * kernel_size[2]
            assert V <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
            
            if needs_grad:
                gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
                    kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
                cache = SparseConv3dNeighborCache(**{
                    'neighbor_map': neighbor_map,
                    'neighbor_map_bwd': neighbor_map_bwd,
                    'gray_code': gray_code,
                    'sorted_idx': sorted_idx,
                    'valid_signal_seg': valid_signal_seg,
                    'valid_signal_i': valid_signal_i,
                    'valid_signal_o': valid_signal_o,
                })
                if any([s != 1 for s in stride]) or is_out_coords_given:
                    # NOTE:
                    # In backward pass, workload reordering (sort by kernel mask) is only
                    # necessary when the kernel mask distribution is highly irregular.
                    #
                    # For stride == 1 and auto-generated out_coords:
                    #   - Almost all kernel offsets are valid for each input voxel
                    #   - Bwd kernel masks are nearly all valid
                    #   - Reordering is not necessary, use IGEMM
                    #
                    # For stride > 1 or user-provided out_coords:
                    #   - Kernel masks vary significantly across voxels
                    #   - Sorting by kernel mask greatly improves warp-level efficiency
                    gray_code_bwd, sorted_idx_bwd = \
                        kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(neighbor_map_bwd)
                    cache['gray_code_bwd'] = gray_code_bwd
                    cache['sorted_idx_bwd'] = sorted_idx_bwd
                return cache
            else:
                gray_code, sorted_idx = \
                    kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(neighbor_map)
                return SparseConv3dNeighborCache(**{
                    'neighbor_map': neighbor_map,
                    'neighbor_map_bwd': neighbor_map_bwd,
                    'gray_code': gray_code,
                    'sorted_idx': sorted_idx,
                })
                
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

    def _compute_neighbor_cache_torch(
        coords: torch.Tensor,
        out_coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        needs_grad: bool,
    ) -> SparseConv3dNeighborCache:
        assert spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM, "Only explicit_gemm is supported for torch implementation"
        N, C, W, H, D = shape
        M = coords.shape[0]
        L = out_coords.shape[0]
        V = kernel_size[0] * kernel_size[1] * kernel_size[2]
        assert N * W * H * D <= 2**32, "Currently, the max number of elements in a tensor is 2^32"
        OFFSET = torch.tensor([W * H * D, H * D, D, 1], device=coords.device).int()
        
        keys = (coords * OFFSET).sum(dim=-1)
        sorted_keys, indices = torch.sort(keys)
        
        # Compute neighbor coords
        offset = torch.meshgrid(
            torch.arange(kernel_size[0]) * dilation[0] - padding[0],
            torch.arange(kernel_size[1]) * dilation[1] - padding[1],
            torch.arange(kernel_size[2]) * dilation[2] - padding[2],
            indexing='ij'
        )
        offset = torch.stack(offset, dim=-1).reshape(-1, 3).int().to(coords.device)
        t_stride = torch.tensor([1, stride[0], stride[1], stride[2]], device=coords.device).int()
        neighbor_coords = (out_coords * t_stride).unsqueeze(1).repeat(1, V, 1)
        neighbor_coords[:, :, -3:] += offset.unsqueeze(0)                                    # [L, V, 4]
        neighbor_coords = neighbor_coords.reshape(-1, 4)                                     # [L * V, 4]
        neighbor_valid = (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < W) & \
                         (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < H) & \
                         (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < D)
        neighbor_keys = (neighbor_coords * OFFSET).sum(dim=-1)
        neighbor_search_indices = torch.searchsorted(sorted_keys, neighbor_keys)
        neighbor_search_indices = torch.clamp(neighbor_search_indices, 0, sorted_keys.shape[0] - 1)
        neighbor_valid &= sorted_keys[neighbor_search_indices] == neighbor_keys
        neighbor_map = torch.full((L * V,), 0xffffffff, dtype=torch.long, device=coords.device)
        in_indices = indices[neighbor_search_indices[neighbor_valid]]
        v = torch.arange(V, device=coords.device).reshape(1, -1).repeat(L, 1).flatten()[neighbor_valid]
        out_indices = torch.arange(L, device=coords.device).reshape(-1, 1).repeat(1, V).flatten()[neighbor_valid]
        neighbor_map[out_indices * V + v] = in_indices
        if needs_grad:
            neighbor_map_bwd = torch.full((M * V,), 0xffffffff, dtype=torch.long, device=coords.device)
            neighbor_map_bwd[in_indices * V + v] = out_indices
        else:
            neighbor_map_bwd = None
        return SparseConv3dNeighborCache(**{
            'neighbor_map': neighbor_map.reshape(L, -1).to(torch.uint32),
            'neighbor_map_bwd': neighbor_map_bwd.reshape(M, -1).to(torch.uint32) if needs_grad else None,
        })
        
    @staticmethod
    def _sparse_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw
        
        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:        
            neighbor_map = neighbor_cache['neighbor_map']
            
            # im2col
            im2col = torch.zeros((neighbor_map.shape[0] * V, Ci), device=feats.device, dtype=feats.dtype)
            mask = neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(neighbor_map.shape[0], V * Ci)
            
            # addmm
            weight = weight.view(Co, V * Ci).transpose(0, 1)
            if bias is not None:
                output = torch.addmm(bias, im2col, weight)
            else:
                output = torch.mm(im2col, weight)
        
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            output = kernels.triton.sparse_conv_fwd_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_conv_fwd_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
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
    def _sparse_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache['neighbor_map']
            neighbor_map_bwd = neighbor_cache['neighbor_map_bwd']
            
            if feats.requires_grad:
                # im2col
                im2col = torch.zeros((neighbor_map_bwd.shape[0] * V, Co), device=feats.device, dtype=feats.dtype)
                mask = neighbor_map_bwd.view(-1) != 0xffffffff
                im2col[mask] = grad_output[neighbor_map_bwd.view(-1).long()[mask]]
                im2col = im2col.view(neighbor_map_bwd.shape[0], V * Co)
                
                # addmm
                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
                
            if weight.requires_grad:
                # im2col
                im2col = torch.zeros((neighbor_map.shape[0] * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(neighbor_map.shape[0], V * Ci)
                
                # addmm
                grad_weight = torch.mm(im2col.t(), grad_output.view(neighbor_map.shape[0], -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['neighbor_map_bwd']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['neighbor_map_bwd']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['neighbor_map_bwd'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg'],
                neighbor_cache['sorted_idx_bwd'],
                neighbor_cache.valid_kernel_bwd_callback,
                neighbor_cache.valid_kernel_bwd_seg_callback
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
        
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['neighbor_map_bwd'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg'],
                neighbor_cache['sorted_idx_bwd'],
                neighbor_cache.valid_kernel_bwd_callback,
                neighbor_cache.valid_kernel_bwd_seg_callback
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        neighbor_cache: Optional[SparseConv3dNeighborCache],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_coords: Optional[torch.Tensor] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        need_grad = any(ctx.needs_input_grad)
        
        if out_coords is None:
            out_coords = SparseConv3dFunction._get_output_coords(coords, shape, (Kw, Kh, Kd), stride, padding, dilation)

        # check if neighbor map is already computed
        if neighbor_cache is None:
            neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, (Kw, Kh, Kd), stride, padding, dilation, need_grad)
            
        # compute output
        output = SparseConv3dFunction._sparse_conv_forward(feats, neighbor_cache, weight, bias)
        
        # save for backward
        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        
        return output, out_coords, neighbor_cache
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _, __):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        
        grad_input, grad_weight, grad_bias = SparseConv3dFunction._sparse_conv_backward(grad_output, feats, neighbor_cache, weight, bias)
        
        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None, None, None, None


def sparse_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SparseConv3dNeighborCache] = None,
    out_coords: Optional[torch.Tensor] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[torch.Tensor, torch.Tensor, SparseConv3dNeighborCache]:
    """
    Sparse convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SparseConv3dNeighborCache]): neighbor cache for this operation.
            Can be reused for multiple runs using the same coordinates.
            if None, will be computed on the fly.
        out_coords (Optional[torch.Tensor]): [M, 4] tensor of output coordinates.
            If None, will be calculated based on the input shape, kernel size, stride, padding, and dilation.
            If specified, will be used as the output coordinates.
        stride (Tuple[int, int, int]): stride of the convolution.
        padding (Tuple[int, int, int]): padding of the convolution.
        dilation (Tuple[int, int, int]): dilation rate.

    Returns:
        Tuple[torch.Tensor, SparseConv3dNeighborCache]:
            - out_feats (torch.Tensor): [M, Co] tensor of output features.
            - out_coords (torch.Tensor): [M, 4] tensor of output coordinates.
            - neighbor_cache (SparseConv3dNeighborCache): neighbor cache for this operation.
    """
    return SparseConv3dFunction.apply(feats, coords, shape, neighbor_cache, weight, bias, out_coords, stride, padding, dilation)

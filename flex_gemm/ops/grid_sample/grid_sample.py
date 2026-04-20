from typing import *
import torch
from torch.autograd import Function
from .. import grid_sample, utils
from ... import kernels
from ... import config


__all__ = [
    "grid_sample_3d",
]


class GridSample3dNearestFunction(Function):
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: Optional[torch.Size],
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples the input sparse tensor at the given points using nearest neighbor interpolation.
        
        Args:
            feats (torch.Tensor): A [N, C] tensor containing the features to sample from
            coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor. Only needed when using the CUDA backend.
            grid (torch.Tensor): A [B, L, 3] tensor containing the query points
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
        assert coords.dim() == 2 and coords.shape[1] == 4, f"Coords must be of shape [N, 4], got {coords.shape}"
        assert grid.dim() == 3 and grid.shape[2] == 3, f"Query points must be of shape [B, L, 3], got {grid.shape}"
        assert feats.shape[0] == coords.shape[0], "Number of features must match number of coordinates"
        
        N = coords.shape[0]
        B, L = grid.shape[:2]

        if config.USE_CUDA_EXTENSION and shape is not None:
            C, W, H, D = shape[-4:]
            hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(grid_sample.HASHMAP_RATIO * coords.shape[0]), coords.device)
            indices = kernels.cuda.hashmap_build_grid_sample_3d_nearest_neighbor_map(
                hashmap_keys, hashmap_vals,
                coords.int(),
                grid,
                W, H, D
            ).int()
        else:
            if grid.dtype.is_floating_point:
                grid_int = grid.round()
            indices = kernels.triton.hashmap_build_lookup_triton(
                coords.int(),
                grid_int.int().flatten(0, -2),
            ).reshape(grid.shape[:-1])


        valid = indices != -1
        indices.clamp_min_(0)
        out = valid.unsqueeze(-1) * feats.index_select(0, indices.reshape(-1)).reshape(B, L, C)
                
        ctx.save_for_backward(indices, valid)
        ctx.N = N
        ctx.C = C
        
        return out
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None]:
        indices, valid = ctx.saved_tensors
        
        grad_feats = torch.zeros(
            (ctx.N, ctx.C),
            device=grad_output.device,
            dtype=grad_output.dtype
        )
        
        grad_feats.index_add_(
            0,
            indices[valid],
            grad_output[valid].reshape(-1, ctx.C)
        )
        return grad_feats, None, None, None


class GridSample3dTrilinearFunction(Function):
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples the input sparse tensor at the given points using trilinear interpolation.
        
        Args:
            feats (torch.Tensor): A [N, C] tensor containing the features to sample from
            coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor
            grid (torch.Tensor): A [B, L, 3] tensor containing the query points
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
        assert coords.dim() == 2 and coords.shape[1] == 4, f"Coords must be of shape [N, 4], got {coords.shape}"
        assert grid.dim() == 3 and grid.shape[2] == 3, f"Query points must be of shape [B, L, 3], got {grid.shape}"
        assert feats.shape[0] == coords.shape[0], "Number of features must match number of coordinates"
        
        N = coords.shape[0]
        B, L = grid.shape[:2]
        C, W, H, D = shape[-4:]
        
        if config.USE_CUDA_EXTENSION and shape is not None:
            hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(grid_sample.HASHMAP_RATIO * coords.shape[0]), coords.device)
            indices, weight = kernels.cuda.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
                hashmap_keys, hashmap_vals,
                coords.int(),
                grid,
                W, H, D
            )
        else:
            # TODO: Implement trilinear interpolation for the Triton backend
            raise NotImplementedError("Trilinear interpolation is not yet implemented for the Triton backend")
            
        out = kernels.triton.indice_weighed_sum_fwd(
            feats,
            indices.view(-1, 8),
            weight.view(-1, 8),
        ).view(B, L, C)
        
        ctx.save_for_backward(indices, weight)
        ctx.N = N
        ctx.C = C
        
        return out
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None]:
        indices, weight = ctx.saved_tensors

        grad_feats = torch.zeros(
            (ctx.N, ctx.C),
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        grad_feats = kernels.triton.indice_weighed_sum_bwd_input(
            grad_output.reshape(-1, ctx.C).contiguous(),
            indices.view(-1, 8),
            weight.view(-1, 8),
            ctx.N,
        ).view(ctx.N, ctx.C)

        return grad_feats, None, None, None


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    grid: torch.Tensor,
    mode: Literal["nearest", "trilinear"] = "nearest",
) -> torch.Tensor:
    """
    Samples the input sparse tensor at the given points using the specified interpolation mode.
    
    Args:
        feats (torch.Tensor): A [N, C] tensor containing the features to sample from
        coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
        shape (torch.Size): The spatial shape of the sparse tensor
        grid (torch.Tensor): A [B, L, 3] tensor containing the query points
        mode (Literal["nearest", "trilinear"]): The interpolation mode to use
    
    Returns:
        torch.Tensor: A [B, L, C] tensor containing the sampled features
    """
    if mode == "nearest":
        return GridSample3dNearestFunction.apply(feats, coords, shape, grid)
    elif mode == "trilinear":
        return GridSample3dTrilinearFunction.apply(feats, coords, shape, grid)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")

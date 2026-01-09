from typing import *
import torch
from .. import kernels


@torch.no_grad()
def encode_seq(
    coords: torch.Tensor,
    shape: torch.Size,
    permute: List[int] = [0, 1, 2],
    mode: Literal['z_order', 'hilbert'] = 'z_order'
) -> torch.Tensor:
    """
    Encodes 3D coordinates into a code.

    Args:
        coords: a tensor of shape [N, 3] containing the 3D coordinates.
        shape (torch.Size): shape of the input tensor in WHD order.
        permute: the permutation of the coordinates.
        mode: the encoding mode to use.
    """
    assert coords.shape[-1] == 3 and coords.ndim == 2, "Input coordinates must be of shape [N, 3]"
    x = coords[:, permute[0]].int()
    y = coords[:, permute[1]].int()
    z = coords[:, permute[2]].int()
    
    max_coord = max(shape)
    if max_coord <= 2**10:
        codes = torch.empty(coords.shape[0], dtype=torch.int32, device=coords.device)
    elif max_coord <= 2**21:
        codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    else:
        raise ValueError(f"Coordinate value exceeds maximum supported value: {max_coord}")
    
    if mode == 'z_order':
        kernels.cuda.z_order_encode(x, y, z, codes)
    elif mode == 'hilbert':
        kernels.cuda.hilbert_encode(x, y, z, codes)
    else:
        raise ValueError(f"Unknown encoding mode: {mode}")
    
    return codes


def decode_seq(
    code: torch.Tensor,
    permute: List[int] = [0, 1, 2],
    mode: Literal['z_order', 'hilbert'] = 'z_order'
) -> torch.Tensor:
    """
    Decodes a code into 3D coordinates.

    Args:
        code: a tensor of shape [N] containing the code.
        permute: the permutation of the coordinates.
        mode: the decoding mode to use.
    """
    assert code.ndim == 1, "Input code must be of shape [N]"
    if mode == 'z_order':
        coords = kernels.cuda.z_order_decode(code)
    elif mode == 'hilbert':
        coords = kernels.cuda.hilbert_decode(code)
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    x = coords[permute.index(0)]
    y = coords[permute.index(1)]
    z = coords[permute.index(2)]
    return torch.stack([x, y, z], dim=-1)

import os
from typing import Optional
import torch

# Largest im2col block (bytes) the explicit_gemm forward builds at once. FLEX_GEMM_EXPLICIT_BLOCK_MB overrides it.
EXPLICIT_GEMM_BLOCK_BYTES = int(float(os.environ.get("FLEX_GEMM_EXPLICIT_BLOCK_MB", "512")) * 2**20)


def explicit_gemm_forward(
    feats: torch.Tensor,
    neighbor_map: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    explicit_gemm forward (im2col + addmm), built a block of output rows at a time.

    The full im2col is M * V * Ci values (27x the input for a 3x3x3 kernel) and on a large shape it alone can exceed
    GPU memory. explicit_gemm is also the one algorithm that needs no Triton tensor-core kernels (they fail to
    compile on sm_70 / V100 with Triton 3.2), so it is the fallback that has to fit. Each output row depends only on
    its own im2col row, so blocking the rows does not change the result.

    Args:
        feats (torch.Tensor): [N, Ci] input features.
        neighbor_map (torch.Tensor): [M, V] uint32 input index per output row and kernel offset, 0xffffffff = none.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] weights.
        bias (Optional[torch.Tensor]): [Co] bias.

    Returns:
        torch.Tensor: [M, Co] output features.
    """
    Co, Ci = weight.shape[0], weight.shape[-1]
    M, V = neighbor_map.shape
    weight = weight.reshape(Co, V * Ci).transpose(0, 1)
    rows = max(1, EXPLICIT_GEMM_BLOCK_BYTES // max(1, V * Ci * feats.element_size()))
    output = torch.empty((M, Co), device=feats.device, dtype=feats.dtype)
    for start in range(0, M, rows):
        nm = neighbor_map[start:start + rows].reshape(-1)
        im2col = torch.zeros((nm.shape[0], Ci), device=feats.device, dtype=feats.dtype)
        mask = nm != 0xffffffff
        im2col[mask] = feats[nm.long()[mask]]
        im2col = im2col.view(-1, V * Ci)
        if bias is not None:
            torch.addmm(bias, im2col, weight, out=output[start:start + rows])
        else:
            torch.mm(im2col, weight, out=output[start:start + rows])
    return output

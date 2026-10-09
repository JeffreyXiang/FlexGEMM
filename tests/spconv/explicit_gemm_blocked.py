import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from flex_gemm.ops.spconv import explicit_gemm


def one_shot(feats, neighbor_map, weight, bias):
    """The explicit_gemm forward before blocking: the whole im2col at once."""
    Co, Ci = weight.shape[0], weight.shape[-1]
    M, V = neighbor_map.shape
    im2col = torch.zeros((M * V, Ci), device=feats.device, dtype=feats.dtype)
    mask = neighbor_map.view(-1) != 0xffffffff
    im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
    return torch.addmm(bias, im2col.view(M, V * Ci), weight.view(Co, V * Ci).t())


def reference(feats, neighbor_map, weight, bias):
    """Plain fp32 sum over kernel offsets, no im2col."""
    M, V = neighbor_map.shape
    out = bias.float().expand(M, -1).clone()
    w = weight.float().reshape(weight.shape[0], V, -1)
    for v in range(V):
        idx = neighbor_map[:, v].long()
        valid = idx != 0xffffffff
        out[valid] += feats[idx[valid]].float() @ w[:, v].t()
    return out


def test_explicit_gemm_blocked():
    torch.manual_seed(0)
    N, M, V, Ci, Co = 5000, 4000, 27, 64, 96
    for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 5e-3)):
        feats = torch.randn(N, Ci, device='cuda', dtype=dtype)
        weight = (torch.randn(Co, 3, 3, 3, Ci, device='cuda') * 0.05).to(dtype)
        bias = torch.randn(Co, device='cuda', dtype=dtype)
        neighbor_map = torch.randint(0, N, (M, V), device='cuda', dtype=torch.int64)
        neighbor_map[torch.rand(M, V, device='cuda') < 0.6] = 0xffffffff      # most offsets empty, as in real shapes
        neighbor_map = neighbor_map.to(torch.uint32)
        ref = reference(feats, neighbor_map, weight, bias)
        scale = ref.abs().max()
        for block_bytes in (1 << 40, 64 * 1024, 1):      # one block, ~20 rows per block, one row per block
            explicit_gemm.EXPLICIT_GEMM_BLOCK_BYTES = block_bytes
            out = explicit_gemm.explicit_gemm_forward(feats, neighbor_map, weight, bias)
            err = ((out.float() - ref).abs().max() / scale).item()
            print(f'{str(dtype):14} block={block_bytes:>14} rel err vs reference {err:.2e}')
            assert out.shape == (M, Co) and err < tol, f'{dtype} block={block_bytes}: rel err {err:.2e}'
        old = ((one_shot(feats, neighbor_map, weight, bias).float() - ref).abs().max() / scale).item()
        print(f'{str(dtype):14} one-shot (before)      rel err vs reference {old:.2e}')
    print('PASS')


if __name__ == '__main__':
    test_explicit_gemm_blocked()

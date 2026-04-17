import pytest
import torch

import flex_gemm.kernels as kernels
from flex_gemm.ops import spconv
from flex_gemm.ops.spconv import SubMConv3dFunction
from utils import sphere_coords


def _normalize_int_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # Normalize int32/uint32 tensors so -1 and 0xffffffff compare equal.
    t64 = tensor.detach().cpu().to(torch.int64)
    if t64.numel() == 0:
        return t64
    return torch.where(t64 < 0, t64 + 2**32, t64)


def _assert_tensor_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    torch.testing.assert_close(
        _normalize_int_tensor(lhs),
        _normalize_int_tensor(rhs),
        rtol=0,
        atol=0,
    )


def _compute_neighbor_cache(
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    algorithm: spconv.Algorithm,
    use_triton: bool,
):
    spconv.set_algorithm(algorithm)
    spconv.set_use_triton_neighbor_map(use_triton)
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    return SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize(
    "algorithm",
    [
        spconv.Algorithm.EXPLICIT_GEMM,
        spconv.Algorithm.IMPLICIT_GEMM,
        spconv.Algorithm.IMPLICIT_GEMM_SPLITK,
        spconv.Algorithm.MASKED_IMPLICIT_GEMM,
        spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    ],
)
def test_triton_neighbor_cache_matches_cuda(algorithm) -> None:
    has_cuda_ext = (
        hasattr(kernels, "cuda")
        and kernels.cuda is not None
        and hasattr(kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map_cuda")
    )
    feats, coords, shape = sphere_coords(32, 16, dtype=torch.float16)
    weight = torch.empty(16, 3, 3, 3, 16, device=feats.device, dtype=feats.dtype)

    original_algorithm = spconv.ALGORITHM
    original_use_triton = spconv.USE_TRITON_NEIGHBOR_MAP

    cache_triton = _compute_neighbor_cache(
        coords, shape, weight, algorithm, use_triton=True
    )

    if not has_cuda_ext:
        neighbor_map = cache_triton["neighbor_map"]
        print(
            f"[triton-only] algo={algorithm} "
            f"neighbor_map.shape={tuple(neighbor_map.shape)} "
            f"dtype={neighbor_map.dtype}"
        )
        if algorithm in (
            spconv.Algorithm.MASKED_IMPLICIT_GEMM,
            spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
        ):
            print(
                f"[triton-only] gray_code.shape={tuple(cache_triton['gray_code'].shape)} "
                f"sorted_idx.shape={tuple(cache_triton['sorted_idx'].shape)}"
            )
        return

    cache_cuda = _compute_neighbor_cache(
        coords, shape, weight, algorithm, use_triton=False
    )

    _assert_tensor_equal(
        cache_triton["neighbor_map"], cache_cuda["neighbor_map"]
    )

    if algorithm in (
        spconv.Algorithm.MASKED_IMPLICIT_GEMM,
        spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    ):
        print(f"[triton vs cuda] algo={algorithm} neighbor_map match, checking post-processing outputs and callbacks...")
        _assert_tensor_equal(
            cache_triton["gray_code"], cache_cuda["gray_code"]
        )
        _assert_tensor_equal(
            cache_triton["sorted_idx"], cache_cuda["sorted_idx"]
        )
        _assert_tensor_equal(
            cache_triton["valid_signal_i"], cache_cuda["valid_signal_i"]
        )
        _assert_tensor_equal(
            cache_triton["valid_signal_o"], cache_cuda["valid_signal_o"]
        )
        _assert_tensor_equal(
            cache_triton["valid_signal_seg"], cache_cuda["valid_signal_seg"]
        )

        block_size = 64
        _assert_tensor_equal(
            cache_triton.valid_kernel_callback(block_size),
            cache_cuda.valid_kernel_callback(block_size),
        )
        _assert_tensor_equal(
            cache_triton.valid_kernel_seg_callback(block_size),
            cache_cuda.valid_kernel_seg_callback(block_size),
        )
        print("[triton vs cuda] algo={algorithm} neighbor_map and callbacks match")

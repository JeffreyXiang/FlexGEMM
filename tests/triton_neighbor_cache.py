import os
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


def _time_cuda_ms(fn, warmup: int = 20, iters: int = 100) -> float:
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_t = time.time()
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    for _ in range(iters):
        fn()
    # end.record()
    torch.cuda.synchronize()
    end_t = time.time()
    # return start.elapsed_time(end) / iters
    return (end_t - start_t) * 1000 / iters


def _compute_neighbor_cache(
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    algorithm: spconv.Algorithm,
    use_triton: bool,
):
    spconv.set_algorithm(algorithm)
    spconv.set_backend(spconv.Backend.TRITON if use_triton else spconv.Backend.CUDA)
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    if algorithm in (
        spconv.Algorithm.MASKED_IMPLICIT_GEMM,
        spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    ):
        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_2(block_size=64)  # Ensure the callbacks are populated for testing.
    return neighbor_cache


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
        # _assert_tensor_equal(
        #     cache_triton["valid_signal_seg"], cache_cuda["valid_signal_seg"]
        # )

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_triton_neighbor_map_post_process_2_matches_cuda() -> None:
    has_cuda_ext = (
        hasattr(kernels, "cuda")
        and kernels.cuda is not None
        and hasattr(kernels.cuda, "neighbor_map_post_process_for_masked_implicit_gemm_2")
    )
    if not has_cuda_ext:
        pytest.skip("CUDA extension is required for comparison")

    feats, coords, shape = sphere_coords(32, 16, dtype=torch.float16)
    weight = torch.empty(16, 3, 3, 3, 16, device=feats.device, dtype=feats.dtype)

    original_algorithm = spconv.ALGORITHM
    original_backend = spconv.BACKEND
    try:
        cache_cuda = _compute_neighbor_cache(
            coords, shape, weight, spconv.Algorithm.MASKED_IMPLICIT_GEMM, use_triton=False
        )
    finally:
        spconv.set_algorithm(original_algorithm)
        spconv.set_backend(original_backend)

    gray_code = cache_cuda["gray_code"]
    sorted_idx = cache_cuda["sorted_idx"]

    for block_size in (32, 64, 128):
        triton_valid, triton_seg = (
            kernels.triton.neighbor_map_post_process_for_masked_implicit_gemm_2(
                gray_code, sorted_idx, block_size
            )
        )
        cuda_valid, cuda_seg = (
            kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(
                gray_code, sorted_idx, block_size
            )
        )
        _assert_tensor_equal(triton_valid, cuda_valid)
        _assert_tensor_equal(triton_seg, cuda_seg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
@pytest.mark.parametrize(
    "res,ch,algorithm",
    [
        (512, 32, spconv.Algorithm.IMPLICIT_GEMM),
        (512, 32, spconv.Algorithm.MASKED_IMPLICIT_GEMM),
        (512, 32, spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK),
    ],
)
def test_triton_neighbor_cache_benchmark(res: int, ch: int, algorithm: spconv.Algorithm) -> None:
    has_cuda_ext = (
        hasattr(kernels, "cuda")
        and kernels.cuda is not None
        and hasattr(kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map_cuda")
    )
    if not has_cuda_ext:
        pytest.skip("CUDA extension is required for comparison")

    feats, coords, shape = sphere_coords(res, ch, dtype=torch.float16)
    weight = torch.empty(ch, 3, 3, 3, ch, device=feats.device, dtype=feats.dtype)

    original_algorithm = spconv.ALGORITHM
    original_backend = spconv.BACKEND
    try:
        triton_ms = _time_cuda_ms(
            lambda: _compute_neighbor_cache(coords, shape, weight, algorithm, use_triton=True),
            warmup=10,
            iters=50,
        )
        cuda_ms = _time_cuda_ms(
            lambda: _compute_neighbor_cache(coords, shape, weight, algorithm, use_triton=False),
            warmup=10,
            iters=50,
        )
    finally:
        spconv.set_algorithm(original_algorithm)
        spconv.set_backend(original_backend)

    print(
        f"\n[neighbor_cache benchmark] res={res}, ch={ch}, algo={algorithm}, "
        f"triton={triton_ms:.3f} ms, cuda={cuda_ms:.3f} ms"
    )

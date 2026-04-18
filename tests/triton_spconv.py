import os

import pytest
import torch

import flex_gemm.kernels as kernels
from flex_gemm.ops import spconv
from flex_gemm.ops.spconv.submanifold_conv import sparse_submanifold_conv
from flex_gemm.ops.spconv.submanifold_conv import sparse_submanifold_conv3d as sparse_submanifold_conv3d_triton
from flex_gemm.ops.spconv.submanifold_conv3d import sparse_submanifold_conv3d as sparse_submanifold_conv3d_cuda
from flex_gemm.ops.utils import make_conv_neighbor_offsets
from utils import sphere_coords


def _make_sparse_coords(
    spatial_shape: tuple[int, ...],
    batch_size: int,
    density: float,
    device: torch.device,
) -> torch.Tensor:
    axes = [torch.arange(size, device=device) for size in spatial_shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, len(spatial_shape))
    if grid.numel() == 0:
        return torch.empty((0, 1 + len(spatial_shape)), device=device, dtype=torch.int32)

    keep = max(1, int(grid.shape[0] * density))
    perm = torch.randperm(grid.shape[0], device=device)
    coords = grid.index_select(0, perm[:keep]).to(torch.int32)

    if batch_size > 1:
        batch = torch.randint(0, batch_size, (coords.shape[0], 1), device=device, dtype=torch.int32)
    else:
        batch = torch.zeros((coords.shape[0], 1), device=device, dtype=torch.int32)

    return torch.cat([batch, coords], dim=1).contiguous()


def _time_cuda_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_triton_spconv_explicit_gemm_matches_torch_backend() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    spatial_shape = (5, 4, 3, 2)
    coords = _make_sparse_coords(spatial_shape, batch_size=1, density=0.4, device=device)
    feats = torch.randn(coords.shape[0], 8, device=device, dtype=torch.float16)

    kernel_size = (3, 1, 3, 1)
    dilation = (1, 1, 1, 1)
    kernel_offsets = make_conv_neighbor_offsets(
        kernel_size,
        dilation,
        batch_dims=1,
        dtype=torch.int32,
        device=device,
    )
    weight = torch.randn(12, kernel_offsets.shape[0], feats.shape[1], device=device, dtype=feats.dtype)
    bias = torch.randn(12, device=device, dtype=feats.dtype)

    original_backend = spconv.BACKEND
    original_algorithm = spconv.ALGORITHM
    try:
        spconv.set_algorithm(spconv.Algorithm.EXPLICIT_GEMM)
        spconv.set_backend(spconv.Backend.TRITON)
        triton_out, _ = sparse_submanifold_conv(
            feats,
            coords,
            kernel_offsets,
            weight,
            bias,
        )

        # set_backend does not accept TORCH, so switch directly for reference.
        spconv.BACKEND = spconv.Backend.TORCH
        torch_out, _ = sparse_submanifold_conv(
            feats,
            coords,
            kernel_offsets,
            weight,
            bias,
        )
    finally:
        spconv.ALGORITHM = original_algorithm
        spconv.BACKEND = original_backend

    torch.testing.assert_close(triton_out, torch_out, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_triton_spconv_vs_cuda_spconv3d_benchmark() -> None:
    has_cuda_ext = (
        hasattr(kernels, "cuda")
        and kernels.cuda is not None
        and hasattr(kernels.cuda, "hashmap_build_submanifold_conv_neighbour_map_cuda")
    )
    if not has_cuda_ext:
        pytest.skip("CUDA extension is required for neighbor-map comparison")

    res = 64
    ch = 256
    feats, coords, shape = sphere_coords(res, ch, dtype=torch.float16)
    print(f"Generated {feats.shape[0]} points for benchmarking")
    weight3d = torch.randn(ch, 3, 3, 3, ch, device=feats.device, dtype=feats.dtype)
    bias = torch.randn(ch, device=feats.device, dtype=feats.dtype)

    original_backend = spconv.BACKEND
    original_algorithm = spconv.ALGORITHM
    try:
        spconv.set_algorithm(spconv.Algorithm.IMPLICIT_GEMM)

        spconv.set_backend(spconv.Backend.TRITON)
        triton_ms = _time_cuda_ms(
            lambda: sparse_submanifold_conv3d_triton(
                feats,
                coords,
                weight3d,
                bias,
            )[0]
        )

        spconv.set_backend(spconv.Backend.CUDA)
        cuda_ms = _time_cuda_ms(
            lambda: sparse_submanifold_conv3d_cuda(
                feats,
                coords,
                shape,
                weight3d,
                bias,
            )[0]
        )
    finally:
        spconv.ALGORITHM = original_algorithm
        spconv.BACKEND = original_backend

    speedup = cuda_ms / triton_ms if triton_ms > 0 else float("inf")
    print(
        f"\n[triton_spconv benchmark] triton={triton_ms:.3f} ms, "
        f"cuda={cuda_ms:.3f} ms, speedup(cuda/triton)={speedup:.2f}x"
    )

if __name__ == "__main__":
    test_triton_spconv_explicit_gemm_matches_torch_backend()
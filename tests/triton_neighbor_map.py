import itertools
import os

import pytest
import torch

from flex_gemm.kernels.triton.neighbor_map import (
    build_neighbor_map_triton,
)
from flex_gemm.ops.utils import make_conv_neighbor_offsets
from utils import sphere_coords


def _time_cuda_ms(fn, warmup: int = 20, iters: int = 100) -> float:
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


def _reference_neighbor_map(
    coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> torch.Tensor:
    coords_cpu = coords.cpu().to(torch.int64)
    n_coords = coords_cpu.shape[0]
    coord_to_idx = {tuple(coords_cpu[i].tolist()): i for i in range(n_coords)}

    ranges = [
        range(-(k // 2) * d, (k // 2 + 1) * d, d)
        for k, d in zip(kernel_size, dilation)
    ]
    offsets = list(itertools.product(*ranges))

    neighbor_map = torch.full(
        (n_coords, len(offsets)), -1, dtype=torch.int32
    )
    for i in range(n_coords):
        base = coords_cpu[i]
        for j, offset in enumerate(offsets):
            neighbor = (
                base[0].item() + offset[0],
                base[1].item() + offset[1],
                base[2].item() + offset[2],
            )
            neighbor_map[i, j] = coord_to_idx.get(neighbor, -1)

    return neighbor_map


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_build_submanifold_conv3d_neighbour_map_matches_reference() -> None:
    device = torch.device("cuda")
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(2, device=device),
            torch.arange(2, device=device),
            torch.arange(2, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32)

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    out = build_neighbor_map_triton(
        coords, 
        offsets=make_conv_neighbor_offsets(kernel_size, dilation, dtype=torch.int32, device=device)
    )

    expected = _reference_neighbor_map(
        coords, kernel_size, dilation
    )

    assert out.shape == (coords.shape[0], 27)
    assert out.dtype == torch.int32
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_neighbor_map_triton_dense_speed_benchmark() -> None:
    device = torch.device("cuda")
    res = 128
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32).contiguous()
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    build_ms = _time_cuda_ms(
        lambda: build_neighbor_map_triton(
            coords, 
            offsets=make_conv_neighbor_offsets(kernel_size, dilation, dtype=torch.int32, device=device)
        ),
        warmup=10,
        iters=50,
    )

    out = build_neighbor_map_triton(
        coords, 
        offsets=make_conv_neighbor_offsets(kernel_size, dilation, dtype=torch.int32, device=device)
    )
    assert out.shape == (n_coords, 27)

    neighbor_qps = (n_coords * 27) / (build_ms * 1e-3)
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_neighbor_map_triton_shuffled_speed_benchmark() -> None:
    device = torch.device("cuda")
    res = 128
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = grid.reshape(-1, 3).to(torch.int32).contiguous()
    coords = coords.index_select(0, torch.randperm(coords.shape[0], device=device)) # Randomly select half of the coordinates to create sparsity
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    build_ms = _time_cuda_ms(
        lambda: build_neighbor_map_triton(
            coords, 
            offsets=make_conv_neighbor_offsets(kernel_size, dilation, dtype=torch.int32, device=device)
        ),
        warmup=10,
        iters=50,
    )

    out = build_neighbor_map_triton(
        coords, 
        offsets=make_conv_neighbor_offsets(kernel_size, dilation, dtype=torch.int32, device=device)
    )
    assert out.shape == (n_coords, 27)

    neighbor_qps = (n_coords * 27) / (build_ms * 1e-3)
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_neighbor_map_triton_sparse_speed_benchmark() -> None:
    device = torch.device("cuda")
    res = 512
    _, coords, _ = sphere_coords(res, 16, dtype=torch.float16)
    n_coords = coords.shape[0]

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)
    build_ms = _time_cuda_ms(
        lambda: build_neighbor_map_triton(
            coords, 
            offsets=make_conv_neighbor_offsets(kernel_size, dilation, batch_dims=coords.shape[1] - 3, dtype=coords.dtype, device=device)
        ),
        warmup=10,
        iters=50,
    )

    out = build_neighbor_map_triton(
        coords, 
        offsets=make_conv_neighbor_offsets(kernel_size, dilation, batch_dims=coords.shape[1] - 3, dtype=coords.dtype, device=device)
    )
    assert out.shape == (n_coords, 27)

    neighbor_qps = (n_coords * 27) / (build_ms * 1e-3)
    print(
        f"\n[neighbor_map benchmark] n_coords={n_coords}, kernel={kernel_size}, "
        f"dilation={dilation}, build={build_ms:.3f} ms, neighbor_qps={neighbor_qps:,.0f}/s"
    )

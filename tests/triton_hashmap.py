import os

import pytest
import torch

from flex_gemm.kernels.triton import hashmap_build_triton, hashmap_lookup_triton


def _make_unique_keys(n: int, dim: int, device: torch.device) -> torch.Tensor:
    # Use a deterministic linear transform so each row is unique.
    base = torch.arange(n, device=device, dtype=torch.int32)
    cols = [base * (97 + i * 13) + (17 + i) for i in range(dim)]
    return torch.stack(cols, dim=1)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_hashmap_build_triton_basic_properties() -> None:
    device = torch.device("cuda")
    n_keys = 512
    keys = _make_unique_keys(n_keys, dim=4, device=device)

    hashmap = hashmap_build_triton(keys)

    assert hashmap.ndim == 1
    assert hashmap.device.type == "cuda"
    assert hashmap.dtype == torch.int32

    expected_size = 1 << ((n_keys - 1).bit_length() + 1)
    assert hashmap.shape[0] == expected_size

    occupied = (hashmap >= 0).sum().item()
    assert occupied == n_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_hashmap_lookup_triton_matches_reference() -> None:
    device = torch.device("cuda")
    n_keys = 1024
    key_dim = 4
    keys = _make_unique_keys(n_keys, dim=key_dim, device=device)

    # Half queries are present keys, half are guaranteed missing keys.
    present_idx = torch.tensor([0, 1, 17, 123, 511, 700, 1023], device=device)
    present_queries = keys[present_idx]
    missing_queries = _make_unique_keys(8, dim=key_dim, device=device) + 10_000_000
    queries = torch.cat([present_queries, missing_queries], dim=0)

    hashmap = hashmap_build_triton(keys)
    out = hashmap_lookup_triton(hashmap, keys, queries)

    assert out.dtype == torch.int64
    assert out.shape == (queries.shape[0],)

    expected = torch.full((queries.shape[0],), -1, dtype=torch.int64, device=device)
    expected[: present_idx.numel()] = present_idx.to(torch.int64)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.skipif(os.getenv("RUN_BENCHMARKS") != "1", reason="Set RUN_BENCHMARKS=1 to run benchmark tests")
def test_hashmap_triton_speed_benchmark() -> None:
    device = torch.device("cuda")
    n_keys = 1024 * 1024
    key_dim = 4

    keys = _make_unique_keys(n_keys, dim=key_dim, device=device)
    n_queries = n_keys // 2
    present_queries = keys[:n_queries]
    missing_queries = _make_unique_keys(n_queries, dim=key_dim, device=device) + 20_000_000
    queries = torch.cat([present_queries, missing_queries], dim=0)

    build_ms = _time_cuda_ms(lambda: hashmap_build_triton(keys), warmup=20, iters=50)
    hashmap = hashmap_build_triton(keys)
    lookup_ms = _time_cuda_ms(lambda: hashmap_lookup_triton(hashmap, keys, queries), warmup=20, iters=100)

    out = hashmap_lookup_triton(hashmap, keys, queries)
    assert (out[:n_queries] >= 0).all()
    assert (out[n_queries:] == -1).all()

    qps = queries.shape[0] / (lookup_ms * 1e-3)
    print(
        f"\n[hashmap benchmark] n_keys={n_keys}, n_queries={queries.shape[0]}, dim={key_dim}, "
        f"build={build_ms:.3f} ms, lookup={lookup_ms:.3f} ms, lookup_qps={qps:,.0f}/s"
    )

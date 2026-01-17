import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import torch
import flex_gemm
from flex_gemm.ops.spconv import SparseConv3dFunction
from utils import sphere_coords, benchmark_kernel

flex_gemm.ops.spconv.OUT_COORD_ALGO = flex_gemm.ops.spconv.SparseConv3dOutCoordAlgorithm.HASHMAP
flex_gemm.ops.spconv.SERIALIZATION_MODE = flex_gemm.ops.spconv.SerializationMode.BXYZ
    
def egemm_torch_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache_torch(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache
    

def egemm_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache
    
    
def igemm_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache
    

def igemmk_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache
    

def migemm_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache
    

def migemmk_prepare_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation, needs_grad
):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, needs_grad)
    return neighbor_cache


def test_neighbor_cache():
    RES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    test_cases = []
    for res in RES:
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), False))
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), True))
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (2, 2, 2), (2, 2, 2), False))
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (2, 2, 2), (2, 2, 2), True))
        test_cases.append((res, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), False))
        test_cases.append((res, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), True))
    
    # List of custom kernel functions.
    kernel_functions = {
        'egemm_torch': (egemm_torch_prepare_fn, None),
        'egemm': (egemm_prepare_fn, None),
        # 'igemm': (igemm_prepare_fn, None),
        # 'igemmk': (igemmk_prepare_fn, None),
        'migemm': (migemm_prepare_fn, None),
        # 'migemmk': (migemmk_prepare_fn, None),
    }
    
    results = {}
    for res, ksize, stride, padding, dilation, needs_grad in tqdm(test_cases):

        # Create random input matrices.
        feats, coords, shape = sphere_coords(res, 0, dtype=torch.float16)
        args = {
            'coords': coords,
            'shape': shape,
            'ksize': ksize,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'needs_grad': needs_grad,
        }

        config_key = f"res={res},K={ksize},S={stride},P={padding},D={dilation},grad={needs_grad}"
        results[config_key] = []

        C_ref = egemm_torch_prepare_fn(**args).neighbor_map

        # Benchmark each custom kernel.
        for kernel_fn, prepare_fn in kernel_functions.values():
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            C_kernel = C_kernel.neighbor_map
            assert torch.equal(C_kernel, C_ref), f"Neighbor cache mismatch for {kernel_fn.__name__}."
            results[config_key].append(f'{avg_time:.3f}/{memory:.3f}G')

    # Print results as a formatted table.
    print("\Sparse Conv Neighbor Cache Benchmark Results")
    print("-" * 180)
    items = [f'{"settings":<80}']
    for f in kernel_functions.keys():
        items.append(f'{f:<20}')
    print(' | '.join(items))
    print("-" * 180)
    for k, v in results.items():
        items = [f'{k:<80}']
        items.extend([f'{x:<20}' for x in v])
        print(' | '.join(items))
        

if __name__ == "__main__":
    test_neighbor_cache()

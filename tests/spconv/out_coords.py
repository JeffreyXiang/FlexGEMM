import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from tqdm import tqdm
import flex_gemm
from flex_gemm.ops.spconv import SparseConv3dFunction
from utils import sphere_coords, benchmark_kernel


flex_gemm.ops.spconv.OUT_COORD_ALGO = flex_gemm.ops.spconv.SparseConv3dOutCoordAlgorithm.HASHMAP
flex_gemm.ops.spconv.SERIALIZATION_MODE = flex_gemm.ops.spconv.SerializationMode.BXYZ


def torch_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation
) -> torch.Tensor:
    return SparseConv3dFunction._get_output_coords_torch(
        coords, shape, ksize, stride, padding, dilation
    )
    

def cuda_fn(
    coords: torch.Tensor, shape: torch.Size,
    ksize, stride, padding, dilation
) -> torch.Tensor:
    return SparseConv3dFunction._get_output_coords(
        coords, shape, ksize, stride, padding, dilation
    )


def test_out_coords():
    # Matrix dimensions.
    RES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    test_cases = []
    for res in RES:
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)))
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (2, 2, 2), (2, 2, 2)))
        test_cases.append((res, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1)))
    
    results = {}
    for res, ksize, stride, padding, dilation in tqdm(test_cases):
        feats, coords, shape = sphere_coords(res, 0)
        args = {
            'coords': coords,
            'shape': shape,
            'ksize': ksize,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }
        config_key = f"res={res},ksize={ksize},stride={stride},padding={padding},dilation={dilation}"
        
        # Benchmark
        avg_time_torch, memory_torch, C_torch = benchmark_kernel(torch_fn, **args)
        flex_gemm.ops.spconv.OUT_COORD_ALGO = flex_gemm.ops.spconv.SparseConv3dOutCoordAlgorithm.HASHMAP
        avg_time_cuda_hasmap, memory_cuda_hasmap, C_cuda_hasmap = benchmark_kernel(cuda_fn, **args)
        flex_gemm.ops.spconv.OUT_COORD_ALGO = flex_gemm.ops.spconv.SparseConv3dOutCoordAlgorithm.EXPAND_UNIQUE
        avg_time_cuda_expand, memory_cuda_expand, C_cuda_expand = benchmark_kernel(cuda_fn, **args)
                
        # Compare results
        assert torch.all(C_torch == C_cuda_hasmap), f"Output coordinates mismatch for {config_key} using HASHMAP"
        assert torch.all(C_torch == C_cuda_expand), f"Output coordinates mismatch for {config_key} using EXPAND_UNIQUE"
        results[config_key] = [
            f'{avg_time_torch:.3f}/{memory_torch:.3f}G',
            f'{avg_time_cuda_hasmap:.3f}/{memory_cuda_hasmap:.3f}G',
            f'{avg_time_cuda_expand:.3f}/{memory_cuda_expand:.3f}G'
        ]
        
    # Print results as a formatted table.
    print("\nSparse Conv3d Output Coords Benchmark Results")
    print("-" * 180)
    items = [f'{"settings":<80}', f'{"torch":<20}', f'{"cuda (HASHMAP)":<20}', f'{"cuda (EXPAND_UNIQUE)":<20}']
    print(' | '.join(items))
    print("-" * 180)
    for k, v in results.items():
        items = [f'{k:<80}']
        items.extend([f'{x:<20}' for x in v])
        print(' | '.join(items))


if __name__ == '__main__':
    test_out_coords()

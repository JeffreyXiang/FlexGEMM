
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


def test_out_coords(start=0, resolution_power=10, device="cuda"):
    # Matrix dimensions.
    RES = [2**x for x in range(start, start+resolution_power+1)]
    
    test_cases = []
    for res in RES:
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)))
        test_cases.append((res, (3, 3, 3), (1, 1, 1), (2, 2, 2), (2, 2, 2)))
        test_cases.append((res, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1)))
    
    sequence_no = 0
    results = {}
    for res, ksize, stride, padding, dilation in tqdm(test_cases):
        feats, coords, shape = sphere_coords(res, 0, device=device)
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

        args["coords"] = args["coords"].to("cuda")
        # Use the gpu to do this part as the cpu is way too inefficient
        avg_time_torch, memory_torch, C_torch = benchmark_kernel(torch_fn, **args)
        args["coords"] = args["coords"].to(device)
        C_torch = C_torch.to(device)

        flex_gemm.ops.spconv.OUT_COORD_ALGO = flex_gemm.ops.spconv.SparseConv3dOutCoordAlgorithm.HASHMAP
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            avg_time_cuda_hasmap, memory_cuda_hasmap, C_cuda_hasmap = benchmark_kernel(cuda_fn, **args)

        print(f"RESOLUTION: {res}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        prof.export_chrome_trace(f"benchmarks/trace_{res}_{sequence_no}.json")
        sequence_no+=1

        # Compare results
        assert torch.all(C_torch == C_cuda_hasmap), f"Output coordinates mismatch for {config_key} using HASHMAP"
        


if __name__ == '__main__':
    test_out_coords(start=4, resolution_power=4, device="cpu")

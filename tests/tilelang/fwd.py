import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import torch
import spconv.pytorch as spconv
import flex_gemm
from flex_gemm.ops.spconv import SparseConv3dFunction
from utils import sphere_coords, calc_err, benchmark_kernel, lexsort, get_device_max_flops


def torch_conv3d_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor,
                      ksize, stride, padding, dilation, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    
    # Init module.
    module = torch.nn.Conv3d(Ci, Co, ksize, stride=stride, padding=padding, dilation=dilation, bias=True).cuda().to(feats.dtype)
    module.weight.data.copy_(weight.permute(0, 4, 1, 2, 3).contiguous())
    module.bias.data.zero_()
    
    dense_feats = torch.zeros(shape, device=feats.device, dtype=feats.dtype)
    dense_feats[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = feats
    
    return {
        'module': module,
        'input': dense_feats,
        'bias': bias,
    }
    

def torch_conv3d_kernel_fn(module, input, bias):
    output = module(input)
    coords = torch.any(output.abs() > 1e-6, dim=1).nonzero(as_tuple=False)
    feats = output[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] + bias
    return [feats, coords]


def spconv_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor,
                      ksize, stride, padding, dilation, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    
    # Init module.
    module = spconv.SparseConv3d(Ci, Co, ksize, stride, padding, dilation, indice_key='test', algo=spconv.ConvAlgo.MaskSplitImplicitGemm).cuda().to(feats.dtype)
    module.weight.data.copy_(weight)
    module.bias.data.copy_(bias)
    
    # Init input tensor and its cache
    input_spconv = spconv.SparseConvTensor(feats, coords, shape[-3:], shape[0])
    out_spconv = module(input_spconv)
    input_spconv.indice_dict = out_spconv.indice_dict.copy()
    
    return {
        'module': module,
        'input': input_spconv,
    }
    

def spconv_kernel_fn(module, input):
    out = module(input)
    return [out.features, out.indices]


def egemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, False)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    
    
def igemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, False)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def igemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, False)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, False)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, ksize, stride, padding, dilation, False)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def flex_gemm_kernel_fn(**kwargs):
    feats = SparseConv3dFunction._sparse_conv_forward(**kwargs)
    coords = kwargs['neighbor_cache'].out_coords
    return [feats, coords]


def flex_gemm_T_kernel_fn(**kwargs):
    flex_gemm.ops.set_backend(flex_gemm.ops.Backend.TILELANG)
    feats = SparseConv3dFunction._sparse_conv_forward(**kwargs)
    flex_gemm.ops.set_backend(flex_gemm.ops.Backend.TRITON)
    coords = kwargs['neighbor_cache'].out_coords
    return [feats, coords]


def test_conv_fwd():
    # Matrix dimensions.
    configs = [
        {'RES': 8, 'C': 1024, 'B': 256},
        {'RES': 16, 'C': 1024, 'B': 64},
        {'RES': 32, 'C': 1024, 'B': 16},
        {'RES': 64, 'C': 1024, 'B': 4},
        {'RES': 128, 'C': 512, 'B': 4},
        {'RES': 256, 'C': 256, 'B': 2},
        {'RES': 512, 'C': 128, 'B': 1},
        # {'RES': 1024, 'C': 64, 'B': 1},
        # {'RES': 2048, 'C': 32, 'B': 1},
    ]
    
    test_cases = []
    for config in configs:
        test_cases.append({**config, 'ksize': (3, 3, 3), 'stride': (1, 1, 1), 'padding': (1, 1, 1), 'dilation': (1, 1, 1)})
        # test_cases.append({**config, 'ksize': (2, 2, 2), 'stride': (2, 2, 2), 'padding': (0, 0, 0), 'dilation': (1, 1, 1)})
    
    # List of custom kernel functions.
    kernel_functions = {
        # 'dense': (torch_conv3d_kernel_fn, torch_conv3d_prepare_fn),
        # 'spconv': (spconv_kernel_fn, spconv_prepare_fn),
        # 'egemm': (flex_gemm_kernel_fn, egemm_prepare_fn),
        'igemm': (flex_gemm_kernel_fn, igemm_prepare_fn),
        'igemm_t': (flex_gemm_T_kernel_fn, igemm_prepare_fn),
        # 'igemmk': (flex_gemm_kernel_fn, igemmk_prepare_fn),
        'migemm': (flex_gemm_kernel_fn, migemm_prepare_fn),
        'migemm_t': (flex_gemm_T_kernel_fn, migemm_prepare_fn),
        # 'migemmk': (flex_gemm_kernel_fn, migemmk_prepare_fn),
    }
    
    reference = (flex_gemm_kernel_fn, migemmk_prepare_fn)
    
    max_flops = get_device_max_flops(torch.float16)
    
    results = {}
    for c in tqdm(test_cases, leave=False):
        RES, C, B, K, S, P, D = c['RES'], c['C'], c['B'], c['ksize'], c['stride'], c['padding'], c['dilation']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, B, dtype=torch.float16)
        weight = torch.randn(C, K[0], K[1], K[2], C, device=feats.device, dtype=feats.dtype)
        bias = torch.randn(C, device=feats.device, dtype=feats.dtype)
        args = {
            'feats': feats,
            'coords': coords,
            'shape': shape,
            'weight': weight,
            'bias': bias,
            'ksize': K,
            'stride': S,
            'padding': P,
            'dilation': D,
        }

        config_key = f'RES={RES},C={C},B={B},K={K[0]},S={S[0]},P={P[0]},D={D[0]}'
        results[config_key] = {
            'time': [],
            'memory': [],
            'err_max': [],
            'err_mean': [],
            'tflops': [],
        }
        
        neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, shape, K, S, P, D, False)
        L = (neighbor_cache['neighbor_map']!=0xffffffff).sum()
        total_flops = 2 * L * C * C
        
        # Benchmark the reference kernel.
        avg_time_ref, memory_ref, C_ref = benchmark_kernel(reference[0], **args, prepare_fn=reference[1])
        C_ref_feats, C_ref_coords = C_ref
        C_ref_idx = lexsort(C_ref_coords.T)
        C_ref = C_ref_feats[C_ref_idx]
        C_ref_coords = C_ref_coords[C_ref_idx]

        # Benchmark each custom kernel.
        for name, (kernel_fn, prepare_fn) in kernel_functions.items():
            if RES > 128 and name == 'dense':
                results[config_key]['time'].append('N/A')
                results[config_key]['memory'].append('N/A')
                results[config_key]['err_max'].append('N/A')
                results[config_key]['err_mean'].append('N/A')
                results[config_key]['tflops'].append('N/A')
                continue
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            C_kernel_feats, C_kernel_coords = C_kernel
            C_kernel_idx = lexsort(C_kernel_coords.T)
            C_kernel = C_kernel_feats[C_kernel_idx]
            C_kernel_coords = C_kernel_coords[C_kernel_idx]
            assert torch.equal(C_ref_coords, C_kernel_coords), f"Coords mismatch for {kernel_fn.__name__}. Got {C_ref_coords} and {C_kernel_coords}."
            results[config_key]['time'].append(f'{avg_time:.2f} ms ({avg_time_ref/avg_time*100:.1f}%)')
            results[config_key]['memory'].append(f'{memory:.1f}G')
            if C_kernel is not None:
                err_max, err_mean = calc_err(C_kernel, C_ref)
                results[config_key]['err_max'].append(f'{err_max * 1000:.0f}‰')
                results[config_key]['err_mean'].append(f'{err_mean * 1000:.0f}‰')
            else:
                results[config_key]['err_max'].append('N/A')
                results[config_key]['err_mean'].append('N/A')
            if max_flops is not None:
                real_flops = total_flops / avg_time * 1e3
                real_tflops = real_flops / 1e12
                results[config_key]['tflops'].append(f'{real_tflops:.2f} ({real_flops/max_flops*100:.1f}%)')
            else:
                results[config_key]['tflops'].append('N/A')
                
    # Print results as a formatted table.
    print("=" * 180)
    print("Sparse Conv Forward Benchmark Results")
    print("=" * 180)
    for m in ['time','memory', 'err_max', 'err_mean', 'tflops']:
        print(m.capitalize())
        print("-" * 180)
        items = [f'{"settings":<36}']
        for f in kernel_functions.keys():
            items.append(f'{f:<20}')
        print(' | '.join(items))
        print("-" * 180)
        for k, v in results.items():
            items = [f'{k:<36}']
            items.extend([f'{x:<20}' for x in v[m]])
            print(' | '.join(items))
        print("-" * 180)
        

if __name__ == "__main__":
    test_conv_fwd()

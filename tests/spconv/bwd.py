import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import torch
import spconv.pytorch as spconv
import flex_gemm
from flex_gemm.ops.spconv import SparseConv3dFunction
from utils import sphere_coords, calc_err, benchmark_kernel, lexsort, get_device_max_flops


def torch_conv3d_prepare_fn(grad_output: torch.Tensor, feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor,
                      ksize, stride, padding, dilation, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    
    # Init module.
    module = torch.nn.Conv3d(Ci, Co, ksize, stride=stride, padding=padding, dilation=dilation, bias=True).cuda().to(feats.dtype)
    module.weight.data.copy_(weight.permute(0, 4, 1, 2, 3).contiguous())
    module.bias.data.zero_()
    
    dense_feats = torch.zeros(shape, device=feats.device, dtype=feats.dtype)
    dense_feats[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = feats
    output = module(dense_feats)
    
    coords = torch.any(output.abs() > 1e-6, dim=1).nonzero(as_tuple=False)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(coords.T)] = grad_output
    output = output[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] + bias
    
    return {
        'input': feats,
        'weight': module.weight,
        'bias': bias,
        'output': output,
        'grad_output': grad_output_,
    }
    

def torch_conv3d_kernel_fn(input, weight, bias, output, grad_output):
    input.grad = None
    weight.grad = None
    bias.grad = None
    output.backward(grad_output, retain_graph=True)
    input_grad = input.grad
    weight_grad = weight.grad.permute(0, 2, 3, 4, 1).contiguous()
    bias_grad = bias.grad
    return input_grad, weight_grad, bias_grad


def spconv_prepare_fn(grad_output: torch.Tensor, feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor,
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
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_spconv.indices.T)] = grad_output
    
    return {
        'input': input_spconv.features,
        'weight': module.weight,
        'bias': module.bias,
        'output': out_spconv.features,
        'grad_output': grad_output_,
    }
    

def spconv_kernel_fn(input, weight, bias, output, grad_output):
    input.grad = None
    weight.grad = None
    bias.grad = None
    output.backward(grad_output, retain_graph=True)
    return input.grad, weight.grad, bias.grad


def egemm_prepare_fn(grad_output: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ksize, stride, padding, dilation)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ksize, stride, padding, dilation, True)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_coords.T)] = grad_output
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        'grad_output': grad_output_,
        **kwargs,
    }
    
    
def igemm_prepare_fn(grad_output: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ksize, stride, padding, dilation)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ksize, stride, padding, dilation, True)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_coords.T)] = grad_output
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        'grad_output': grad_output_,
        **kwargs,
    }
    

def igemmk_prepare_fn(grad_output: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ksize, stride, padding, dilation)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ksize, stride, padding, dilation, True)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_coords.T)] = grad_output
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        'grad_output': grad_output_,
        **kwargs,
    }
    

def migemm_prepare_fn(grad_output: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ksize, stride, padding, dilation)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ksize, stride, padding, dilation, True)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_coords.T)] = grad_output
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        'grad_output': grad_output_,
        **kwargs,
    }
    

def migemmk_prepare_fn(grad_output: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, ksize, stride, padding, dilation, **kwargs):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ksize, stride, padding, dilation)
    neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ksize, stride, padding, dilation, True)
    grad_output_ = torch.empty_like(grad_output)
    grad_output_[lexsort(out_coords.T)] = grad_output
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        'grad_output': grad_output_,
        **kwargs,
    }
    

def flex_gemm_kernel_fn(**kwargs):
    return SparseConv3dFunction._sparse_conv_backward(**kwargs)


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
        test_cases.append({**config, 'ksize': (2, 2, 2), 'stride': (2, 2, 2), 'padding': (0, 0, 0), 'dilation': (1, 1, 1)})
    
    # List of custom kernel functions.
    kernel_functions = {
        'dense': (torch_conv3d_kernel_fn, torch_conv3d_prepare_fn),
        'spconv': (spconv_kernel_fn, spconv_prepare_fn),
        # 'egemm': (flex_gemm_kernel_fn, egemm_prepare_fn),
        'igemm': (flex_gemm_kernel_fn, igemm_prepare_fn),
        'igemmk': (flex_gemm_kernel_fn, igemmk_prepare_fn),
        'migemm': (flex_gemm_kernel_fn, migemm_prepare_fn),
        'migemmk': (flex_gemm_kernel_fn, migemmk_prepare_fn),
    }
    
    reference = (flex_gemm_kernel_fn, egemm_prepare_fn)
    
    max_flops = get_device_max_flops(torch.float16)
    
    results = {}
    for c in tqdm(test_cases, leave=False):
        RES, C, B, K, S, P, D = c['RES'], c['C'], c['B'], c['ksize'], c['stride'], c['padding'], c['dilation']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, B, dtype=torch.float16)
        weight = torch.randn(C, K[0], K[1], K[2], C, device=feats.device, dtype=feats.dtype)
        bias = torch.randn(C, device=feats.device, dtype=feats.dtype)
        feats.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True
        out_coords = SparseConv3dFunction._get_output_coords(coords, shape, K, S, P, D)
        grad_output = torch.randn(out_coords.shape[0], C, device=feats.device, dtype=feats.dtype)
        args = {
            'grad_output': grad_output,
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
        
        # Benchmark the reference kernel.
        avg_time_ref, memory_ref, C_ref = benchmark_kernel(reference[0], **args, prepare_fn=reference[1])

        out_coords = SparseConv3dFunction._get_output_coords(coords, shape, K, S, P, D)
        neighbor_cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, K, S, P, D, False)
        L = (neighbor_cache['neighbor_map']!=0xffffffff).sum()
        total_flops = 4 * L * C * C

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
    print("Sparse Conv Backward Benchmark Results")
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

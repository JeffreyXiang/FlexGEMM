import time
import torch


@torch.no_grad()
def sphere_coords(res, ch, batch_size=1, device='cuda', dtype=torch.float):
    l_coords = []
    for i in range(0, res, 256):
        for j in range(0, res, 256):
            for k in range(0, res, 256):
                coords = torch.stack(torch.meshgrid(
                    torch.arange(i, min(i + 256, res), device=device),
                    torch.arange(j, min(j + 256, res), device=device),
                    torch.arange(k, min(k + 256, res), device=device),
                    indexing='ij'
                ), dim=-1).int().contiguous()
                dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
                active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
                coords = torch.nonzero(active).int() + torch.tensor([i, j, k], device=device, dtype=torch.int32)
                l_coords.append(coords)
    coords = torch.cat(l_coords, dim=0)
    batch_idx = torch.arange(batch_size).repeat_interleave(coords.shape[0]).to(device).int()
    coords = torch.cat([batch_idx.unsqueeze(-1), torch.cat([coords] * batch_size)], dim=-1)
    feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype)
    return feats.contiguous(), coords.contiguous(), torch.Size([batch_size, ch, res, res, res])


def calc_err(src, ref):
    abs_err = (src - ref).float().abs()
    rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
    err = torch.minimum(abs_err, rel_err)
    return err.max().item(), err.mean().item()


def benchmark_kernel(kernel_fn, *args, prepare_fn=None, num_warmup=2, num_iters=20, **kwargs):
    try:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        if prepare_fn is not None:
            kwargs = prepare_fn(*args, **kwargs)
            args = tuple()
        # Warmup iterations.
        for _ in range(num_warmup):
            C = kernel_fn(*args, **kwargs)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # Timing iterations.
        starter.record()
        for _ in range(num_iters):
            C = kernel_fn(*args, **kwargs)
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        memory = torch.cuda.max_memory_allocated() / 1024**3
        avg_time_ms = elapsed / num_iters
        avg_mem_gb = memory
        if isinstance(C, tuple):
            C = torch.cat([c.detach().flatten() for c in C if c is not None], dim=0)
    except Exception as e:
        if isinstance(e, RuntimeError) and 'out of memory' in str(e):
            print('WARNING: OOM error occurred during benchmarking. Skipping this run.')
            return None, 'OOM', None
        else:
            raise e
    return avg_time_ms, avg_mem_gb, C


def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            
            
def lexsort(keys, dim=0):
    """Perform lexicographical sort on multiple keys. Like `numpy.lexsort`. 
    
    Given multiple sorting keys, lexsort returns an array of integer indices that describes the sort order by multiple keys. 
    The last key in the sequence is used for the primary sort order, ties are broken by the second-to-last key, and so on.

    Parameters
    ----
    - `keys`: (Sequence[Tensor]) sequence of Tensors to sort by, or a single Tensor with shape `(num_keys, ...)`.
    - `dim`: (int) the dimension to sort along. Note that if `keys` is a single Tensor, `dim=0` refers to the second dimension of `keys`.

    Returns
    ----
    - `indices`: (Tensor) the indices that would sort the keys lexicographically along the specified dimension.

    Notes
    -----
    Sorting is always stable.
    """
    keys = torch.unbind(keys, dim=0)

    assert len(keys) > 0, "At least one key is required for lexsort"
    
    dim = dim % keys[0].ndim
    for i, key in enumerate(keys):
        if i == 0:
            sorted_indices = torch.argsort(key, dim=dim, stable=True)
        else:
            key = torch.take_along_dim(key, sorted_indices, dim=dim)
            sorted_indices = torch.take_along_dim(sorted_indices, torch.argsort(key, dim=dim, stable=True), dim=dim)
    
    return sorted_indices


def get_device_max_flops(dtype=torch.float):
    TABLE = {
        'A100': {
            torch.float32: 19.5 * 10**12,
            torch.float16: 312 * 10**12,
        },
        'H100': {
            torch.float32: 67 * 10**12,
            torch.float16: 989 * 10**12,
        },
    }
    device_name = torch.cuda.get_device_name()
    if 'A100' in device_name:
        return TABLE['A100'][dtype]
    elif 'H100' in device_name:
        return TABLE['H100'][dtype]
    else:
        return None

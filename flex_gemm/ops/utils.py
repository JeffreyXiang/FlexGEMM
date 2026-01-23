import torch


def init_hashmap(spatial_size, hashmap_size, device, with_values=True):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D
        
    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    if with_values:
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)
        return hashmap_keys, hashmap_vals
    return hashmap_keys

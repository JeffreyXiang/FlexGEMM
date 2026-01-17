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

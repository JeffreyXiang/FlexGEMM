# FlexGEMM

**FlexGEMM** is a **Triton-powered GEMM backend** for **3D sparse convolutions**.
It implements **Explicit**, **Implicit**, **Masked Implicit** algorithm variants, with optional **Split-K** parallelism for sparse GEMM, delivering **state-of-the-art performance** for Submanifold Convolution and voxel-based neural networks.

## Why FlexGEMM?

* **Triton-first**: Built on [Triton](https://github.com/openai/triton) for high performance and cross-platform GPU kernels.
* **Sparse-ready**: Optimized for 3D sparse tensors with highly irregular sparsity.
* **Fast**: Consistently outperforms existing sparse convolution libraries.

## Installation

```bash
git clone https://github.com/JeffreyXiang/FlexGEMM.git
cd FlexGEMM
pip install .
```

Requires:

* PyTorch ≥ 2.5.0
* Triton ≥ 3.2.0

## Example

```python
import torch
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from tests.spconv_fwd import sphere_coords

# Sparse voxel shell
feats, coords, shape = sphere_coords(256, 256, dtype=torch.float16, device='cuda')

# Weight and bias
Ci, Co = 256, 256
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda', requires_grad=True)
bias = torch.randn(Co, dtype=torch.float16, device='cuda', requires_grad=True)

# Set algorithm: Masked + Split-K
flex_gemm.ops.spconv.set_algorithm(
    flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK
)

out_feats, neignbor_cache = sparse_submanifold_conv3d(
    feats, coords, shape,
    weight, bias,
)

out_feats.sum().backward()
```

## Performance 

Environment:

* NVIDIA A100 (80 SMs)
* PyTorch 2.6.0
* CUDA 12.4
* Triton 3.2.0
* Float16 (FP16) precision

### Sparse Conv3D Forward

| RES  | C    | spconv | torchsparse | FlexGEMM   | Speed Up |
| :--: | :--: | :----: | :---------: | :--------: | :------: |
| 8    | 1024 | 0.30ms | 0.95ms      | **0.19ms** |  1.58×   |
| 16   | 1024 | 0.47ms | 0.91ms      | **0.28ms** |  1.68×   |
| 32   | 1024 | 1.38ms | 2.46ms      | **1.03ms** |  1.35×   |
| 64   | 1024 | 5.73ms | 6.99ms      | **3.71ms** |  1.54x   |
| 128  | 512  | 3.62ms | 7.29ms      | **2.96ms** |  1.22x   |
| 256  | 256  | 3.60ms | 7.38ms      | **2.62ms** |  1.37x   |
| 512  | 128  | 4.37ms | 8.27ms      | **3.02ms** |  1.45x   |
| 1024 | 64   | 7.84ms | 9.82ms      | **4.99ms** |  1.57x   |

### Sparse Conv3D Backward

| RES  | C    | spconv | torchsparse | FlexGEMM   | Speed Up |
| :--: | :--: | :----: | :---------: | :--------: | :------: |
| 8    | 1024 | 0.49ms | 7.70ms      | **0.36ms** | 1.36x    |
| 16   | 1024 | 0.88ms | 7.98ms      | **0.45ms** | 1.96x    |
| 32   | 1024 | 2.71ms | 9.67ms      | **1.61ms** | 1.68x    |
| 64   | 1024 | 10.14ms| 17.49ms     | **5.82ms** | 1.74x    |
| 128  | 512  | 8.67ms | 14.26ms     | **5.51ms** | 1.57x    |
| 256  | 256  | 8.96ms | 14.38ms     | **5.34ms** | 1.68x    |
| 512  | 128  | 10.63ms| 21.55ms     | **6.30ms** | 1.69x    |
| 1024 | 64   | 18.89ms| 31.66ms     | **12.15ms**| 1.55x    |


### Summary

* **FlexGEMM consistently outperforms `spconv` and `torchsparse`** across both forward and backward sparse convolution benchmarks.
* **Significant speedups** are observed across both low- and high-resolution sparse tensors, achieving up to **\~2× acceleration** and an average **\~1.6× speedup** compared to `spconv` and `torchsparse`.
* **Memory-efficient**: Achieves higher throughput without increasing GPU memory usage.
* **Robust across channel and resolution scales**: Performs well for both wide (C=1024) and narrow (C=64) feature maps, as well as small (RES=8) and large (RES=1024) voxel grids.
* **Ideal for large-scale 3D networks**: Particularly suitable for high-resolution voxelized point clouds, submanifold convolutions, and octree-based architectures.


## License

MIT.


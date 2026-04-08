# FlexGEMM ROCm 迁移实验

## 实验总览表

| Exp | 假设 | 状态 | 关键结果 | 结论 |
|-----|------|------|----------|------|
| Exp-1 | FlexGEMM 可在 MI300X + ROCm 6.4 上直接编译安装 (setup.py 已有 HIP 支持) | 待实验 | — | — |
| Exp-2 | FlexGEMM 基础测试 (hashmap, serialize, spconv fwd) 在 ROCm 上通过 | 待实验 | — | — |
| Exp-3 | FlexGEMM 集成到 TRELLIS.2 pipeline 后 `import flex_gemm` 正常 | 待实验 | — | — |

---

## 背景

FlexGEMM ([JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)) 是 Triton-powered 稀疏 3D 卷积后端。
TRELLIS.2 setup.sh 安装的旧版基于 CUTLASS 导致 ROCm 编译失败，最新版已迁移到 Triton。

### 代码架构

```
flex_gemm/kernels/
├── triton/     ← 16 个 Triton kernel (GEMM, sparse conv fwd/bwd) — 跨平台
│   ├── spconv/       (8 kernel files: implicit_gemm, masked_implicit_gemm, splitk variants)
│   └── grid_sample/  (2 kernel files: indice_weighed_sum fwd/bwd)
└── cuda/       ← 20 个 C++/CUDA 文件 (hash, serialize, grid_sample, neighbor map) — 需 hipify
    ├── hash/         (hash.cu, hash.cuh — atomicCAS linear probing)
    ├── serialize/    (api.cu — z-order, hilbert encoding)
    ├── grid_sample/  (grid_sample.cu)
    ├── spconv/       (subm_neighbor_map.cu, sparse_neighbor_map.cu, migemm_neighmap_pp.cu)
    └── ext.cpp       (PyBind11 binding)
```

### setup.py 已有 HIP 支持

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION

if IS_HIP_EXTENSION:
    IS_HIP = True
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]
```

### ROCm 风险点

| 风险项 | 位置 | 严重度 | 说明 |
|--------|------|--------|------|
| `__syncwarp()` | migemm_neighmap_pp.cu:280 | 中 | ROCm 6.2+ 需 `HIP_ENABLE_WARP_SYNC_BUILTINS` |
| `warpSize` 差异 (32→64) | migemm_neighmap_pp.cu:263,275,278 | 中 | 代码用 runtime 变量自适应，但逻辑正确性需验证 |
| `#include <cuda.h>` | 6 个 .cu/.h 文件 | 低 | PyTorch CUDAExtension + hipcc 自动处理 |
| `atomicCAS` | hash.cuh | 低 | HIP 原生支持 |
| `__ffs`, `__popc` | migemm_neighmap_pp.cu | 低 | HIP 原生支持 |

---

## Exp-1: ROCm 编译安装

### Phase 0 确认

- 输入完备性: repo URL + setup.py (已含 HIP 路径) + MI300X 节点
- 变量: GPU 平台 (CUDA → ROCm)
- 评估标准: `pip install . --no-build-isolation` 成功 + `import flex_gemm` 成功

### 假设

FlexGEMM 可在 AMD MI300X (gfx942) + ROCm 6.4 Docker 上直接编译安装，无需代码修改。
setup.py 已有 `IS_HIP_EXTENSION` 检测，hipcc 应能编译所有 .cu 文件。

### 实验方案

- **Repo**: https://github.com/ZJLi2013/FlexGEMM (rocm branch)
- **Node**: 待选 (MI300X, gfx942)
- **Docker**: `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- **步骤**:
  1. Clone repo (rocm branch)
  2. `GPU_ARCHS=gfx942 pip install . --no-build-isolation`
  3. `python -c "import flex_gemm; print('OK')"`
  4. 如果编译失败，分析具体错误

### 预期

- **假设成立**: 编译 + import 一次通过
- **假设不成立**: hipcc 对 `__syncwarp` 或 `warpSize` 相关代码报错，需 patch

### 结果

（待实验）

### 分析

（待实验）

### 结论与 Next Step

（待实验）

---

## Exp-2: 基础功能测试

### 假设

编译安装后，FlexGEMM 的 hashmap、serialize、spconv forward/backward 测试在 MI300X 上功能正确。

### 实验方案

```bash
# hashmap test
python tests/hashmap.py

# serialize test
python tests/serialize.py

# spconv forward test
python tests/spconv/fwd.py

# spconv backward test
python tests/spconv/bwd.py

# submanifold conv tests
python tests/submconv/fwd.py
python tests/submconv/bwd.py
```

### 预期

- **假设成立**: 全部测试 PASS，数值结果与 CUDA 参考一致
- **假设部分成立**: Triton kernel 部分 PASS, CUDA utility (warpSize=64 相关) 部分 FAIL
- **假设不成立**: 大面积失败

### 结果

（待实验）

---

## Exp-3: TRELLIS.2 集成测试

### 假设

在 TRELLIS.2 环境中安装最新 FlexGEMM 后，pipeline 可成功加载 flex_gemm + o_voxel。

### 实验方案

- 在 TRELLIS.2 Docker 中安装 FlexGEMM (rocm branch)
- 测试 `import flex_gemm; import o_voxel`
- 测试 TRELLIS.2 pipeline load

### 结果

（待实验）

---

## 调试追踪

| 轮次 | 问题 | 修复 | 结果 |
|------|------|------|------|
| — | — | — | — |

---

## 参考

- Fork: https://github.com/ZJLi2013/FlexGEMM
- 上游: https://github.com/JeffreyXiang/FlexGEMM
- TRELLIS.2 实验: ../overnight_tasks/TRELLIS.2/experiments.md
- libs_transfer.md: ../overnight_tasks/libs_transfer.md
- ROCm warp sync: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html

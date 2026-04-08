#!/bin/bash
set -e

echo "=== ENVIRONMENT ==="
python3 -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.version.hip)"
python3 -c "import importlib.metadata; print('Triton version:', importlib.metadata.version('triton'))" 2>&1

echo ""
echo "=== CLONE ==="
cd /data
rm -rf flexgemm-test
mkdir -p flexgemm-test
cd flexgemm-test
git clone --branch rocm https://github.com/ZJLi2013/FlexGEMM.git 2>&1 | tail -3

echo ""
echo "=== COMPILE ==="
cd FlexGEMM
export GPU_ARCHS=gfx942
export PYTORCH_ROCM_ARCH=gfx942
pip install . --no-build-isolation 2>&1

echo ""
echo "=== IMPORT TEST ==="
cd /tmp
python3 -c "
import flex_gemm
print('flex_gemm imported OK')
print('Module:', flex_gemm)

from flex_gemm.kernels import cuda as _C
print('CUDA/HIP extension loaded OK')
print('Available functions:', [x for x in dir(_C) if not x.startswith('_')])
"

echo ""
echo "=== BASIC FUNCTION TEST ==="
python3 -c "
import torch
import flex_gemm

print('Testing hashmap_insert...')
N = 1024
keys = torch.randint(0, 10000, (100,), dtype=torch.uint32, device='cuda')
values = torch.arange(100, dtype=torch.uint32, device='cuda')
hmap_keys = torch.full((N,), 2**32-1, dtype=torch.uint32, device='cuda')
hmap_values = torch.zeros(N, dtype=torch.uint32, device='cuda')
flex_gemm.kernels.cuda.hashmap_insert(hmap_keys, hmap_values, keys, values)
print('hashmap_insert: OK')

print('Testing hashmap_lookup...')
result = flex_gemm.kernels.cuda.hashmap_lookup(hmap_keys, hmap_values, keys)
print('hashmap_lookup: OK, result shape:', result.shape)

print('')
print('=== ALL BASIC TESTS PASSED ===')
"

echo ""
echo "=== DONE ==="

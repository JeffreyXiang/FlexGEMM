#!/bin/bash
set -e

echo "=== ENVIRONMENT ==="
python3 -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.version.hip, 'GPU:', torch.cuda.get_device_name(0))"
python3 -c "import importlib.metadata; print('Triton version:', importlib.metadata.version('triton'))"

echo ""
echo "=== CLONE + INSTALL ==="
cd /data
rm -rf flexgemm-test
mkdir -p flexgemm-test && cd flexgemm-test
git clone --branch rocm https://github.com/ZJLi2013/FlexGEMM.git 2>&1 | tail -3
cd FlexGEMM
pip install tqdm 2>&1 | tail -3
export GPU_ARCHS=gfx942
export PYTORCH_ROCM_ARCH=gfx942
pip install . --no-build-isolation 2>&1 | tail -5

echo ""
echo "=== TEST 1: HASHMAP ==="
cd /data/flexgemm-test/FlexGEMM/tests
python3 -c "
import torch
from flex_gemm import kernels

print('Test 1a: hashmap_insert_3d + lookup_3d')
res = 64
N = 100
coords = torch.randint(0, res, (N, 4), dtype=torch.int32, device='cuda')
coords[:, 0] = 0  # batch=0

hashmap_keys = torch.full((2*N,), (1<<32)-1, dtype=torch.uint32, device='cuda')
hashmap_values = torch.empty(2*N, dtype=torch.uint32, device='cuda')
values = torch.randint(0, 1000, (N,), device='cuda').to(torch.uint32)

kernels.cuda.hashmap_insert_3d(hashmap_keys, hashmap_values, coords, values, res, res, res)
result = kernels.cuda.hashmap_lookup_3d(hashmap_keys, hashmap_values, coords, res, res, res)

match = (result == values).all().item()
print(f'  insert+lookup match: {match}')
assert match, 'Hashmap insert/lookup mismatch!'

print('Test 1b: hashmap_insert_3d_idx_as_val')
hashmap_keys2 = torch.full((2*N,), (1<<32)-1, dtype=torch.uint32, device='cuda')
hashmap_values2 = torch.empty(2*N, dtype=torch.uint32, device='cuda')
kernels.cuda.hashmap_insert_3d_idx_as_val(hashmap_keys2, hashmap_values2, coords, res, res, res)
result2 = kernels.cuda.hashmap_lookup_3d(hashmap_keys2, hashmap_values2, coords, res, res, res)
r2_int = result2.to(torch.int32)
print(f'  idx_as_val result range: [{r2_int.min().item()}, {r2_int.max().item()}]')
print('  PASS')

print()
print('=== HASHMAP TESTS PASSED ===')
"

echo ""
echo "=== TEST 2: SERIALIZE ==="
python3 -c "
import torch
from flex_gemm import kernels
from flex_gemm.ops.serialize import encode_seq, decode_seq

print('Test 2a: z_order encode/decode (via ops API)')
N = 100
shape = torch.Size([1, 64, 64, 64, 64])
coords = torch.randint(0, 64, (N, 4), dtype=torch.int32, device='cuda')
coords[:, 0] = 0

codes = encode_seq(coords, shape, mode='z_order')
decoded = decode_seq(codes, shape, mode='z_order')
match = (decoded == coords).all().item()
print(f'  z_order roundtrip match: {match}')
assert match, 'Z-order roundtrip mismatch!'

print('Test 2b: hilbert encode/decode')
codes_h = encode_seq(coords, shape, mode='hilbert')
decoded_h = decode_seq(codes_h, shape, mode='hilbert')
match_h = (decoded_h == coords).all().item()
print(f'  hilbert roundtrip match: {match_h}')
assert match_h, 'Hilbert roundtrip mismatch!'

print()
print('=== SERIALIZE TESTS PASSED ===')
"

echo ""
echo "=== TEST 3: SPARSE SUBMANIFOLD CONV (Triton kernel) ==="
python3 -c "
import sys
sys.path.insert(0, '/data/flexgemm-test/FlexGEMM/tests')
import torch
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from utils import sphere_coords

print('Test 3: Sparse submanifold conv3d forward')
feats, coords, shape = sphere_coords(64, 64, dtype=torch.float16, device='cuda')
print(f'  Input: {feats.shape} feats, {coords.shape} coords, shape={shape}')

Ci, Co = 64, 64
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda')
bias = torch.randn(Co, dtype=torch.float16, device='cuda')

flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
out_feats, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
print(f'  Output: {out_feats.shape}, dtype={out_feats.dtype}')
print(f'  Output stats: mean={out_feats.mean().item():.4f}, std={out_feats.std().item():.4f}')
assert out_feats.shape[0] == feats.shape[0] and out_feats.shape[1] == Co
print('  Forward PASS')

print()
print('Test 3b: Sparse submanifold conv3d backward')
loss = out_feats.sum()
weight.requires_grad_(True)
bias.requires_grad_(True)
feats2 = feats.clone().requires_grad_(True)
out2, cache2 = sparse_submanifold_conv3d(feats2, coords, shape, weight, bias)
out2.sum().backward()
print(f'  feats grad shape: {feats2.grad.shape}')
print(f'  weight grad shape: {weight.grad.shape}')
print(f'  bias grad shape: {bias.grad.shape}')
print('  Backward PASS')

print()
print('=== SPARSE CONV TESTS PASSED ===')
"

echo ""
echo "=== TEST 4: MASKED IMPLICIT GEMM ==="
python3 -c "
import sys
sys.path.insert(0, '/data/flexgemm-test/FlexGEMM/tests')
import torch
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from utils import sphere_coords

print('Test 4: Masked Implicit GEMM')
feats, coords, shape = sphere_coords(64, 64, dtype=torch.float16, device='cuda')
Ci, Co = 64, 64
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda', requires_grad=True)
bias = torch.randn(Co, dtype=torch.float16, device='cuda', requires_grad=True)

flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
feats_g = feats.clone().requires_grad_(True)
out, cache = sparse_submanifold_conv3d(feats_g, coords, shape, weight, bias)
out.sum().backward()
print(f'  Output shape: {out.shape}')
print(f'  Grad shapes: feats={feats_g.grad.shape}, weight={weight.grad.shape}')
print('  Masked Implicit GEMM PASS')

print()
flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
feats_g2 = feats.clone().requires_grad_(True)
weight2 = weight.data.clone().requires_grad_(True)
bias2 = bias.data.clone().requires_grad_(True)
out2, cache2 = sparse_submanifold_conv3d(feats_g2, coords, shape, weight2, bias2)
out2.sum().backward()
print(f'  SplitK Output shape: {out2.shape}')
print('  Masked Implicit GEMM SplitK PASS')

print()
print('=== ALL ALGORITHM VARIANTS PASSED ===')
"

echo ""
echo "=========================================="
echo "=== ALL FLEXGEMM ROCM TESTS COMPLETED ==="
echo "=========================================="

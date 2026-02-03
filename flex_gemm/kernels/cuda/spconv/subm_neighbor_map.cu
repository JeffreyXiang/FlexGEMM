#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "subm_neighbor_map.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"


namespace flex_gemm {
namespace spconv {

/**
 * Lookup sparse submanifold convolution neighbor map with hashmap
 * 
 * @param N             number of elements in the hashmap
 * @param M             number of 3d coordinates
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param V             the volume of the kernel
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param neighbor      [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution nerbor map
 */
template<typename T>
__global__ void hashmap_lookup_submanifold_conv_neighbour_map_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const int V,
    const int Kw,
    const int Kh,
    const int Kd,
    const int Dw,
    const int Dh,
    const int Dd,
    const T* __restrict__  hashmap_keys,
    const uint32_t* __restrict__  hashmap_vals,
    const int32_t* __restrict__  coords,
    uint32_t* __restrict__ neighbor
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int half_V = V / 2 + 1;
    uint32_t idx = static_cast<uint32_t>(thread_id / half_V);
    if (idx < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[idx];
        int b = coord.x;
        int x = coord.y - Kw / 2 * Dw;
        int y = coord.z - Kh / 2 * Dh;
        int z = coord.w - Kd / 2 * Dd;
        int KhKd = Kh * Kd;
        int v = thread_id % half_V;
        
        uint32_t value = std::numeric_limits<uint32_t>::max();
        if (v == half_V - 1) {
            value = idx;
        }
        else {
            int kx = x + v / KhKd * Dw;
            int ky = y + v / Kd % Kh * Dh;
            int kz = z + v % Kd * Dd;
            if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
                size_t flat_idx = (size_t)b * W * H * D + (size_t)kx * H * D + (size_t)ky * D + (size_t)kz;
                T key = static_cast<T>(flat_idx);
                value = flex_gemm::hash::linear_probing_lookup(hashmap_keys, hashmap_vals, key, N);
                if (value != std::numeric_limits<uint32_t>::max()) {
                    neighbor[value * V + V - 1 - v] = idx;
                }
            }
        }
        neighbor[idx * V + v] = value;
    }
}


/**
 * Build sparse submanifold convolution neighbor map with hashmap
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 *  
 * @return              [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution neighbor map
 */
torch::Tensor hashmap_build_submanifold_conv_neighbour_map(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd
) {
    // Allocate output tensor
    int V = Kw * Kh * Kd;

    // Insert 3D coordinates into the hashmap
    flex_gemm::hash::hashmap_insert_3d_idx_as_val(
        hashmap_keys,
        hashmap_vals,
        coords,
        W, H, D
    );

    auto neighbor = torch::full({coords.size(0), V}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(hashmap_keys.device()));

    if (hashmap_keys.dtype() == torch::kUInt32) {
        hashmap_lookup_submanifold_conv_neighbour_map_kernel<<<
            (coords.size(0) * (V / 2 + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        hashmap_lookup_submanifold_conv_neighbour_map_kernel<<<
            (coords.size(0) * (V / 2 + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported hashmap dtype. Expect uint32 or uint64.");
    }

    return neighbor;
}

} // namespace spconv
} // namespace flex_gemm

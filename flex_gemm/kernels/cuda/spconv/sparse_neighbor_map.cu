#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sparse_neighbor_map.h"
#include "utils.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"


namespace flex_gemm {
namespace spconv {

__forceinline__ __host__ int bit_length(int n) {
    return (n > 0) ? 1 + std::floor(std::log2(n)) : 0;
}

template<typename T, typename SerializeFunc>
__global__ void build_sparse_conv_out_coords_hashmap_insert_kernel(
    const size_t N,
    const size_t M,
    const int Wo, const int Ho, const int Do,
    const int V, const int Kw, const int Kh, const int Kd,
    const int Sw, const int Sh, const int Sd,
    const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const int32_t* __restrict__  coords,
    T* __restrict__  hashmap_keys,
    SerializeFunc serialize_func
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx = thread_id / V;
    if (idx < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[idx];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int v = thread_id % V;
        int kx = v / (Kh * Kd);
        int ky = v / Kd % Kh;
        int kz = v % Kd;
        int out_x = (x + Pw - kx * Dw);
        int out_y = (y + Ph - ky * Dh);
        int out_z = (z + Pd - kz * Dd);
        if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
            out_x /= Sw;
            out_y /= Sh;
            out_z /= Sd;
            if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho && out_z >= 0 && out_z < Do) {
                T key = serialize_func.encode(b, out_x, out_y, out_z);
                flex_gemm::hash::linear_probing_insert(hashmap_keys, key, N);
            }
        }
    }
}


template<typename T, typename SerializeFunc>
__global__ void build_sparse_conv_out_coords_decode_key_kernel(
    const size_t N,
    const int Wo, const int Ho, const int Do,
    const T* __restrict__ valid_keys,
    int32_t* __restrict__ out_coords,
    SerializeFunc serialize_func
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
        T key = valid_keys[thread_id];
        *reinterpret_cast<int4*>(out_coords + thread_id * 4) = serialize_func.decode(key);
    }
}


torch::Tensor hashmap_build_sparse_conv_out_coords(
    const torch::Tensor& in_coords,
    const float hashmap_ratio,
    const int serialize_mode,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
) {
    // Calculate output size
    int Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1;
    int Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1;
    int Do = (D + 2 * Pd - Dd * (Kd - 1) - 1) / Sd + 1;
    int V = Kw * Kh * Kd;
    size_t hashmap_size = static_cast<size_t>(
        hashmap_ratio * Kw * Kh * Kd / Sw / Sh / Sd * in_coords.size(0)
    );

    // Implementation
    auto impl = [&](auto type_tag, auto serialize_func) {
        using T = decltype(type_tag);
        constexpr auto udtype = sizeof(T) == 4 ? torch::kUInt32 : torch::kUInt64;
        constexpr auto dtype = sizeof(T) == 4 ? torch::kInt32 : torch::kInt64;

        torch::Tensor hashmap_keys = torch::full({static_cast<int64_t>(hashmap_size)}, std::numeric_limits<T>::max(), torch::dtype(udtype).device(in_coords.device()));

        build_sparse_conv_out_coords_hashmap_insert_kernel<<<
            (in_coords.size(0) * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            in_coords.size(0),
            Wo, Ho, Do,
            V, Kw, Kh, Kd,
            Sw, Sh, Sd,
            Pw, Ph, Pd,
            Dw, Dh, Dd,
            in_coords.data_ptr<int32_t>(),
            hashmap_keys.data_ptr<T>(),
            serialize_func
        );

        torch::Tensor valid_key = hashmap_keys.view(dtype).masked_select(hashmap_keys != std::numeric_limits<T>::max());
        valid_key = std::get<0>(valid_key.sort()).view(udtype);
        torch::Tensor out_coords = torch::empty({valid_key.size(0), 4}, torch::dtype(torch::kInt32).device(hashmap_keys.device()));

        build_sparse_conv_out_coords_decode_key_kernel<<<
            (valid_key.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            valid_key.size(0),
            Wo, Ho, Do,
            valid_key.data_ptr<T>(),
            out_coords.data_ptr<int32_t>(),
            serialize_func
        );

        return out_coords;
    };

    if (serialize_mode == 0) {  // bxyz
        uint64_t VOL;
        bool safe = true;
        safe &= is_safe_mul(static_cast<uint64_t>(B), static_cast<uint64_t>(Wo), VOL);
        safe &= is_safe_mul(VOL, static_cast<uint64_t>(Ho), VOL);
        safe &= is_safe_mul(VOL, static_cast<uint64_t>(Do), VOL);
        if (!safe) {
            TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
        }

        if (VOL < std::numeric_limits<uint32_t>::max()) {
            return impl(uint32_t(), BxyzSerializeFunc<uint32_t>(Wo, Ho, Do));
        }
        else if (VOL < std::numeric_limits<uint64_t>::max()) {
            return impl(uint64_t(), BxyzSerializeFunc<uint64_t>(Wo, Ho, Do));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
        }
    }
    else if (serialize_mode == 1) {  // zorder
        int max_spatial = std::max(std::max(W, H), D);
        int bit_length_spatial = bit_length(max_spatial - 1);
        int bit_length_batch = bit_length(B - 1);
        int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

        if (bit_length_total < 32) {
            return impl(uint32_t(), ZorderSerializeFunc<uint32_t>(bit_length_spatial));
        }
        else if (bit_length_total < 64) {
            return impl(uint64_t(), ZorderSerializeFunc<uint64_t>(bit_length_spatial));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require total bit length < 64.");
        }
    }
    else if (serialize_mode == 2) {  // hilbert
        int max_spatial = std::max(std::max(W, H), D);
        int bit_length_spatial = bit_length(max_spatial - 1);
        int bit_length_batch = bit_length(B - 1);
        int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

        if (bit_length_total < 32) {
            return impl(uint32_t(), HilbertSerializeFunc<uint32_t>(bit_length_spatial));
        }
        else if (bit_length_total < 64) {
            return impl(uint64_t(), HilbertSerializeFunc<uint64_t>(bit_length_spatial));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require total bit length < 64.");
        }
    }
    else {
        TORCH_CHECK(false, "Unsupported serialize mode.");
    }
}


__global__ void build_sparse_conv_out_coords_get_expanded_size_kernel(
    const size_t N,
    const int Wo, const int Ho, const int Do,
    const int Kw, const int Kh, const int Kd,
    const int Sw, const int Sh, const int Sd,
    const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const int32_t* __restrict__  coords,
    int64_t* __restrict__ expanded_size
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int cnt = 0;
        for (int kx = 0; kx < Kw; kx++) {
            for (int ky = 0; ky < Kh; ky++) {
                for (int kz = 0; kz < Kd; kz++) {
                    int out_x = (x + Pw - kx * Dw);
                    int out_y = (y + Ph - ky * Dh);
                    int out_z = (z + Pd - kz * Dd);
                    if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
                        out_x /= Sw;
                        out_y /= Sh;
                        out_z /= Sd;
                        if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho && out_z >= 0 && out_z < Do) {
                            cnt++;
                        }
                    }
                }
            }
        }
        expanded_size[thread_id] = static_cast<int64_t>(cnt);
    }
}


template<typename T, typename SerializeFunc>
__global__ void build_sparse_conv_out_coords_expand_kernel(
    const size_t N,
    const int Wo, const int Ho, const int Do,
    const int Kw, const int Kh, const int Kd,
    const int Sw, const int Sh, const int Sd,
    const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const int32_t* __restrict__  coords,
    const int64_t* __restrict__ expanded_start,
    T* __restrict__ expanded_keys,
    SerializeFunc serialize_func
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int64_t ptr = thread_id == 0 ? 0 : expanded_start[thread_id - 1];
        for (int kx = 0; kx < Kw; kx++) {
            for (int ky = 0; ky < Kh; ky++) {
                for (int kz = 0; kz < Kd; kz++) {
                    int out_x = (x + Pw - kx * Dw);
                    int out_y = (y + Ph - ky * Dh);
                    int out_z = (z + Pd - kz * Dd);
                    if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
                        out_x /= Sw;
                        out_y /= Sh;
                        out_z /= Sd;
                        if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho && out_z >= 0 && out_z < Do) {
                            expanded_keys[ptr] = serialize_func.encode(b, out_x, out_y, out_z);
                            ptr++;
                        }
                    }
                }
            }
        }
    }
}


torch::Tensor expand_unique_build_sparse_conv_out_coords(
    const torch::Tensor& in_coords,
    const int serialize_mode,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
) {
    // Calculate output size
    int Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1;
    int Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1;
    int Do = (D + 2 * Pd - Dd * (Kd - 1) - 1) / Sd + 1;

    // Implementation
    auto impl = [&](auto type_tag, auto serialize_func) {
        using T = decltype(type_tag);
        constexpr auto udtype = sizeof(T) == 4 ? torch::kUInt32 : torch::kUInt64;

        auto expanded_size = torch::empty({static_cast<int64_t>(in_coords.size(0))}, torch::dtype(torch::kInt64).device(in_coords.device()));

        build_sparse_conv_out_coords_get_expanded_size_kernel<<<
            (in_coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            in_coords.size(0),
            Wo, Ho, Do,
            Kw, Kh, Kd,
            Sw, Sh, Sd,
            Pw, Ph, Pd,
            Dw, Dh, Dd,
            in_coords.data_ptr<int32_t>(),
            expanded_size.data_ptr<int64_t>()
        );

        auto expanded_start = expanded_size.cumsum(0);
        auto expanded_keys = torch::empty({expanded_start[-1].item<int64_t>()}, torch::dtype(udtype).device(in_coords.device()));

        build_sparse_conv_out_coords_expand_kernel<<<
            (in_coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            in_coords.size(0),
            Wo, Ho, Do,
            Kw, Kh, Kd,
            Sw, Sh, Sd,
            Pw, Ph, Pd,
            Dw, Dh, Dd,
            in_coords.data_ptr<int32_t>(),
            expanded_start.data_ptr<int64_t>(),
            expanded_keys.data_ptr<T>(),
            serialize_func
        );

        auto unique_results = at::_unique(expanded_keys);
        auto valid_keys = std::get<0>(unique_results);
        auto out_coords = torch::empty({valid_keys.size(0), 4}, torch::dtype(torch::kInt32).device(in_coords.device()));

        build_sparse_conv_out_coords_decode_key_kernel<<<
            (valid_keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            valid_keys.size(0),
            Wo, Ho, Do,
            valid_keys.data_ptr<T>(),
            out_coords.data_ptr<int32_t>(),
            serialize_func
        );
        return out_coords;
    };

    if (serialize_mode == 0) {  // bxyz
        uint64_t VOL;
        bool safe = true;
        safe &= is_safe_mul(static_cast<uint64_t>(B), static_cast<uint64_t>(Wo), VOL);
        safe &= is_safe_mul(VOL, static_cast<uint64_t>(Ho), VOL);
        safe &= is_safe_mul(VOL, static_cast<uint64_t>(Do), VOL);
        if (!safe) {
            TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
        }

        if (VOL < std::numeric_limits<uint32_t>::max()) {
            return impl(uint32_t(), BxyzSerializeFunc<uint32_t>(Wo, Ho, Do));
        }
        else if (VOL < std::numeric_limits<uint64_t>::max()) {
            return impl(uint64_t(), BxyzSerializeFunc<uint64_t>(Wo, Ho, Do));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
        }
    }
    else if (serialize_mode == 1) {  // zorder
        int max_spatial = std::max(std::max(W, H), D);
        int bit_length_spatial = bit_length(max_spatial - 1);
        int bit_length_batch = bit_length(B - 1);
        int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

        if (bit_length_total < 32) {
            return impl(uint32_t(), ZorderSerializeFunc<uint32_t>(bit_length_spatial));
        }
        else if (bit_length_total < 64) {
            return impl(uint64_t(), ZorderSerializeFunc<uint64_t>(bit_length_spatial));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require total bit length < 64.");
        }
    }
    else if (serialize_mode == 2) {  // hilbert
        int max_spatial = std::max(std::max(W, H), D);
        int bit_length_spatial = bit_length(max_spatial - 1);
        int bit_length_batch = bit_length(B - 1);
        int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

        if (bit_length_total < 32) {
            return impl(uint32_t(), HilbertSerializeFunc<uint32_t>(bit_length_spatial));
        }
        else if (bit_length_total < 64) {
            return impl(uint64_t(), HilbertSerializeFunc<uint64_t>(bit_length_spatial));
        }
        else {
            TORCH_CHECK(false, "The spatial size is too large. Require total bit length < 64.");
        }
    }
    else {
        TORCH_CHECK(false, "Unsupported serialize mode.");
    }
}


/**
 * Lookup sparse convolution neighbor map with hashmap
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
 * @param Sw            the stride of width
 * @param Sh            the stride of height
 * @param Sd            the stride of depth
 * @param Pw            the padding of width
 * @param Ph            the padding of height
 * @param Pd            the padding of depth
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [L, 4] int32 tensor containing the keys to be looked up
 * @param neighbor      [L, Kw * Kh * Kd] uint32 tensor containing the sparse convolution nerbor map
 * @param neighbor_bwd  [M, Kw * Kh * Kd] optional uint32 tensor containing the sparse convolution nerbor map for backward pass
 */
template<typename T>
__global__ void hashmap_lookup_sparse_conv_neighbour_map_kernel(
    const size_t N,
    const size_t L,
    const int W, const int H, const int D,
    const int V, const int Kw, const int Kh, const int Kd,
    const int Sw, const int Sh, const int Sd,
    const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const T* __restrict__  hashmap_keys,
    const uint32_t* __restrict__  hashmap_vals,
    const int32_t* __restrict__  coords,
    uint32_t* __restrict__ neighbor,
    uint32_t* __restrict__ neighbor_bwd
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = static_cast<uint32_t>(thread_id / V);
    if (idx < L) {
        int4 coord = reinterpret_cast<const int4*>(coords)[idx];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int v = thread_id % V;
        uint32_t value = std::numeric_limits<uint32_t>::max();
        int kx = x * Sw - Pw + v / (Kh * Kd) * Dw;
        int ky = y * Sh - Ph + v / Kd % Kh * Dh;
        int kz = z * Sd - Pd + v % Kd * Dd;
        if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
            size_t flat_idx = (size_t)b * W * H * D + (size_t)kx * H * D + (size_t)ky * D + (size_t)kz;
            T key = static_cast<T>(flat_idx);
            value = flex_gemm::hash::linear_probing_lookup(hashmap_keys, hashmap_vals, key, N);
        }
        neighbor[idx * V + v] = value;
        if (neighbor_bwd && value != std::numeric_limits<uint32_t>::max()) {
            neighbor_bwd[value * V + v] = idx;
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> hashmap_build_sparse_conv_neighbour_map(
    const torch::Tensor& in_coords,
    const torch::Tensor& out_coords,
    const float hashmap_ratio,
    const bool include_bwd,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
) {
    uint64_t VOL;
    bool safe = true;
    safe &= is_safe_mul(static_cast<uint64_t>(B), static_cast<uint64_t>(W), VOL);
    safe &= is_safe_mul(VOL, static_cast<uint64_t>(H), VOL);
    safe &= is_safe_mul(VOL, static_cast<uint64_t>(D), VOL);
    if (!safe) {
        TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
    }

    // Allocate output tensor
    int V = Kw * Kh * Kd;
    auto neighbor = torch::full({out_coords.size(0), V}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(in_coords.device()));
    auto neighbor_bwd = include_bwd ?
        torch::full({in_coords.size(0), V}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(in_coords.device())) :
        torch::Tensor();

    // Build hashmap
    size_t hashmap_size = static_cast<size_t>(hashmap_ratio * in_coords.size(0));

    if (VOL < std::numeric_limits<uint32_t>::max()) {
        auto hashmap_keys = torch::full({static_cast<int64_t>(hashmap_size)}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(in_coords.device()));
        auto hashmap_vals = torch::empty({static_cast<int64_t>(hashmap_size)}, torch::dtype(torch::kUInt32).device(in_coords.device()));
    
        // Insert 3D coordinates into the hashmap
        flex_gemm::hash::hashmap_insert_3d_idx_as_val(
            hashmap_keys,
            hashmap_vals,
            in_coords,
            W, H, D
        );

        hashmap_lookup_sparse_conv_neighbour_map_kernel<<<
            (out_coords.size(0) * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            out_coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Sw, Sh, Sd,
            Pw, Ph, Pd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            out_coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>(),
            include_bwd ? neighbor_bwd.data_ptr<uint32_t>() : nullptr
        );
    } else if (VOL < std::numeric_limits<uint64_t>::max()) {
        auto hashmap_keys = torch::full({static_cast<int64_t>(hashmap_size)}, std::numeric_limits<uint64_t>::max(), torch::dtype(torch::kUInt64).device(in_coords.device()));
        auto hashmap_vals = torch::empty({static_cast<int64_t>(hashmap_size)}, torch::dtype(torch::kUInt32).device(in_coords.device()));
    
        // Insert 3D coordinates into the hashmap
        flex_gemm::hash::hashmap_insert_3d_idx_as_val(
            hashmap_keys,
            hashmap_vals,
            in_coords,
            W, H, D
        );

        hashmap_lookup_sparse_conv_neighbour_map_kernel<<<
            (out_coords.size(0) * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            out_coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Sw, Sh, Sd,
            Pw, Ph, Pd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            out_coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>(),
            include_bwd ? neighbor_bwd.data_ptr<uint32_t>() : nullptr
        );
    }
    else {
        TORCH_CHECK(false, "The spatial size is too large. Require B*W*H*D < 2^64.");
    }

    return std::make_tuple(neighbor, neighbor_bwd);
}

} // namespace spconv
} // namespace flex_gemm

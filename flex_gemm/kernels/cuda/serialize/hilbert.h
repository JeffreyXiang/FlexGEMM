#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


namespace flex_gemm {
namespace serialize {

template <typename T> struct HilbertType;
template <> struct HilbertType<uint32_t> { static constexpr int BITS = 10; };
template <> struct HilbertType<uint64_t> { static constexpr int BITS = 21; };

namespace cuda {

/**
 * Hilbert encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
template<typename T>
__global__ void hilbert_encode(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    T* codes
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;

    uint32_t point[3] = {x[thread_id], y[thread_id], z[thread_id]};

    uint32_t m = 1 << (HilbertType<T>::BITS - 1), q, p, t;

    // Inverse undo excess work
    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p;  // invert
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q >>= 1;
    }

    // Gray encode
    for (int i = 1; i < 3; i++) {
        point[i] ^= point[i - 1];
    }
    t = 0;
    q = m;
    while (q > 1) {
        if (point[2] & q) {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for (int i = 0; i < 3; i++) {
        point[i] ^= t;
    }

    // Convert to 3D Hilbert code
    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[0]));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[1]));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[2]));

    codes[thread_id] = xx * 4 + yy * 2 + zz;
}


/**
 * Hilbert decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 */
template<typename T>
__global__ void hilbert_decode(
    size_t N,
    const T* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;

    uint32_t point[3];
    point[0] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 2));
    point[1] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 1));
    point[2] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id]));

    uint32_t m = 2 << (HilbertType<T>::BITS - 1), q, p, t;

    // Gray decode by H ^ (H/2)
    t = point[2] >> 1;
    for (int i = 2; i > 0; i--) {
        point[i] ^= point[i - 1];
    }
    point[0] ^= t;

    // Undo excess work
    q = 2;
    while (q != m) {
        p = q - 1;
        for (int i = 2; i >= 0; i--) {
            if (point[i] & q) {
                point[0] ^= p;
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q <<= 1;
    }

    x[thread_id] = point[0];
    y[thread_id] = point[1];
    z[thread_id] = point[2];
}

} // namespace cuda

namespace cpu {

/**
 * Hilbert encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
template<typename T>
__host__ void hilbert_encode(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    T* codes
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        uint32_t point[3] = {x[thread_id], y[thread_id], z[thread_id]};

        uint32_t m = 1 << (HilbertType<T>::BITS - 1), q, p, t;

        // Inverse undo excess work
        q = m;
        while (q > 1) {
            p = q - 1;
            for (int i = 0; i < 3; i++) {
                if (point[i] & q) {
                    point[0] ^= p;  // invert
                } else {
                    t = (point[0] ^ point[i]) & p;
                    point[0] ^= t;
                    point[i] ^= t;
                }
            }
            q >>= 1;
        }

        // Gray encode
        for (int i = 1; i < 3; i++) {
            point[i] ^= point[i - 1];
        }
        t = 0;
        q = m;
        while (q > 1) {
            if (point[2] & q) {
                t ^= q - 1;
            }
            q >>= 1;
        }
        for (int i = 0; i < 3; i++) {
            point[i] ^= t;
        }

        // Convert to 3D Hilbert code
        T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[0]));
        T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[1]));
        T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[2]));

        codes[thread_id] = xx * 4 + yy * 2 + zz;
    }
}


/**
 * Hilbert decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 */
template<typename T>
__host__ void hilbert_decode(
    size_t N,
    const T* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        uint32_t point[3];
        point[0] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 2));
        point[1] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 1));
        point[2] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id]));

        uint32_t m = 2 << (HilbertType<T>::BITS - 1), q, p, t;

        // Gray decode by H ^ (H/2)
        t = point[2] >> 1;
        for (int i = 2; i > 0; i--) {
            point[i] ^= point[i - 1];
        }
        point[0] ^= t;

        // Undo excess work
        q = 2;
        while (q != m) {
            p = q - 1;
            for (int i = 2; i >= 0; i--) {
                if (point[i] & q) {
                    point[0] ^= p;
                } else {
                    t = (point[0] ^ point[i]) & p;
                    point[0] ^= t;
                    point[i] ^= t;
                }
            }
            q <<= 1;
        }

        x[thread_id] = point[0];
        y[thread_id] = point[1];
        z[thread_id] = point[2];
    }
}

} // namespace cpu
} // namespace serialize
} // namespace flex_gemm

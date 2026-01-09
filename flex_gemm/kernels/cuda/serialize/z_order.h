#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


namespace flex_gemm {
namespace serialize {
namespace cuda {

/**
 * Z-order encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
template<typename T>
__global__ void z_order_encode(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    T* codes
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id >= N) return;

    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(x[thread_id]));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(y[thread_id]));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(z[thread_id]));

    codes[thread_id] = xx * 4 + yy * 2 + zz;
}


/**
 * Z-order decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 */
template<typename T>
__global__ void z_order_decode(
    size_t N,
    const T* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;

    x[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 2));
    y[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 1));
    z[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id]));
}

} // namespace cuda

namespace cpu {
/**
 * Z-order encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
template<typename T>
__host__ void z_order_encode(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    T* codes
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(x[thread_id]));
        T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(y[thread_id]));
        T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(z[thread_id]));

        codes[thread_id] = xx * 4 + yy * 2 + zz;
    }
}


/**
 * Z-order decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 */
template<typename T>
__host__ void z_order_decode(
    size_t N,
    const T* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        x[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 2));
        y[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id] >> 1));
        z[thread_id] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(codes[thread_id]));
    }
}

} // namespace cpu
} // namespace serialize
} // namespace flex_gemm

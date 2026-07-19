#pragma once

#include "utils.hpp"
#include <cstddef>
#include <limits>

namespace flex_gemm {
namespace serialize {

// I know that there is more efficient OMP code written in the cuda kernel
// folder but OMP threads can fight with pytorch threads in SOME cases so I'll
// use parallel_for here if it is required.
//
// For now the default one is BxyzSerialize so it is not a very high priority
// right now.

/**
 * Hilbert encode 3D points
 *
 * @param b     batch index
 * @param x     x coordinate
 * @param y     y coordinate
 * @param z     z coordinate
 * @param bit_length  bit length of the coordinates
 * @param code  hilbert encoded value
 */
template <typename T>
FORCE_INLINE void hilbert_encode(uint32_t b, uint32_t x, uint32_t y, uint32_t z,
                                 size_t bit_length, T &code) {
    uint32_t point[3] = {x, y, z};
    uint32_t m = 1 << (bit_length - 1);
    uint32_t q, p, t;

    // Inverse undo excess work
    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p; // invert
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

    // Convert transformed coordinates to Morton code (which represents Hilbert
    // index here)
    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[0]));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[1]));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[2]));

    T c_code = xx * 4 + yy * 2 + zz;
    T mask =
        (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));

    // Combine with batch index
    code = (c_code & mask) | (static_cast<T>(b) << (3 * bit_length));
}

/**
 * Hilbert decode 3D points
 *
 * @param code  hilbert encoded value
 * @param bit_length  bit length of the coordinates
 * @param b     decoded batch index
 * @param x     decoded x coordinate
 * @param y     decoded y coordinate
 * @param z     decoded z coordinate
 */
template <typename T>
FORCE_INLINE void hilbert_decode(T code, size_t bit_length, uint32_t &b,
                                 uint32_t &x, uint32_t &y, uint32_t &z) {
    // Extract Batch and Spatial Code
    T mask =
        (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    T c_code = (code & mask);
    b = static_cast<uint32_t>(code >> (3 * bit_length));

    uint32_t point[3];
    // Morton decode to get back the transformed coordinates
    point[0] = static_cast<uint32_t>(
        flex_gemm::serialize::utils::extractBits(c_code >> 2));
    point[1] = static_cast<uint32_t>(
        flex_gemm::serialize::utils::extractBits(c_code >> 1));
    point[2] =
        static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code));

    uint32_t m =
        2 << (bit_length -
              1); // equivalent to 1 << bit_length but handling loop bound
    uint32_t q, p, t;

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

    x = point[0];
    y = point[1];
    z = point[2];
}

} // namespace serialize
} // namespace flex_gemm

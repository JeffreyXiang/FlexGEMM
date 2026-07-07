#pragma once

#include "../utils.h"
#include "utils.hpp"
#include <cstddef>
#include <limits>

namespace flex_gemm {
namespace serialize {

/**
 * Z-order encode 3D points
 *
 * @param b     batch index
 * @param x     x coordinate
 * @param y     y coordinate
 * @param z     z coordinate
 * @param bit_length  bit length of the coordinates
 * @param code  z-order encoded value
 */
template <typename T>
FORCE_INLINE void z_order_encode(uint32_t b, uint32_t x, uint32_t y, uint32_t z,
                                 size_t bit_length, T &code) {
    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(x));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(y));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(z));
    T c_code = xx * 4 + yy * 2 + zz;
    T mask =
        (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    code = (c_code & mask) | (b << (3 * bit_length));
}

/**
 * Z-order decode 3D points
 *
 * @param code  z-order encoded value
 * @param bit_length  bit length of the coordinates
 * @param b     decoded batch index
 * @param x     decoded x coordinate
 * @param y     decoded y coordinate
 * @param z     decoded z coordinate
 */
template <typename T>
FORCE_INLINE void z_order_decode(T code, size_t bit_length, uint32_t &b,
                                 uint32_t &x, uint32_t &y, uint32_t &z) {
    T mask =
        (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    T c_code = (code & mask);
    x = static_cast<uint32_t>(
        flex_gemm::serialize::utils::extractBits(c_code >> 2));
    y = static_cast<uint32_t>(
        flex_gemm::serialize::utils::extractBits(c_code >> 1));
    z = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code));
    b = static_cast<uint32_t>(code >> (3 * bit_length));
}

} // namespace serialize
} // namespace flex_gemm

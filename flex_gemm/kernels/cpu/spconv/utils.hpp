
#pragma once

#include <cstddef>
#include <type_traits>

#include "../serialize/hilbert.hpp"
#include "../serialize/z_order.hpp"
#include "../utils.h"

namespace flex_gemm {
namespace spconv {

using int4 = flex_gemm::utils::int4;

template <typename T> inline bool is_safe_mul(T a, T b, T &res) {
    static_assert(std::is_unsigned<T>::value,
                  "is_safe_mul supports unsigned types only.");
    if (a == 0 || b == 0) {
        res = 0;
        return true;
    }
    res = a * b;
    return (res / a == b);
}

template <typename T> struct BxyzSerializeFunc {
    int W, H, D;

    BxyzSerializeFunc(int W, int H, int D) : W(W), H(H), D(D) {}

    FORCE_INLINE T encode(int b, int x, int y, int z) const {
        size_t key = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D +
                     (size_t)z;
        return static_cast<T>(key);
    }

    FORCE_INLINE int4 decode(T key) const {
        int b = key / ((size_t)W * H * D);
        int x = key / ((size_t)H * D) % W;
        int y = key / D % H;
        int z = key % D;
        return flex_gemm::utils::make_int4(b, x, y, z);
    }
};

template <typename T> struct ZorderSerializeFunc {
    size_t bit_length;

    ZorderSerializeFunc(size_t bit_length) : bit_length(bit_length) {}

    FORCE_INLINE T encode(int b, int x, int y, int z) const {
        T key;
        flex_gemm::serialize::z_order_encode(
            static_cast<uint32_t>(b), static_cast<uint32_t>(x),
            static_cast<uint32_t>(y), static_cast<uint32_t>(z), bit_length,
            key);
        return key;
    }

    FORCE_INLINE int4 decode(T key) const {
        uint32_t b, x, y, z;
        flex_gemm::serialize::z_order_decode(key, bit_length, b, x, y, z);
        return flex_gemm::utils::make_int4(
            static_cast<int>(b), static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(z));
    }
};

template <typename T> struct HilbertSerializeFunc {
    size_t bit_length;

    HilbertSerializeFunc(size_t bit_length) : bit_length(bit_length) {}

    FORCE_INLINE T encode(int b, int x, int y, int z) const {
        T key;
        flex_gemm::serialize::hilbert_encode(
            static_cast<uint32_t>(b), static_cast<uint32_t>(x),
            static_cast<uint32_t>(y), static_cast<uint32_t>(z), bit_length,
            key);
        return key;
    }

    FORCE_INLINE int4 decode(T key) const {
        uint32_t b, x, y, z;
        flex_gemm::serialize::hilbert_decode(key, bit_length, b, x, y, z);
        return flex_gemm::utils::make_int4(
            static_cast<int>(b), static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(z));
    }
};

} // namespace spconv
} // namespace flex_gemm

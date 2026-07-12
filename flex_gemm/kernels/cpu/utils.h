#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif

#if defined(__cpp_lib_hardware_interference_size)
constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
constexpr size_t CACHE_LINE_SIZE = 64; // Safe default for most x86_64
#endif

namespace flex_gemm {
namespace utils {
// Check if the user or build system defined 'SIMD'
#if defined(SIMD)

// GCC/Clang native vector extension
typedef int32_t int4 __attribute__((ext_vector_type(4)));

#else

// Standard C++ fallback for MSVC or non-SIMD builds
struct int4 {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
};

#endif

int4 make_int4(int32_t x, int32_t y, int32_t z, int32_t w) {
    return (struct int4){.x = x, .y = y, .z = z, .w = w};
}


} // namespace utils
} // namespace flex_gemm

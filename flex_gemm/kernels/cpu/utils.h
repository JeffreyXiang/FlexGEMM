#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

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

// Include necessary headers for CPU intrinsics
#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(__i386__) || defined(__x86_64__)
        #include <immintrin.h>
    #endif
#endif

inline void cpu_pause() {
#if defined(_MSC_VER)
    // Microsoft Visual C++ Compiler
    #if defined(_M_IX86) || defined(_M_X64)
        _mm_pause(); // x86/x64 pause
    #elif defined(_M_ARM) || defined(_M_ARM64)
        __yield();   // ARM yield
    #endif

#elif defined(__GNUC__) || defined(__clang__)
    // GCC or Clang Compiler
    #if defined(__i386__) || defined(__x86_64__)
        _mm_pause(); // x86/x64 pause
    #elif defined(__arm__) || defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory"); // ARM yield hint
    #else
        // Fallback for unknown architectures: 
        // A compiler memory barrier to prevent loop unrolling/reordering
        __asm__ __volatile__("" ::: "memory");
    #endif
#else
    // Unknown compiler fallback
#endif
}

// Make sure the atomics are right here.
class RangeBucket {
  private:
    size_t bucket_size, array_size;
    std::unique_ptr<int64_t[]> counts;
  public:
    RangeBucket(size_t array_size, size_t bucket_size)
        : bucket_size(bucket_size), array_size(array_size),
          counts(std::make_unique<int64_t[]>(bucket_size)) {
        for (int i = 0; i < bucket_size; i++) {
            counts[i] = 0;
        }
    }
    size_t index_to_bucket(size_t index){
        return (index*bucket_size)/array_size;
    }
    size_t range_index_begin(size_t index){
        return (index*array_size+bucket_size-1)/bucket_size;
    }
    size_t range_index_end(size_t index){
        return range_index_begin(index+1);
    }
    void inc(size_t index){
        // No checking here. If the index is out of range, the behaviour is undefined
        size_t bucket_id = index_to_bucket(index);

        // Do an atomic operation here as this function might be accessed by different threads
        __atomic_fetch_add(&counts[bucket_id], 1, __ATOMIC_RELAXED);
    }
    int64_t& operator[](const size_t bucket_index){
        return counts[bucket_index];
    }
    const int64_t& operator[](const size_t bucket_index) const {
        return counts[bucket_index];
    }
    size_t total_count(int index){
        size_t cnt = 0;
        for(int i = 0; i < index; i++){
            cnt+=counts[i];
        }
        return cnt;
    }
};


} // namespace utils
} // namespace flex_gemm

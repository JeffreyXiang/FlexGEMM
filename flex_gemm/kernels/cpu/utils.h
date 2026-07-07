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

enum LOCK_DATA : uint32_t { UNLOCKED = 0, LOCKED = 1, BROKEN = 2 };
template <class State> struct alignas(CACHE_LINE_SIZE) PaddedLock {
    using lock_type = std::atomic<LOCK_DATA>;
    using data_type = State;
    lock_type lock;
    data_type state;

    constexpr PaddedLock() noexcept(
        std::is_nothrow_default_constructible_v<State>)
        : lock(UNLOCKED), state{} {}

    PaddedLock(const PaddedLock &) = delete;
    PaddedLock &operator=(const PaddedLock &) = delete;

    data_type &operator*() { return state; }
    const data_type &operator*() const { return state; }

    LOCK_DATA try_lock() {
        LOCK_DATA expected = UNLOCKED;
        lock.compare_exchange_strong(expected, LOCKED,
                                     std::memory_order_acquire,
                                     std::memory_order_relaxed);
        return expected;
    }
    void mark_broken() { lock.store(BROKEN, std::memory_order_release); }
    void unlock() { lock.store(UNLOCKED, std::memory_order_release); }
};

template <class State, size_t S> struct LockArray {
    using LockType = PaddedLock<State>;
    LockType locks[S];

    LockType &operator[](size_t x) {
        assert(x >= 0 && x < S);
        return locks[x];
    }
};

} // namespace utils
} // namespace flex_gemm

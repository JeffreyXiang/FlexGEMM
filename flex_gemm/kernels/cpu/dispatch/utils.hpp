#pragma once

#include "../utils.h"

namespace flex_gemm{
namespace dispatch{

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

}
}


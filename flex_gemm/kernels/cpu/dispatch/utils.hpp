#pragma once

#include "../utils.h"

namespace flex_gemm {
namespace dispatch {

// This is a data structure used to wrap some other datastructure(class State)
// in an efficient cache_line aligned way and define functions to lock and
// unlock the state.
// I also have defined a "BROKEN" state to indicate that that unit cannot be
// unlocked. This will be useful when we just want to mark some state as "done"
// and done want other threads to bother unlocking it and figuring out that all
// the work is done and waste CPU cycles.
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
    PaddedLock(PaddedLock &&r_value) : lock(UNLOCKED), state(r_value.state) {
        __atomic_load(&r_value.lock, &lock, __ATOMIC_RELAXED);
        r_value.state = State{};
    }
    PaddedLock &operator=(PaddedLock &&r_value){
        __atomic_load(&r_value.lock, &lock, __ATOMIC_RELAXED);
        state = r_value.state;
        r_value.state = State{};
        return *this;
    }

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

} // namespace dispatch
} // namespace flex_gemm

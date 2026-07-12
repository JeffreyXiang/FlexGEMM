
#pragma once
#include "../utils.h"
#include <cstddef>
#include <cstdio>
#include <limits>

namespace flex_gemm {
namespace hash {

// 32 bit Murmur3 hash
FORCE_INLINE size_t hash(uint32_t k, size_t N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}

// 64 bit Murmur3 hash
FORCE_INLINE size_t hash(uint64_t k, size_t N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}

template <typename K>
FORCE_INLINE size_t
linear_probing_insert(K *hashmap_keys, const K key, const size_t N,
                      int release_order = __ATOMIC_RELEASE) {
    size_t slot = hash(key, N);

    K expected = std::numeric_limits<K>::max();
    while (!__atomic_compare_exchange_n(&hashmap_keys[slot], &expected, key,
                                        false, release_order,
                                        __ATOMIC_RELAXED)) {
        if (expected == key) {
            break;
        }
        expected = std::numeric_limits<K>::max();
        slot++;
        slot = (slot < N) ? slot : 0; // I write it this way so that the
                                      // compiler generates CMOV instruction
    }
    return slot; // I choose to return the slot here so i dont have to rewrite
                 // this code later
}

template <typename K>
FORCE_INLINE size_t linear_probing_insert_cpu(K *hashmap_keys, const K key,
                                              const size_t N) {
    size_t slot = hash(key, N);

    K expected = std::numeric_limits<K>::max();
    while (hashmap_keys[slot] != expected && hashmap_keys[slot] != key) {
        slot++;
        slot = (slot < N) ? slot : 0;
    }
    hashmap_keys[slot] = key;
    return slot; // I choose to return the slot here so i dont have to rewrite
                 // this code later
}

template <typename K, typename V>
FORCE_INLINE size_t linear_probing_insert(K *hashmap_keys, V *hashmap_values,
                                          const K key, const V value,
                                          const size_t N) {
    size_t slot = linear_probing_insert(hashmap_keys, key, N, __ATOMIC_RELAXED);
    __atomic_store_n(&hashmap_values[slot], value, __ATOMIC_RELEASE);
    return slot;
}

template <typename K, typename V>
FORCE_INLINE V linear_probing_lookup(const K *hashmap_keys,
                                     const V *hashmap_values, const K key,
                                     const size_t N) {
    size_t slot = hash(key, N);
    while (true) {
        // K prev = hashmap_keys[slot].load(std::memory_order_relaxed);
        K prev = __atomic_load_n(&hashmap_keys[slot], __ATOMIC_RELAXED);
        if (prev == std::numeric_limits<K>::max()) {
            return std::numeric_limits<V>::max();
        }
        if (prev == key) {
            return __atomic_load_n(&hashmap_keys[slot], __ATOMIC_ACQUIRE);
        }
        slot++;
        slot = (slot < N) ? slot : 0; // I write it this way so that the
                                      // compiler generates CMOV instruction
    }
}

} // namespace hash
} // namespace flex_gemm

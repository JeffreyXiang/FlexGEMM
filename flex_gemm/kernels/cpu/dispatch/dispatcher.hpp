#pragma once

#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace flex_gemm {
namespace dispatch {

// This is a bucket dispatcher class. This lets each thread lock a bucket to
// perform calculations without needing to do any more CPU synchronization.
// It takes 1 template argument which is the Bucket class that each bucket will
// store. In the constructor, it also takes in a function used to initialize
// each bucket.
template <class Bucket> class BucketDispatcher {
  public:
    using BucketType = flex_gemm::dispatch::PaddedLock<Bucket>;
    using OptRefBucket = std::optional<std::reference_wrapper<Bucket>>;

  private:
    int num_of_buckets;
    std::unique_ptr<BucketType[]> buckets;
    int num_of_threads;
    std::unique_ptr<int[]> bucket_lock;

  public:
    template <class BucketInitFunc>
    BucketDispatcher(int num_of_buckets, int num_of_threads,
                     BucketInitFunc bucket_init_func)
        : num_of_buckets(num_of_buckets), num_of_threads(num_of_threads),
          buckets(std::make_unique<BucketType[]>(num_of_buckets)),
          bucket_lock(std::make_unique<int[]>(num_of_threads)) {
        for (int i = 0; i < num_of_buckets; i++) {
            bucket_init_func(*buckets[i], i);
        }

        for (int i = 0; i < num_of_threads; i++) {
            bucket_lock[i] = -1;
        }
    }
    BucketDispatcher(BucketDispatcher &other) = delete;
    BucketDispatcher &operator=(BucketDispatcher &other) = delete;
    BucketDispatcher(BucketDispatcher &&r_value)
        : num_of_buckets(r_value.num_of_buckets),
          num_of_threads(r_value.num_of_threads),
          buckets(std::move(r_value.buckets)),
          bucket_lock(std::move(r_value.bucket_lock)) {}
    BucketDispatcher &operator=(BucketDispatcher &&r_value) {
        num_of_buckets = r_value.num_of_buckets;
        num_of_threads = r_value.num_of_threads;
        buckets = std::move(r_value.buckets);
        bucket_lock = std::move(r_value.bucket_lock);
        return *this;
    }
    void register_thread_lock(int thread_id, int bucket_id) {
        // Be careful. This function overwrites the previous lock.
        bucket_lock[thread_id] = bucket_id;
    }
    void free_thread_lock(int thread_id) { bucket_lock[thread_id] = -1; }
    OptRefBucket try_lock(int thread_id, int num_of_tries) {
        // This is non blocking schedule
        // This function does not check if you have locked something before so
        // be sure to free it. If you try to lock 2 places at once, the function
        // will forget the first locked bucket_id and it will not be able to
        // unlock it. linear lock schedule
        int start = thread_id % num_of_buckets;
        for (int i = 0; i < num_of_tries; i++) {
            for (int j = 0; j < num_of_buckets; j++) {
                LOCK_DATA result = buckets[start].try_lock();
                if (result == UNLOCKED) {
                    register_thread_lock(thread_id, start);
                    return OptRefBucket{*buckets[start]};
                }

                start++;
                start = (start >= num_of_buckets) ? 0 : start;
            }
        }
        return OptRefBucket{};
    }
    Bucket &lock(int thread_id) {
        // This is blocking schedule
        // This function does not check if you have locked something before so
        // be sure to free it. If you try to lock 2 places at once, the function
        // will forget the first locked bucket_id and it will not be able to
        // unlock it. linear lock schedule
        int start = thread_id % num_of_buckets;
        while (true) {
            // fprintf(stdout, "Thread %d: LOCKER: Trying %d\n", thread_id,
            // start); fflush(stdout);
            LOCK_DATA result = buckets[start].try_lock();
            if (result == UNLOCKED) {
                register_thread_lock(thread_id, start);
                return *buckets[start];
            }

            start++;
            start = (start >= num_of_buckets) ? 0 : start;
        }
    }
    void mark_broken(int thread_id) {
        // This is miscellaneous function. I will use it to indicate that there
        // is nothing to be done in the bucket making it unable to lock by any
        // thread.
        int bucket_id = bucket_lock[thread_id];
        buckets[bucket_id].mark_broken();
        free_thread_lock(thread_id);
    }
    void unlock(int thread_id) {
        // Again, if you try to free something that was not locked, the result
        // is undefined
        int bucket_id = bucket_lock[thread_id];
        buckets[bucket_id].unlock();
        free_thread_lock(thread_id);
    }
    void status(FILE *file, int thread_id) {
        // assume the bucket has bucket_id, start, and end
        for (int i = 0; i < num_of_buckets; i++) {
            Bucket &state = *buckets[i];
            size_t total = state.end - state.start;
            fprintf(file, "Thread %d: Query: Bucket %ld : %ld\n", thread_id,
                    state.bucket_id, total);
        }
    }
    ~BucketDispatcher() {}
};
} // namespace dispatch
} // namespace flex_gemm

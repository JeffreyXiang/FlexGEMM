#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

namespace flex_gemm {
namespace dispatch {
template <class Bucket> class BucketDispatcher {
    using BucketType = flex_gemm::dispatch::PaddedLock<Bucket>;

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
    void register_thread_lock(int thread_id, int bucket_id) {
        // Be careful. This function overwrites the previous lock.
        bucket_lock[thread_id] = bucket_id;
    }
    void free_thread_lock(int thread_id) { bucket_lock[thread_id] = -1; }
    Bucket &lock(int thread_id) {
        // This is blocking schedule
        // This function does not check if you have locked something before so
        // be sure to free it. If you try to lock 2 places at once, the function
        // will forget the first locked bucket_id and it will not be able to
        // unlock. linear lock schedule
        // int track = 20;
        int start = thread_id % num_of_buckets;
        while (true) {
            //fprintf(stdout, "Thread %d: LOCKER: Trying %d\n", thread_id, start);
            //fflush(stdout);
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

        //fprintf(stdout, "Thread %d: BROKEN: Broken %d\n", thread_id, bucket_id);
        //fflush(stdout);
    }
    void unlock(int thread_id) {
        // Again, if you try to free something that was not locked, the result
        // is undefined
        int bucket_id = bucket_lock[thread_id];
        buckets[bucket_id].unlock();
        free_thread_lock(thread_id);

        //fprintf(stdout, "Thread %d: UNLOCKED: Unlocked %d\n", thread_id, bucket_id);
        //fflush(stdout);
    }
    ~BucketDispatcher() {}
};
} // namespace dispatch
} // namespace flex_gemm

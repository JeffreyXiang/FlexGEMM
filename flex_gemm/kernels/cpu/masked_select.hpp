#pragma once

#include "dispatch/dispatcher.hpp"
#include "utils.h"
#include <ATen/Parallel.h>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <omp.h>
#include <optional>
#include <torch/extension.h>

#define GRAIN_SIZE 256

namespace flex_gemm {
namespace select {
template <class T>
void remove_select_cpu(size_t input_size, size_t output_size,
                       size_t bucket_size, utils::RangeBucket &range_bucket,
                       T remove_element, const T * __restrict__ input, T * __restrict__ output) {
    int num_threads = at::get_num_threads();

    // Make the CPU dispatch here.
    struct State {
        size_t bucket_id;
        size_t start, end;
        size_t ptr;
    };

    auto state_init = [&](State &state, size_t bucket_id) {
        state.bucket_id = bucket_id;
        state.start = range_bucket.range_index_begin(bucket_id);
        state.end = range_bucket.range_index_end(bucket_id);
        state.ptr = range_bucket.total_count(bucket_id);
    };

    dispatch::BucketDispatcher<State> dispatch(bucket_size, num_threads,
                                               state_init);

    // Parallel it out here.
    at::parallel_for(0, input_size, GRAIN_SIZE, [&](size_t start, size_t end) {
        size_t thread_id = at::get_thread_num();
        size_t total = end - start;
        while (total > 0) {
            State &state = dispatch.lock(thread_id);
            size_t it;
            size_t ptr = state.ptr;
            for (it = state.start; it < state.end && total > 0; it++, total--) {
                T elem = input[it];
                // Remove this effy code rn and just add a branch
                output[ptr] = (elem != remove_element) ? elem : output[ptr];
                ptr += (elem != remove_element) ? 1 : 0;
            }
            state.start = it;
            state.ptr = ptr;
            if (state.start >= state.end) {
                dispatch.mark_broken(thread_id);
            } else {
                dispatch.unlock(thread_id);
            }
        }
    });
}

template <class T>
void remove_select_cpu_omp(size_t input_size, size_t output_size,
                           size_t bucket_size, utils::RangeBucket &range_bucket,
                           T remove_element, const T *input, T *output) {
    static_assert(false, "This function does not work yet. Do not use it.");
    int num_threads = at::get_num_threads();

    // Make the CPU dispatch here.
    struct State {
        size_t bucket_id;
        size_t start, end;
        size_t ptr;
    };

    auto state_init = [&](State &state, size_t bucket_id) {
        state.bucket_id = bucket_id;
        state.start = range_bucket.range_index_begin(bucket_id);
        state.end = range_bucket.range_index_end(bucket_id);
        state.ptr = range_bucket.total_count(bucket_id);
    };

    dispatch::BucketDispatcher<State> dispatch(bucket_size, num_threads,
                                               state_init);

    fprintf(stdout, "Threads Starting %ld\n", input_size);
    fflush(stdout);

    size_t counter = 0; // PURELY FOR DEBUGGINg

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        using OptStateRef =
            typename dispatch::BucketDispatcher<State>::OptRefBucket;
        OptStateRef state = OptStateRef{};
        size_t it = std::numeric_limits<size_t>::max();
        size_t end = std::numeric_limits<size_t>::max();
        size_t ptr = std::numeric_limits<size_t>::max();

// Parallel it out here.
#pragma omp for schedule(dynamic, GRAIN_SIZE)
        for (size_t total = 0; total < input_size; total++) {
            if (!state.has_value()) {
                fprintf(stdout, "Thread %d: Trying to lock!\n", thread_id);
                dispatch.status(stdout, thread_id);
                fflush(stdout);
                state = dispatch.try_lock(thread_id, 5);
                if(state.has_value()){
                    it = state->get().start;
                    ptr = state->get().ptr;
                    end = state->get().end;
                    fprintf(stdout, "Thread %d: %d Locked!\tStart: %ld\tEnd: %ld\n",
                            thread_id, state->get().bucket_id, it, end);
                    fflush(stdout);
                }else{
                    // Could not lock
                    // Yeah this wont fix it
                }

                total--; // Total not counted here
            } else if (it >= end) {
                state->get().start = it;
                state->get().ptr = ptr;
                if (state->get().start >= state->get().end) {
                    dispatch.mark_broken(thread_id);
                } else {
                    dispatch.unlock(thread_id);
                }
                state = OptStateRef{};
                fprintf(stdout, "Thread %d: %d Unlocked with %ld counter!\n",
                        thread_id, state->get().bucket_id, counter);
                fflush(stdout);

                total--; // Total not counted here
            } else {
                __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
                fprintf(stdout, "Thread %d: Counter: %d\tIt: %ld\n", thread_id,
                        counter, it);
                fflush(stdout);

                T elem = input[it];
                // Remove this effy code rn and just add a branch
                output[ptr] = (elem != remove_element) ? elem : output[ptr];
                ptr += (elem != remove_element) ? 1 : 0;

                it++;
            }
        }

        if (state.has_value()) {
            state->get().start = it;
            state->get().ptr = ptr;
            if (state->get().start >= state->get().end) {
                dispatch.mark_broken(thread_id);
            } else {
                dispatch.unlock(thread_id);
            }
            state = OptStateRef{};
            fprintf(stdout, "Thread %d: %d Unlocked with %ld counter!\n",
                    thread_id, state->get().bucket_id, counter);
            fflush(stdout);
        }

        fprintf(stdout, "Thread %d: Ending with counter %ld\n", thread_id,
                counter);
        fflush(stdout);
    }

    fprintf(stdout, "Ending Threads\n");
    fflush(stdout);
}
} // namespace select
} // namespace flex_gemm

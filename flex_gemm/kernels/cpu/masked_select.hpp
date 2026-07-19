#pragma once

#include "dispatch/dispatcher.hpp"
#include "utils.h"
#include <ATen/Parallel.h>
#include <cstddef>
#include <cstdio>
#include <omp.h>
#include <torch/extension.h>

#define GRAIN_SIZE 256

namespace flex_gemm {
namespace select {

// This is an efficient masked select operation for the CPU. The
// torch::masked_select is wildly inefficient for the CPU for some reason.
template <class T>
void remove_select_cpu(size_t input_size, size_t output_size,
                       size_t bucket_size, utils::RangeBucket &range_bucket,
                       T remove_element, const T *__restrict__ input,
                       T *__restrict__ output) {
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

} // namespace select
} // namespace flex_gemm

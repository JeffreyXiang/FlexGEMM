
#include <ATen/Parallel.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <torch/extension.h>

#include "../hash/hash.hpp"
#include "../utils.h"
#include "utils.hpp"

#include "../dispatch/dispatcher.hpp"

#define GRAIN_SIZE 256
#define MIN_HASHMAP_SEGMENT_SIZE 32

namespace flex_gemm {
namespace spconv {

FORCE_INLINE size_t bit_length(uint64_t x) {
  if (x == 0)
    return 0;
  size_t len = 0;

#pragma unroll
  for (int i = 5; i >= 0; --i) {
    uint64_t shift = 1 << i;

    bool mask = (x >= (1U << shift));
    x >>= (mask) ? shift : 0;
    len += (mask) ? shift : 0;
  }
  return len + 1;
}

using int4 = flex_gemm::utils::int4;

template <class T, class SerializeFunc>
void build_sparse_conv_out_coords_hashmap_insert_kernel_cpu_dispatched(
    const size_t N, const size_t M, const int Wo, const int Ho, const int Do,
    const int V, const int Kw, const int Kh, const int Kd, const int Sw,
    const int Sh, const int Sd, const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const int32_t *__restrict__ coords, T *__restrict__ hashmap_keys,
    SerializeFunc serialize_func) {
  int num_of_threads = at::get_num_threads();
  int num_of_buckets = num_of_threads;
  int min_to_equal = MIN_HASHMAP_SEGMENT_SIZE * num_of_buckets;
  num_of_buckets =
      (N >= min_to_equal) ? num_of_buckets : N / MIN_HASHMAP_SEGMENT_SIZE;
  num_of_buckets = (num_of_buckets > 0) ? num_of_buckets : 1;

  // fprintf(stdout, "M*V: %ld\tN: %ld\n", M * V, N);
  //fprintf(stdout, "Number of Buckets: %d\tNumber of Threads: %d\n", num_of_buckets, num_of_threads);
  //fflush(stdout);

  struct ProblemState {
    size_t start;
    size_t end;

    // Divide hashmap_keys here.
    T *hashmap_chunk;
    size_t hashmap_size;
  };
  auto problem_state_init = [&](ProblemState &state, int &bucket_id) {
    size_t temp0 = M * V;
    size_t temp1 = temp0 * bucket_id;
    size_t temp2 = temp1 + temp0;
    state.start = temp1 / num_of_buckets;
    state.end = temp2 / num_of_buckets;

    size_t temp3 = N;
    size_t temp4 = temp3 * bucket_id;
    size_t temp5 = temp3 + temp4;
    state.hashmap_chunk = hashmap_keys + (temp4 / num_of_buckets);
    state.hashmap_size = (temp5 / num_of_buckets) - (temp4 / num_of_buckets);
    // Check if the above is ACTUALLY CORRECT!!!
    //fprintf(stdout,
    // "State: Assigned %d: start=%ld\tend=%ld\thashmap_size=%ld\n",
    // bucket_id, state.start, state.end, state.hashmap_size);
    //fflush(stdout);
  };
  dispatch::BucketDispatcher<ProblemState> dispatcher(
      num_of_buckets, num_of_threads, problem_state_init);

  at::parallel_for(0, M * V, GRAIN_SIZE, [&](size_t start, size_t end) {
    size_t total = end - start;
    // get the thread_id here;
    int thread_id = at::get_thread_num();
    while (total) {
      //fprintf(stdout, "Thread Id %d: Trying to lock %ld\n", thread_id, total);
      //fflush(stdout);

      ProblemState &dis = dispatcher.lock(thread_id);

      //fprintf(stdout, "Thread Id %d: Locked onto %ld: %ld\n", thread_id,
      // dis.start, dis.end);
      //fflush(stdout);

      size_t t_id;
      for (t_id = dis.start; t_id < dis.end && total > 0; t_id++, total--) {
        const size_t idx = t_id / V;
        int4 coord = reinterpret_cast<const int4 *>(coords)[idx];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int v = t_id % V;
        int kx = v / (Kh * Kd);
        int ky = v / Kd % Kh;
        int kz = v % Kd;
        int out_x = (x + Pw - kx * Dw);
        int out_y = (y + Ph - ky * Dh);
        int out_z = (z + Pd - kz * Dd);
        if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
          out_x /= Sw;
          out_y /= Sh;
          out_z /= Sd;
          if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho &&
              out_z >= 0 && out_z < Do) {
            T key = serialize_func.encode(b, out_x, out_y, out_z);
            flex_gemm::hash::linear_probing_insert_cpu(dis.hashmap_chunk, key,
                                                       dis.hashmap_size);
          }
        }
      }
      dis.start = t_id;
      if (dis.start >= dis.end) {
        dispatcher.mark_broken(thread_id);
        //fprintf(stdout, "Thread Id %d: Marked broken\n", thread_id);
        //fflush(stdout);
      } else {
        dispatcher.unlock(thread_id);
        //fprintf(stdout, "Thread Id %d: Unlocked\n", thread_id);
        //fflush(stdout);
      }
    }
  });
}

template <class T, class SerializeFunc>
void build_sparse_conv_out_coords_hashmap_insert_kernel(
    const size_t N, const size_t M, const int Wo, const int Ho, const int Do,
    const int V, const int Kw, const int Kh, const int Kd, const int Sw,
    const int Sh, const int Sd, const int Pw, const int Ph, const int Pd,
    const int Dw, const int Dh, const int Dd,
    const int32_t *__restrict__ coords, T *__restrict__ hashmap_keys,
    SerializeFunc serialize_func) {

  // There is a chance that the coords are not aligned to 16 bytes (if it is a
  // view or something) so we have to take this precaution.
  bool is_aligned = (reinterpret_cast<std::uintptr_t>(coords) % 16) == 0;

  if (is_aligned) {
    at::parallel_for(0, M * V, GRAIN_SIZE, [&](size_t start, size_t end) {
      for (size_t t_id = start; t_id < end; t_id++) {
        const size_t idx = t_id / V;
        int4 coord = reinterpret_cast<const int4 *>(coords)[idx];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        int v = t_id % V;
        int kx = v / (Kh * Kd);
        int ky = v / Kd % Kh;
        int kz = v % Kd;
        int out_x = (x + Pw - kx * Dw);
        int out_y = (y + Ph - ky * Dh);
        int out_z = (z + Pd - kz * Dd);
        if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
          out_x /= Sw;
          out_y /= Sh;
          out_z /= Sd;
          if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho &&
              out_z >= 0 && out_z < Do) {
            T key = serialize_func.encode(b, out_x, out_y, out_z);
            flex_gemm::hash::linear_probing_insert(hashmap_keys, key, N);
          }
        }
      }
    });
  } else {
    // Not aligned version
    at::parallel_for(0, M * V, GRAIN_SIZE, [&](size_t start, size_t end) {
      for (size_t t_id = start; t_id < end; t_id++) {
        const size_t idx = t_id / V;
        // int4 coord = reinterpret_cast<const int4 *>(coords)[idx];
        const int32_t *root = coords + (idx * 4);
        int b = *root;
        int x = *(root + 1);
        int y = *(root + 2);
        int z = *(root + 3);
        int v = t_id % V;
        int kx = v / (Kh * Kd);
        int ky = v / Kd % Kh;
        int kz = v % Kd;
        int out_x = (x + Pw - kx * Dw);
        int out_y = (y + Ph - ky * Dh);
        int out_z = (z + Pd - kz * Dd);
        if (out_x % Sw == 0 && out_y % Sh == 0 && out_z % Sd == 0) {
          out_x /= Sw;
          out_y /= Sh;
          out_z /= Sd;
          if (out_x >= 0 && out_x < Wo && out_y >= 0 && out_y < Ho &&
              out_z >= 0 && out_z < Do) {
            T key = serialize_func.encode(b, out_x, out_y, out_z);
            flex_gemm::hash::linear_probing_insert(hashmap_keys, key, N);
          }
        }
      }
    });
  }
}

template <typename T, typename SerializeFunc>
void build_sparse_conv_out_coords_decode_key_kernel(
    const size_t N, const int Wo, const int Ho, const int Do,
    const T *__restrict__ valid_keys, int32_t *__restrict__ out_coords,
    SerializeFunc serialize_func) {

  // Have to cache line aware this code.
  // REALLY HAVE TO MAKE THIS CACHE AWARE HERE
  // THERE CAN BE MISALIGNMENT
  at::parallel_for(0, N, GRAIN_SIZE, [&](size_t start, size_t end) {
    for (size_t t_id = start; t_id < end; t_id++) {
      T key = valid_keys[t_id];
      *reinterpret_cast<int4 *>(out_coords + t_id * 4) =
          serialize_func.decode(key);
    }
  });
}

torch::Tensor hashmap_build_sparse_conv_out_coords(
    const torch::Tensor &in_coords, const float hashmap_ratio,
    const int serialize_mode, int B, int W, int H, int D, int Kw, int Kh,
    int Kd, int Sw, int Sh, int Sd, int Pw, int Ph, int Pd, int Dw, int Dh,
    int Dd) {
  // Calculate output size
  int Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1;
  int Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1;
  int Do = (D + 2 * Pd - Dd * (Kd - 1) - 1) / Sd + 1;
  int V = Kw * Kh * Kd;
  size_t hashmap_size = static_cast<size_t>(hashmap_ratio * Kw * Kh * Kd / Sw /
                                            Sh / Sd * in_coords.size(0));

  // Implementation
  auto impl = [&](auto type_tag, auto serialize_func) {
    using T = decltype(type_tag);
    constexpr auto udtype = sizeof(T) == 4 ? torch::kUInt32 : torch::kUInt64;
    constexpr auto dtype = sizeof(T) == 4 ? torch::kInt32 : torch::kInt64;

    torch::Tensor hashmap_keys = torch::full(
        {static_cast<int64_t>(hashmap_size)}, std::numeric_limits<T>::max(),
        torch::dtype(udtype).device(in_coords.device()));

    //fprintf(stdout, "Starting Dispatch\n");
    //fflush(stdout);

    build_sparse_conv_out_coords_hashmap_insert_kernel_cpu_dispatched(
        hashmap_keys.size(0), in_coords.size(0), Wo, Ho, Do, V, Kw, Kh, Kd, Sw,
        Sh, Sd, Pw, Ph, Pd, Dw, Dh, Dd, in_coords.data_ptr<int32_t>(),
        hashmap_keys.data_ptr<T>(), serialize_func);

    //fprintf(stdout, "Dispatch ended\n");
    //fflush(stdout);

    torch::Tensor valid_key = hashmap_keys.view(dtype).masked_select(
        hashmap_keys != std::numeric_limits<T>::max());
    // valid_key = std::get<0>(valid_key.sort()).view(udtype);
    valid_key = std::get<0>(at::_unique(valid_key, true, false)).view(udtype);

    torch::Tensor out_coords =
        torch::empty({valid_key.size(0), 4},
                     torch::dtype(torch::kInt32).device(hashmap_keys.device()));

    build_sparse_conv_out_coords_decode_key_kernel(
        valid_key.size(0), Wo, Ho, Do, valid_key.data_ptr<T>(),
        out_coords.data_ptr<int32_t>(), serialize_func);

    return out_coords;
  };

  if (serialize_mode == 0) { // bxyz
    uint64_t VOL;
    bool safe = true;
    safe &=
        is_safe_mul(static_cast<uint64_t>(B), static_cast<uint64_t>(Wo), VOL);
    safe &= is_safe_mul(VOL, static_cast<uint64_t>(Ho), VOL);
    safe &= is_safe_mul(VOL, static_cast<uint64_t>(Do), VOL);
    if (!safe) {
      TORCH_CHECK(false,
                  "The spatial size is too large. Require B*W*H*D < 2^64.");
    }

    if (VOL < std::numeric_limits<uint32_t>::max()) {
      return impl(uint32_t(), BxyzSerializeFunc<uint32_t>(Wo, Ho, Do));
    } else if (VOL < std::numeric_limits<uint64_t>::max()) {
      return impl(uint64_t(), BxyzSerializeFunc<uint64_t>(Wo, Ho, Do));
    } else {
      TORCH_CHECK(false,
                  "The spatial size is too large. Require B*W*H*D < 2^64.");
    }
  } else if (serialize_mode == 1) { // zorder
    int max_spatial = std::max(std::max(W, H), D);
    int bit_length_spatial = bit_length(max_spatial - 1);
    int bit_length_batch = bit_length(B - 1);
    int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

    if (bit_length_total < 32) {
      return impl(uint32_t(),
                  ZorderSerializeFunc<uint32_t>(bit_length_spatial));
    } else if (bit_length_total < 64) {
      return impl(uint64_t(),
                  ZorderSerializeFunc<uint64_t>(bit_length_spatial));
    } else {
      TORCH_CHECK(false, "The spatial size is too large. Require total "
                         "bit length < 64.");
    }
  } else if (serialize_mode == 2) { // hilbert
    int max_spatial = std::max(std::max(W, H), D);
    int bit_length_spatial = bit_length(max_spatial - 1);
    int bit_length_batch = bit_length(B - 1);
    int bit_length_total = bit_length_batch + 3 * bit_length_spatial;

    if (bit_length_total < 32) {
      return impl(uint32_t(),
                  HilbertSerializeFunc<uint32_t>(bit_length_spatial));
    } else if (bit_length_total < 64) {
      return impl(uint64_t(),
                  HilbertSerializeFunc<uint64_t>(bit_length_spatial));
    } else {
      TORCH_CHECK(false, "The spatial size is too large. Require total "
                         "bit length < 64.");
    }
  } else {
    TORCH_CHECK(false, "Unsupported serialize mode.");
  }
}

} // namespace spconv
} // namespace flex_gemm

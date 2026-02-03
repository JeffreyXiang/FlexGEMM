/*
 * Neighbor map for sparse submanifold convolution
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>


#define BLOCK_SIZE 256


namespace flex_gemm {
namespace spconv {

/**
 * Interpret the neighbor bitmask as a Gray-code word and sort elements by its
 * decoded binary index. This induces a Gray-order linearization of the mask
 * space, grouping similar neighbor patterns together, which reduces kernel
 * specialization and active pattern diversity within a thread block.
 *
 * @param neighbor_map     [N, V] uint32 tensor containing the neighbor map
 *
 * @return                [N] neighbor mask (interpreted as Gray code)
 *                        [N] indices sorted by Gray traversal order
 */
std::tuple<torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(
    const torch::Tensor& neighbor_map
);


/**
 * Interpret the neighbor bitmask as a Gray-code word and sort elements by its
 * decoded binary index. 
 * Also prepare valid pairs for masked implicit gemm bachward pass
 * 
 * @param neighbor_map     [N, V] uint32 tensor containing the neighbor map
 * 
 * @return                 [N] gray code
 *                         [N] sorted idx
 *                         [N] valid signal idx for input tensor
 *                         [N] valid signal idx for output tensor
 *                         [V+1] valid signal segment
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
);


/**
 * Get valid kernel indices for masked implicit gemm
 * 
 * @param gray_code     [N] gray code
 * @param sorted_idx    [N] sorted idx
 * @param block_size    the block size of CUDA kernel (must be power of 2)
 * 
 * @return              [L] uint32 tensor containing the valid kernel indices
 *                      [num_blocks + 1] uint32 tensor containing the segment of valid kernel indices
 */
std::tuple<torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_2(
    const torch::Tensor& gray_code,
    const torch::Tensor& sorted_idx,
    int block_size
);

} // namespace spconv
} // namespace flex_gemm

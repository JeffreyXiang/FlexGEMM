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
 * Build sparse submanifold convolution neighbor map with hashmap
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 *  
 * @return              [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution neighbor map
 */
torch::Tensor hashmap_build_submanifold_conv_neighbour_map(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd
);

} // namespace spconv
} // namespace flex_gemm

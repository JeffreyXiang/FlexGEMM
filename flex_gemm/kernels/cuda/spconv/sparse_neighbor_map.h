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
 * Build sparse convolution neighbor map with hashmap
 * 
 * @param in_coords         [M, 4] int32 tensor containing the coordinates of input tensor
 * @param hashmap_ratio     the ratio of hashmap size to the potential output size
 * @param serialize_mode    the serialize mode, 0 for bxyz, 1 for z_order, 2 for hilbert
 * @param B                 the number of batch dimensions
 * @param W                 the number of width dimensions
 * @param H                 the number of height dimensions
 * @param D                 the number of depth dimensions
 * @param Kw                the number of width kernel dimensions
 * @param Kh                the number of height kernel dimensions
 * @param Kd                the number of depth kernel dimensions
 * @param Sw                the stride of width
 * @param Sh                the stride of height
 * @param Sd                the stride of depth
 * @param Pw                the padding of width
 * @param Ph                the padding of height
 * @param Pd                the padding of depth
 * @param Dw                the dialation of width
 * @param Dh                the dialation of height
 * @param Dd                the dialation of depth
 *  
 * @return                  [L, 4] uint32 tensor containing the sparse convolution output coordinates
 */
torch::Tensor hashmap_build_sparse_conv_out_coords(
    const torch::Tensor& in_coords,
    const float hashmap_ratio,
    const int serialize_mode,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
);


/**
 * Build sparse convolution neighbor map with expand-unique
 * 
 * @param in_coords         [M, 4] int32 tensor containing the coordinates of input tensor
 * @param serialize_mode    the serialize mode, 0 for bxyz, 1 for z_order, 2 for hilbert
 * @param B                 the number of batch dimensions
 * @param W                 the number of width dimensions
 * @param H                 the number of height dimensions
 * @param D                 the number of depth dimensions
 * @param Kw                the number of width kernel dimensions
 * @param Kh                the number of height kernel dimensions
 * @param Kd                the number of depth kernel dimensions
 * @param Sw                the stride of width
 * @param Sh                the stride of height
 * @param Sd                the stride of depth
 * @param Pw                the padding of width
 * @param Ph                the padding of height
 * @param Pd                the padding of depth
 * @param Dw                the dialation of width
 * @param Dh                the dialation of height
 * @param Dd                the dialation of depth
 *  
 * @return                  [L, 4] uint32 tensor containing the sparse convolution output coordinates
 */
torch::Tensor expand_unique_build_sparse_conv_out_coords(
    const torch::Tensor& in_coords,
    const int serialize_mode,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
);


/**
 * Build sparse convolution neighbor map with hashmap
 * 
 * @param in_coords         [M, 4] int32 tensor containing the coordinates of input tensor
 * @param out_coords        [L, 4] int32 tensor containing the coordinates of output tensor
 * @param hashmap_ratio     the ratio of hashmap size to the potential output size
 * @param include_bwd       whether to include the backward neighbor map
 * @param B                 the number of batch dimensions
 * @param W                 the number of width dimensions
 * @param H                 the number of height dimensions
 * @param D                 the number of depth dimensions
 * @param Kw                the number of width kernel dimensions
 * @param Kh                the number of height kernel dimensions
 * @param Kd                the number of depth kernel dimensions
 * @param Sw                the stride of width
 * @param Sh                the stride of height
 * @param Sd                the stride of depth
 * @param Pw                the padding of width
 * @param Ph                the padding of height
 * @param Pd                the padding of depth
 * @param Dw                the dialation of width
 * @param Dh                the dialation of height
 * @param Dd                the dialation of depth
 *  
 * @return                  [L, Kw * Kh * Kd] uint32 tensor containing the sparse convolution neighbor map for forward pass
 *                          [M, Kw * Kh * Kd] optional uint32 tensor containing the sparse convolution neighbor map for backward pass
 */
std::tuple<torch::Tensor, torch::Tensor> hashmap_build_sparse_conv_neighbour_map(
    const torch::Tensor& in_coords,
    const torch::Tensor& out_coords,
    const float hashmap_ratio,
    const bool include_bwd,
    int B, int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Sw, int Sh, int Sd,
    int Pw, int Ph, int Pd,
    int Dw, int Dh, int Dd
);

} // namespace spconv
} // namespace flex_gemm

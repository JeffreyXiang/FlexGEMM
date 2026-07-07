
#pragma once
#include <torch/extension.h>

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
    const torch::Tensor &in_coords, const float hashmap_ratio,
    const int serialize_mode, int B, int W, int H, int D, int Kw, int Kh,
    int Kd, int Sw, int Sh, int Sd, int Pw, int Ph, int Pd, int Dw, int Dh,
    int Dd);

} // namespace spconv
} // namespace flex_gemm

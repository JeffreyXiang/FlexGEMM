#include <torch/extension.h>

#include "spconv/api.h"

using namespace flex_gemm;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashmap_build_sparse_conv_out_coords", &spconv::hashmap_build_sparse_conv_out_coords);
}

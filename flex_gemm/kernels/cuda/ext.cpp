#include <torch/extension.h>
#include "hash/api.h"
#include "serialize/api.h"
#include "grid_sample/api.h"
#include "spconv/api.h"


using namespace flex_gemm;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hash functions
    m.def("hashmap_insert", &hash::hashmap_insert);
    m.def("hashmap_lookup", &hash::hashmap_lookup);
    m.def("hashmap_insert_3d", &hash::hashmap_insert_3d);
    m.def("hashmap_lookup_3d", &hash::hashmap_lookup_3d);
    m.def("hashmap_insert_3d_idx_as_val", &hash::hashmap_insert_3d_idx_as_val);

    // Serialization functions
    m.def("z_order_encode", &serialize::z_order_encode);
    m.def("z_order_decode", &serialize::z_order_decode);
    m.def("hilbert_encode", &serialize::hilbert_encode);
    m.def("hilbert_decode", &serialize::hilbert_decode);

    // Grid sample functions
    m.def("hashmap_build_grid_sample_3d_nearest_neighbor_map", &grid_sample::hashmap_build_grid_sample_3d_nearest_neighbor_map);
    m.def("hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight", &grid_sample::hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight);
   
    // Convolution functions
    m.def("hashmap_build_submanifold_conv_neighbour_map", &spconv::hashmap_build_submanifold_conv_neighbour_map);
    m.def("hashmap_build_sparse_conv_out_coords", &spconv::hashmap_build_sparse_conv_out_coords);
    m.def("expand_unique_build_sparse_conv_out_coords", &spconv::expand_unique_build_sparse_conv_out_coords);
    m.def("hashmap_build_sparse_conv_neighbour_map", &spconv::hashmap_build_sparse_conv_neighbour_map);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_1", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_1);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_2", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_2);
}

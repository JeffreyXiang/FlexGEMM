#include <torch/extension.h>
#include "api.h"
#include "z_order.h"
#include "hilbert.h"


namespace flex_gemm {
namespace serialize {

void z_order_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z,
    torch::Tensor& codes
) {
    // Call kernel
    if (x.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::z_order_encode<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::z_order_encode<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::z_order_encode(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::z_order_encode(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> z_order_decode(
    const torch::Tensor& codes
) {
    // Allocate output tensors
    torch::Tensor x = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor y = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor z = torch::empty_like(codes, torch::dtype(torch::kInt32));

    // Call kernel
    if (codes.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::z_order_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::z_order_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::z_order_decode(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::z_order_decode(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    }

    return std::make_tuple(x, y, z);
}


void hilbert_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z,
    torch::Tensor& codes
) {
    // Call kernel
    if (x.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::hilbert_encode<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::hilbert_encode<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::hilbert_encode(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::hilbert_encode(
                x.size(0),
                reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hilbert_decode(
    const torch::Tensor& codes
) {
    // Allocate output tensors
    torch::Tensor x = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor y = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor z = torch::empty_like(codes, torch::dtype(torch::kInt32));

    // Call kernel
    if (codes.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::hilbert_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::hilbert_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::hilbert_decode(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::hilbert_decode(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
                reinterpret_cast<uint32_t*>(z.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    }

    return std::make_tuple(x, y, z);
}

} // namespace serialize
} // namespace flex_gemm

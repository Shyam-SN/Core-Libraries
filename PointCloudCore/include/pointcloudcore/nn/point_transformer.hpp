#pragma once

#include <pointcloudcore/core/tensor.hpp>
#include <pointcloudcore/core/point_cloud.hpp>

namespace pointcloudcore {

class PointTransformerBlock {
public:
    PointTransformerBlock(size_t in_dim, size_t out_dim, size_t k_neighbors = 16)
        : in_dim_(in_dim), out_dim_(out_dim), k_neighbors_(k_neighbors) {}

    // Forward pass computing 3D vector self-attention over local k-NN point graph
    Tensor forward(const PointCloud& cloud, const Tensor& features) const;

private:
    size_t in_dim_;
    size_t out_dim_;
    size_t k_neighbors_;
};

} // namespace pointcloudcore

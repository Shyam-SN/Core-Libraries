#pragma once

#include <pointcloudcore/core/tensor.hpp>
#include <pointcloudcore/core/point_cloud.hpp>

namespace pointcloudcore {

class PointNetSetAbstraction {
public:
    PointNetSetAbstraction(size_t npoint, double radius, size_t nsample, size_t in_channel, size_t out_channel)
        : npoint_(npoint), radius_(radius), nsample_(nsample), in_channel_(in_channel), out_channel_(out_channel) {}

    // Forward pass: sample points via FPS, group neighbors via ball query, apply MLP and max-pooling
    Tensor forward(const PointCloud& cloud, const Tensor& features) const;

private:
    size_t npoint_;
    double radius_;
    size_t nsample_;
    size_t in_channel_;
    size_t out_channel_;
};

class PointNetFeaturePropagation {
public:
    PointNetFeaturePropagation(size_t in_channel, size_t out_channel)
        : in_channel_(in_channel), out_channel_(out_channel) {}

    // Forward pass: interpolate features from subsampled points to dense points via inverse distance weighting
    Tensor forward(const PointCloud& dense_cloud, const PointCloud& sparse_cloud, const Tensor& sparse_features) const;

private:
    size_t in_channel_;
    size_t out_channel_;
};

} // namespace pointcloudcore

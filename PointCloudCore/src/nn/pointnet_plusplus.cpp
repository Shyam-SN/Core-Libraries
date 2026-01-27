#include <pointcloudcore/nn/pointnet_plusplus.hpp>
#include <pointcloudcore/ops/sampling.hpp>
#include <pointcloudcore/ops/neighborhood.hpp>

namespace pointcloudcore {

Tensor PointNetSetAbstraction::forward(const PointCloud& cloud, const Tensor& features) const {
    (void)features;
    (void)radius_;
    (void)nsample_;
    (void)in_channel_;

    PointCloud sampled_cloud = Sampling::farthest_point_sampling(cloud, npoint_);
    (void)sampled_cloud;

    Tensor out_features({1, npoint_, out_channel_}, 0.0);
    return out_features;
}

Tensor PointNetFeaturePropagation::forward(const PointCloud& dense_cloud, const PointCloud& sparse_cloud, const Tensor& sparse_features) const {
    (void)sparse_cloud;
    (void)sparse_features;
    (void)in_channel_;

    Tensor interpolated({1, dense_cloud.size(), out_channel_}, 0.0);
    return interpolated;
}

} // namespace pointcloudcore

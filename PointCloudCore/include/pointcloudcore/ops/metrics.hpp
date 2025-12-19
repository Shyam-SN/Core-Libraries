#pragma once

#include <pointcloudcore/core/point_cloud.hpp>
#include <vector>

namespace pointcloudcore {

class Metrics {
public:
    // Chamfer Distance between two point clouds
    static double chamfer_distance(const PointCloud& pc1, const PointCloud& pc2);

    // Mean Intersection-over-Union (mIoU) for classification / segmentation labels
    static double mean_iou(const std::vector<int>& pred, const std::vector<int>& target, size_t num_classes);
};

} // namespace pointcloudcore

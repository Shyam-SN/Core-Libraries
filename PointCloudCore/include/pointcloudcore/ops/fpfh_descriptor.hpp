#pragma once

#include <pointcloudcore/core/point_cloud.hpp>
#include <vector>

namespace pointcloudcore {

class FPFHDescriptor {
public:
    // Compute 33-dimensional Fast Point Feature Histogram (FPFH) features for each point
    static std::vector<std::vector<double>> compute_fpfh(const PointCloud& cloud, double search_radius = 0.25);
};

} // namespace pointcloudcore

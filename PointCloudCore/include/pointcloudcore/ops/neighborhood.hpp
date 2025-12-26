#pragma once

#include <pointcloudcore/core/point_cloud.hpp>
#include <vector>

namespace pointcloudcore {

class Neighborhood {
public:
    // k-Nearest Neighbors search for each query point
    static std::vector<std::vector<size_t>> knn_search(const PointCloud& cloud, const PointCloud& queries, size_t k);

    // Ball query / radius search for each query point
    static std::vector<std::vector<size_t>> radius_search(const PointCloud& cloud, const PointCloud& queries, double radius);
};

} // namespace pointcloudcore

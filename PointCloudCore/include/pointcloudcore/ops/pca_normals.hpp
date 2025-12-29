#pragma once

#include <pointcloudcore/core/point_cloud.hpp>

namespace pointcloudcore {

class PCANormals {
public:
    // Estimate surface normals using PCA local covariance Eigen-decomposition
    static void estimate_normals(PointCloud& cloud, size_t k_neighbors = 20);
};

} // namespace pointcloudcore

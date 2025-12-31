#pragma once

#include <pointcloudcore/core/point_cloud.hpp>
#include <vector>

namespace pointcloudcore {

class Sampling {
public:
    // Farthest Point Sampling (FPS) selecting N_samples points with maximal spatial distance
    static PointCloud farthest_point_sampling(const PointCloud& cloud, size_t num_samples);

    // Voxel grid spatial downsampling
    static PointCloud voxel_grid_sampling(const PointCloud& cloud, double voxel_size);
};

} // namespace pointcloudcore

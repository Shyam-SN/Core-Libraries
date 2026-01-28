#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/pointcloud/point_cloud.hpp>
#include <vector>
#include <unordered_map>

namespace geometrycore {

struct Voxel {
    Vector3 grid_index;
    Vector3 center;
    size_t count{0};
    Vector3 color{0.0, 0.0, 0.0};
};

class VoxelGrid {
public:
    explicit VoxelGrid(double voxel_size = 0.05) : voxel_size_(voxel_size) {}

    [[nodiscard]] double voxel_size() const noexcept { return voxel_size_; }

    PointCloud downsample(const PointCloud& cloud) const;

private:
    double voxel_size_;
};

} // namespace geometrycore

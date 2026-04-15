#include <geometrycore/spatial/voxel_grid.hpp>
#include <cmath>
#include <map>

namespace geometrycore {

PointCloud VoxelGrid::downsample(const PointCloud& cloud) const {
    if (cloud.empty() || voxel_size_ <= 0.0) return cloud;

    struct VoxelKey {
        int x, y, z;
        bool operator<(const VoxelKey& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };

    struct VoxelCentroid {
        Vector3 sum_pt{0.0, 0.0, 0.0};
        size_t count{0};
    };

    std::map<VoxelKey, VoxelCentroid> voxel_map;

    for (const auto& p : cloud.points) {
        VoxelKey key{
            static_cast<int>(std::floor(p.x / voxel_size_)),
            static_cast<int>(std::floor(p.y / voxel_size_)),
            static_cast<int>(std::floor(p.z / voxel_size_))
        };
        voxel_map[key].sum_pt += p;
        voxel_map[key].count += 1;
    }

    PointCloud downsampled;
    downsampled.points.reserve(voxel_map.size());

    for (const auto& [key, centroid] : voxel_map) {
        downsampled.points.push_back(centroid.sum_pt / static_cast<double>(centroid.count));
    }

    return downsampled;
}

} // namespace geometrycore

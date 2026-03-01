#include <pointcloudcore/ops/sampling.hpp>
#include <limits>
#include <map>
#include <cmath>

namespace pointcloudcore {

PointCloud Sampling::farthest_point_sampling(const PointCloud& cloud, size_t num_samples) {
    if (cloud.empty() || num_samples == 0) return PointCloud();
    if (num_samples >= cloud.size()) return cloud;

    size_t N = cloud.size();
    std::vector<size_t> sampled_indices;
    sampled_indices.reserve(num_samples);

    std::vector<double> min_dists(N, std::numeric_limits<double>::max());

    // Start with first point
    size_t farthest_idx = 0;
    sampled_indices.push_back(farthest_idx);

    for (size_t iter = 1; iter < num_samples; ++iter) {
        const Vector3d& last_pt = cloud.points[farthest_idx];
        double max_dist = -1.0;
        size_t next_farthest = 0;

        for (size_t i = 0; i < N; ++i) {
            double dist_sq = (cloud.points[i] - last_pt).squared_norm();
            if (dist_sq < min_dists[i]) {
                min_dists[i] = dist_sq;
            }
            if (min_dists[i] > max_dist) {
                max_dist = min_dists[i];
                next_farthest = i;
            }
        }

        farthest_idx = next_farthest;
        sampled_indices.push_back(farthest_idx);
    }

    PointCloud sampled;
    sampled.points.reserve(num_samples);
    if (!cloud.normals.empty()) sampled.normals.reserve(num_samples);
    if (!cloud.colors.empty()) sampled.colors.reserve(num_samples);

    for (size_t idx : sampled_indices) {
        sampled.points.push_back(cloud.points[idx]);
        if (!cloud.normals.empty()) sampled.normals.push_back(cloud.normals[idx]);
        if (!cloud.colors.empty()) sampled.colors.push_back(cloud.colors[idx]);
    }

    return sampled;
}

PointCloud Sampling::voxel_grid_sampling(const PointCloud& cloud, double voxel_size) {
    if (cloud.empty() || voxel_size <= 0.0) return cloud;

    struct VoxelKey {
        int x, y, z;
        bool operator<(const VoxelKey& o) const {
            if (x != o.x) return x < o.x;
            if (y != o.y) return y < o.y;
            return z < o.z;
        }
    };

    struct VoxelData {
        Vector3d sum_p{0.0, 0.0, 0.0};
        Vector3d sum_n{0.0, 0.0, 0.0};
        size_t count{0};
    };

    std::map<VoxelKey, VoxelData> grid;
    bool has_normals = !cloud.normals.empty();

    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& p = cloud.points[i];
        VoxelKey key{
            static_cast<int>(std::floor(p.x / voxel_size)),
            static_cast<int>(std::floor(p.y / voxel_size)),
            static_cast<int>(std::floor(p.z / voxel_size))
        };
        grid[key].sum_p += p;
        if (has_normals) grid[key].sum_n += cloud.normals[i];
        grid[key].count += 1;
    }

    PointCloud downsampled;
    downsampled.points.reserve(grid.size());
    if (has_normals) downsampled.normals.reserve(grid.size());

    for (const auto& [key, data] : grid) {
        downsampled.points.push_back(data.sum_p / static_cast<double>(data.count));
        if (has_normals) downsampled.normals.push_back((data.sum_n / static_cast<double>(data.count)).normalized());
    }

    return downsampled;
}

} // namespace pointcloudcore

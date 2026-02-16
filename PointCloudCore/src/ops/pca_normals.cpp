#include <pointcloudcore/ops/pca_normals.hpp>
#include <pointcloudcore/ops/neighborhood.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace pointcloudcore {

void PCANormals::estimate_normals(PointCloud& cloud, size_t k_neighbors) {
    if (cloud.empty()) return;
    size_t N = cloud.size();
    cloud.normals.resize(N);

    auto neighbor_indices = Neighborhood::knn_search(cloud, cloud, k_neighbors);

    for (size_t i = 0; i < N; ++i) {
        const auto& neighbors = neighbor_indices[i];
        size_t k = neighbors.size();
        if (k < 3) {
            cloud.normals[i] = {0.0, 0.0, 1.0};
            continue;
        }

        // Compute centroid
        Vector3d centroid{0.0, 0.0, 0.0};
        for (size_t idx : neighbors) {
            centroid += cloud.points[idx];
        }
        centroid /= static_cast<double>(k);

        // Compute 3x3 covariance matrix C
        double c00 = 0.0, c01 = 0.0, c02 = 0.0;
        double c11 = 0.0, c12 = 0.0, c22 = 0.0;

        for (size_t idx : neighbors) {
            Vector3d dp = cloud.points[idx] - centroid;
            c00 += dp.x * dp.x; c01 += dp.x * dp.y; c02 += dp.x * dp.z;
            c11 += dp.y * dp.y; c12 += dp.y * dp.z;
            c22 += dp.z * dp.z;
        }
        c00 /= k; c01 /= k; c02 /= k;
        c11 /= k; c12 /= k; c22 /= k;

        // Power iteration / Jacobi rotation to estimate eigenvector corresponding to smallest eigenvalue
        Vector3d normal{0.0, 0.0, 1.0};
        Vector3d v = (c00 <= c11 && c00 <= c22) ? Vector3d{1.0, 0.0, 0.0} :
                     (c11 <= c22 ? Vector3d{0.0, 1.0, 0.0} : Vector3d{0.0, 0.0, 1.0});

        for (int iter = 0; iter < 10; ++iter) {
            // Inverse power iteration step
            Vector3d w{
                c00 * v.x + c01 * v.y + c02 * v.z,
                c01 * v.x + c11 * v.y + c12 * v.z,
                c02 * v.x + c12 * v.y + c22 * v.z
            };
            v = v * 1.5 - w * 0.5;
            if (v.norm() > 1e-6) v.normalize();
        }

        normal = v.normalized();

        // Orient normal towards viewpoint vector (0, 0, 10)
        Vector3d view_dir = Vector3d{0.0, 0.0, 10.0} - cloud.points[i];
        if (normal.dot(view_dir) < 0.0) {
            normal = -normal;
        }

        cloud.normals[i] = normal;
    }
}

} // namespace pointcloudcore

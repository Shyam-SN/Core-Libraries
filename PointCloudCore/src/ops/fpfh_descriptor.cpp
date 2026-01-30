#include <pointcloudcore/ops/fpfh_descriptor.hpp>
#include <pointcloudcore/ops/neighborhood.hpp>
#include <cmath>

namespace pointcloudcore {

std::vector<std::vector<double>> FPFHDescriptor::compute_fpfh(const PointCloud& cloud, double search_radius) {
    size_t N = cloud.size();
    std::vector<std::vector<double>> fpfh_features(N, std::vector<double>(33, 0.0));
    if (N == 0) return fpfh_features;

    // Find neighbors for SPFH
    auto neighbors_list = Neighborhood::radius_search(cloud, cloud, search_radius);

    // Compute SPFH for each point
    std::vector<std::vector<double>> spfh_features(N, std::vector<double>(33, 0.0));

    for (size_t i = 0; i < N; ++i) {
        const auto& p_i = cloud.points[i];
        Vector3d n_i = cloud.normals.empty() ? Vector3d{0.0, 0.0, 1.0} : cloud.normals[i];
        const auto& neighbors = neighbors_list[i];

        if (neighbors.size() <= 1) continue;

        for (size_t j_idx : neighbors) {
            if (j_idx == i) continue;

            const auto& p_j = cloud.points[j_idx];
            Vector3d n_j = cloud.normals.empty() ? Vector3d{0.0, 0.0, 1.0} : cloud.normals[j_idx];

            Vector3d dp = p_j - p_i;
            double d = dp.norm();
            if (d < 1e-8) continue;
            dp /= d;

            // Darboux frame u, v, w
            Vector3d u = n_i;
            Vector3d v = dp.cross(u);
            if (v.norm() < 1e-8) continue;
            v.normalize();
            Vector3d w = u.cross(v);

            // Angles alpha, phi, theta
            double alpha = v.dot(n_j);
            double phi = u.dot(dp);
            double theta = std::atan2(w.dot(n_j), u.dot(n_j));

            // Binning (11 bins each)
            int b_alpha = std::clamp(static_cast<int>(std::floor((alpha + 1.0) * 0.5 * 11.0)), 0, 10);
            int b_phi   = std::clamp(static_cast<int>(std::floor((phi + 1.0) * 0.5 * 11.0)), 0, 10);
            int b_theta = std::clamp(static_cast<int>(std::floor((theta + M_PI) / (2.0 * M_PI) * 11.0)), 0, 10);

            spfh_features[i][b_alpha] += 1.0;
            spfh_features[i][11 + b_phi] += 1.0;
            spfh_features[i][22 + b_theta] += 1.0;
        }

        // Normalize SPFH
        double num_pairs = static_cast<double>(neighbors.size() - 1);
        if (num_pairs > 0.0) {
            for (int b = 0; b < 33; ++b) spfh_features[i][b] /= num_pairs;
        }
    }

    // Compute FPFH by weighting SPFH across neighbors
    for (size_t i = 0; i < N; ++i) {
        const auto& p_i = cloud.points[i];
        const auto& neighbors = neighbors_list[i];

        for (int b = 0; b < 33; ++b) {
            fpfh_features[i][b] = spfh_features[i][b];
        }

        double sum_weight = 0.0;
        for (size_t j_idx : neighbors) {
            if (j_idx == i) continue;
            double dist = (cloud.points[j_idx] - p_i).norm();
            if (dist < 1e-8) continue;

            double w = 1.0 / dist;
            for (int b = 0; b < 33; ++b) {
                fpfh_features[i][b] += w * spfh_features[j_idx][b];
            }
            sum_weight += w;
        }

        if (sum_weight > 0.0) {
            for (int b = 0; b < 33; ++b) fpfh_features[i][b] /= (1.0 + sum_weight);
        }
    }

    return fpfh_features;
}

} // namespace pointcloudcore

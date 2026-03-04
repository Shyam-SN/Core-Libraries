#include <pointcloudcore/registration/icp.hpp>
#include <pointcloudcore/ops/neighborhood.hpp>
#include <cmath>

namespace pointcloudcore {

RegistrationResult ICPRegistration::align(const PointCloud& source, const PointCloud& target,
                                           size_t max_iterations, double max_distance) {
    RegistrationResult res;
    if (source.empty() || target.empty()) return res;

    PointCloud src_transformed = source;
    double max_dist_sq = max_distance * max_distance;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        auto neighbor_indices = Neighborhood::knn_search(target, src_transformed, 1);

        std::vector<Vector3d> src_corr;
        std::vector<Vector3d> tgt_corr;
        double sum_sq_err = 0.0;

        for (size_t i = 0; i < src_transformed.size(); ++i) {
            size_t tgt_idx = neighbor_indices[i][0];
            double dist_sq = (src_transformed.points[i] - target.points[tgt_idx]).squared_norm();
            if (dist_sq <= max_dist_sq) {
                src_corr.push_back(src_transformed.points[i]);
                tgt_corr.push_back(target.points[tgt_idx]);
                sum_sq_err += dist_sq;
            }
        }

        size_t n_corr = src_corr.size();
        if (n_corr < 3) break;

        // Centroids
        Vector3d mu_src{0.0, 0.0, 0.0}, mu_tgt{0.0, 0.0, 0.0};
        for (size_t i = 0; i < n_corr; ++i) {
            mu_src += src_corr[i];
            mu_tgt += tgt_corr[i];
        }
        mu_src /= static_cast<double>(n_corr);
        mu_tgt /= static_cast<double>(n_corr);

        // Simple step translation
        Vector3d t_step = mu_tgt - mu_src;
        for (auto& p : src_transformed.points) {
            p += t_step;
        }

        res.t += t_step;
        res.inlier_rmse = std::sqrt(sum_sq_err / static_cast<double>(n_corr));
        res.fitness = static_cast<double>(n_corr) / static_cast<double>(source.size());
        res.converged = true;
    }

    return res;
}

} // namespace pointcloudcore

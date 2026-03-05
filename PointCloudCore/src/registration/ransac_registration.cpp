#include <pointcloudcore/registration/ransac_registration.hpp>
#include <random>
#include <cmath>
#include <algorithm>

namespace pointcloudcore {

RegistrationResult RANSACRegistration::register_fpfh(const PointCloud& source, const PointCloud& target,
                                                     const std::vector<std::vector<double>>& fpfh_src,
                                                     const std::vector<std::vector<double>>& fpfh_tgt,
                                                     size_t max_iterations, double distance_threshold) {
    RegistrationResult best_res;
    if (source.empty() || target.empty() || fpfh_src.empty() || fpfh_tgt.empty()) return best_res;

    // Build correspondences based on FPFH feature distance
    std::vector<std::pair<size_t, size_t>> correspondences;
    for (size_t i = 0; i < source.size(); ++i) {
        double min_feature_dist = 1e9;
        size_t best_j = 0;

        for (size_t j = 0; j < target.size(); ++j) {
            double feat_dist = 0.0;
            for (size_t b = 0; b < 33; ++b) {
                double diff = fpfh_src[i][b] - fpfh_tgt[j][b];
                feat_dist += diff * diff;
            }
            if (feat_dist < min_feature_dist) {
                min_feature_dist = feat_dist;
                best_j = j;
            }
        }
        correspondences.push_back({i, best_j});
    }

    if (correspondences.size() < 3) return best_res;

    std::mt19937 rng(42);
    size_t best_inliers = 0;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Sample 3 random correspondences
        std::uniform_int_distribution<size_t> dist(0, correspondences.size() - 1);
        size_t idx0 = dist(rng);
        size_t idx1 = dist(rng);
        size_t idx2 = dist(rng);

        const auto& p0_src = source.points[correspondences[idx0].first];
        const auto& p1_src = source.points[correspondences[idx1].first];
        const auto& p2_src = source.points[correspondences[idx2].first];

        const auto& p0_tgt = target.points[correspondences[idx0].second];
        const auto& p1_tgt = target.points[correspondences[idx1].second];
        const auto& p2_tgt = target.points[correspondences[idx2].second];

        Vector3d mu_src = (p0_src + p1_src + p2_src) / 3.0;
        Vector3d mu_tgt = (p0_tgt + p1_tgt + p2_tgt) / 3.0;

        Vector3d t_cand = mu_tgt - mu_src;

        // Evaluate inliers
        size_t inlier_count = 0;
        double dist_thresh_sq = distance_threshold * distance_threshold;

        for (const auto& [src_idx, tgt_idx] : correspondences) {
            Vector3d p_trans = source.points[src_idx] + t_cand;
            if ((p_trans - target.points[tgt_idx]).squared_norm() <= dist_thresh_sq) {
                inlier_count += 1;
            }
        }

        if (inlier_count > best_inliers) {
            best_inliers = inlier_count;
            best_res.t = t_cand;
            best_res.fitness = static_cast<double>(inlier_count) / static_cast<double>(source.size());
            best_res.converged = true;
        }
    }

    return best_res;
}

} // namespace pointcloudcore

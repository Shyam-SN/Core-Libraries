#include <pointcloudcore/ops/neighborhood.hpp>
#include <algorithm>

namespace pointcloudcore {

std::vector<std::vector<size_t>> Neighborhood::knn_search(const PointCloud& cloud, const PointCloud& queries, size_t k) {
    size_t Q = queries.size();
    size_t N = cloud.size();
    std::vector<std::vector<size_t>> results(Q);

    if (N == 0 || k == 0) return results;

    for (size_t q = 0; q < Q; ++q) {
        const auto& q_pt = queries.points[q];
        std::vector<std::pair<double, size_t>> dist_idx;
        dist_idx.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            double dist_sq = (cloud.points[i] - q_pt).squared_norm();
            dist_idx.push_back({dist_sq, i});
        }

        size_t actual_k = std::min(k, N);
        std::partial_sort(dist_idx.begin(), dist_idx.begin() + actual_k, dist_idx.end());

        results[q].reserve(actual_k);
        for (size_t i = 0; i < actual_k; ++i) {
            results[q].push_back(dist_idx[i].second);
        }
    }

    return results;
}

std::vector<std::vector<size_t>> Neighborhood::radius_search(const PointCloud& cloud, const PointCloud& queries, double radius) {
    size_t Q = queries.size();
    size_t N = cloud.size();
    std::vector<std::vector<size_t>> results(Q);

    double r2 = radius * radius;

    for (size_t q = 0; q < Q; ++q) {
        const auto& q_pt = queries.points[q];
        for (size_t i = 0; i < N; ++i) {
            if ((cloud.points[i] - q_pt).squared_norm() <= r2) {
                results[q].push_back(i);
            }
        }
    }

    return results;
}

} // namespace pointcloudcore

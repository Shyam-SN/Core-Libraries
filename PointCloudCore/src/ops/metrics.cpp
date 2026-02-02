#include <pointcloudcore/ops/metrics.hpp>
#include <limits>
#include <algorithm>

namespace pointcloudcore {

double Metrics::chamfer_distance(const PointCloud& pc1, const PointCloud& pc2) {
    if (pc1.empty() || pc2.empty()) return 0.0;

    double sum1 = 0.0;
    for (const auto& p1 : pc1.points) {
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto& p2 : pc2.points) {
            double dist_sq = (p1 - p2).squared_norm();
            if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
        }
        sum1 += min_dist_sq;
    }
    sum1 /= static_cast<double>(pc1.size());

    double sum2 = 0.0;
    for (const auto& p2 : pc2.points) {
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto& p1 : pc1.points) {
            double dist_sq = (p2 - p1).squared_norm();
            if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
        }
        sum2 += min_dist_sq;
    }
    sum2 /= static_cast<double>(pc2.size());

    return sum1 + sum2;
}

double Metrics::mean_iou(const std::vector<int>& pred, const std::vector<int>& target, size_t num_classes) {
    if (pred.size() != target.size() || num_classes == 0) return 0.0;

    std::vector<size_t> intersection(num_classes, 0);
    std::vector<size_t> union_set(num_classes, 0);

    for (size_t i = 0; i < pred.size(); ++i) {
        int p = pred[i];
        int t = target[i];

        if (p >= 0 && static_cast<size_t>(p) < num_classes && t >= 0 && static_cast<size_t>(t) < num_classes) {
            if (p == t) {
                intersection[p] += 1;
            }
            union_set[p] += 1;
            if (p != t) {
                union_set[t] += 1;
            }
        }
    }

    double total_iou = 0.0;
    size_t valid_classes = 0;

    for (size_t c = 0; c < num_classes; ++c) {
        if (union_set[c] > 0) {
            total_iou += static_cast<double>(intersection[c]) / static_cast<double>(union_set[c]);
            valid_classes += 1;
        }
    }

    return valid_classes > 0 ? total_iou / static_cast<double>(valid_classes) : 0.0;
}

} // namespace pointcloudcore

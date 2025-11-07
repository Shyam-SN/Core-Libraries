#pragma once

#include <visioncore/algorithms/homography.hpp>
#include <visioncore/core/point.hpp>
#include <vector>
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Fits a 3x3 homography matrix robustly using the RANSAC algorithm.
 * 
 * @param src_pts Vector of source 2D points.
 * @param dst_pts Vector of corresponding destination 2D points.
 * @param inliers Output mask (1 for inliers, 0 for outliers).
 * @param threshold Max error distance in pixels to count a point as an inlier.
 * @param max_iters Maximum iterations.
 * @return math::Matrix Best estimated 3x3 homography matrix.
 */
inline math::Matrix find_homography_ransac(const std::vector<Point2D>& src_pts,
                                           const std::vector<Point2D>& dst_pts,
                                           std::vector<uint8_t>& inliers,
                                           double threshold = 3.0,
                                           int max_iters = 1000) {
    if (src_pts.size() < 4 || dst_pts.size() < 4) {
        throw std::invalid_argument("At least 4 points are required.");
    }
    if (src_pts.size() != dst_pts.size()) {
        throw std::invalid_argument("Number of source and destination points must match.");
    }

    const size_t N = src_pts.size();
    inliers.assign(N, 0);

    std::mt19937 rng(1337); // Seeded for deterministic validation
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    size_t best_inliers_count = 0;
    math::Matrix best_H = math::Matrix::identity(3);
    std::vector<uint8_t> best_inliers(N, 0);

    const double thresh_sq = threshold * threshold;

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. Pick 4 unique point indices randomly
        std::vector<size_t> idx;
        while (idx.size() < 4) {
            size_t rand_idx = dist(rng);
            if (std::find(idx.begin(), idx.end(), rand_idx) == idx.end()) {
                idx.push_back(rand_idx);
            }
        }

        std::vector<Point2D> sample_src = { src_pts[idx[0]], src_pts[idx[1]], src_pts[idx[2]], src_pts[idx[3]] };
        std::vector<Point2D> sample_dst = { dst_pts[idx[0]], dst_pts[idx[1]], dst_pts[idx[2]], dst_pts[idx[3]] };

        // 2. Solve candidate homography
        math::Matrix H;
        try {
            H = find_homography(sample_src, sample_dst);
        } catch (...) {
            continue; // Singular configuration, skip this iteration
        }

        // 3. Score candidate homography on all points
        size_t current_inliers_count = 0;
        std::vector<uint8_t> current_inliers(N, 0);

        for (size_t i = 0; i < N; ++i) {
            double w = H(2, 0) * src_pts[i].x + H(2, 1) * src_pts[i].y + H(2, 2);
            if (std::abs(w) < 1e-10) continue;

            double px = (H(0, 0) * src_pts[i].x + H(0, 1) * src_pts[i].y + H(0, 2)) / w;
            double py = (H(1, 0) * src_pts[i].x + H(1, 1) * src_pts[i].y + H(1, 2)) / w;

            double dx = dst_pts[i].x - px;
            double dy = dst_pts[i].y - py;
            double err_sq = dx * dx + dy * dy;

            if (err_sq < thresh_sq) {
                current_inliers[i] = 1;
                current_inliers_count++;
            }
        }

        // 4. Update if we found a better model
        if (current_inliers_count > best_inliers_count) {
            best_inliers_count = current_inliers_count;
            best_H = H;
            best_inliers = current_inliers;
        }

        // Break early if we've successfully mapped almost all points
        if (best_inliers_count >= N * 0.98) {
            break;
        }
    }

    // 5. Final refinement using ALL best inliers
    std::vector<Point2D> refine_src;
    std::vector<Point2D> refine_dst;
    for (size_t i = 0; i < N; ++i) {
        if (best_inliers[i]) {
            refine_src.push_back(src_pts[i]);
            refine_dst.push_back(dst_pts[i]);
        }
    }

    if (refine_src.size() >= 4) {
        try {
            math::Matrix refined_H = find_homography(refine_src, refine_dst);

            // Re-evaluate inliers with refined homography matrix
            size_t refined_inliers_count = 0;
            std::vector<uint8_t> refined_inliers(N, 0);

            for (size_t i = 0; i < N; ++i) {
                double w = refined_H(2, 0) * src_pts[i].x + refined_H(2, 1) * src_pts[i].y + refined_H(2, 2);
                if (std::abs(w) < 1e-10) continue;

                double px = (refined_H(0, 0) * src_pts[i].x + refined_H(0, 1) * src_pts[i].y + refined_H(0, 2)) / w;
                double py = (refined_H(1, 0) * src_pts[i].x + refined_H(1, 1) * src_pts[i].y + refined_H(1, 2)) / w;

                double dx = dst_pts[i].x - px;
                double dy = dst_pts[i].y - py;
                double err_sq = dx * dx + dy * dy;

                if (err_sq < thresh_sq) {
                    refined_inliers[i] = 1;
                    refined_inliers_count++;
                }
            }

            if (refined_inliers_count >= best_inliers_count) {
                best_H = refined_H;
                best_inliers = refined_inliers;
            }
        } catch (...) {
            // If refinement fails due to noise, keep the best iteration matrix
        }
    }

    inliers = best_inliers;
    return best_H;
}

} // namespace visioncore

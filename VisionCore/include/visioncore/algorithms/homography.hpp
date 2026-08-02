#pragma once

#include <visioncore/math/linear_algebra.hpp>
#include <visioncore/core/point.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Computes the 3x3 homography matrix H mapping src_pts to dst_pts.
 * Uses the normalized Direct Linear Transform (DLT) algorithm.
 * 
 * @param src_pts Vector of at least 4 source points.
 * @param dst_pts Vector of at least 4 destination points.
 * @return math::Matrix 3x3 Homography matrix.
 */
inline math::Matrix find_homography(const std::vector<Point2D>& src_pts,
                                    const std::vector<Point2D>& dst_pts) {
    if (src_pts.size() < 4 || dst_pts.size() < 4) {
        throw std::invalid_argument("At least 4 point correspondences are required.");
    }
    if (src_pts.size() != dst_pts.size()) {
        throw std::invalid_argument("Number of source and destination points must match.");
    }

    const size_t N = src_pts.size();

    // 1. Compute normalization centroids and average distance
    double src_mx = 0.0, src_my = 0.0;
    double dst_mx = 0.0, dst_my = 0.0;
    for (size_t i = 0; i < N; ++i) {
        src_mx += src_pts[i].x;
        src_my += src_pts[i].y;
        dst_mx += dst_pts[i].x;
        dst_my += dst_pts[i].y;
    }
    src_mx /= N; src_my /= N;
    dst_mx /= N; dst_my /= N;

    double src_dist = 0.0;
    double dst_dist = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double dx = src_pts[i].x - src_mx;
        double dy = src_pts[i].y - src_my;
        src_dist += std::sqrt(dx*dx + dy*dy);

        double dx_d = dst_pts[i].x - dst_mx;
        double dy_d = dst_pts[i].y - dst_my;
        dst_dist += std::sqrt(dx_d*dx_d + dy_d*dy_d);
    }
    src_dist /= N;
    dst_dist /= N;

    double src_scale = (src_dist > 1e-8) ? (std::sqrt(2.0) / src_dist) : 1.0;
    double dst_scale = (dst_dist > 1e-8) ? (std::sqrt(2.0) / dst_dist) : 1.0;

    // Normalization transformation matrix T_src
    math::Matrix T_src(3, 3, 0.0);
    T_src(0, 0) = src_scale; T_src(0, 2) = -src_scale * src_mx;
    T_src(1, 1) = src_scale; T_src(1, 2) = -src_scale * src_my;
    T_src(2, 2) = 1.0;

    // Inverse normalization transformation matrix for destination T_dst_inv
    math::Matrix T_dst_inv(3, 3, 0.0);
    T_dst_inv(0, 0) = 1.0 / dst_scale; T_dst_inv(0, 2) = dst_mx;
    T_dst_inv(1, 1) = 1.0 / dst_scale; T_dst_inv(1, 2) = dst_my;
    T_dst_inv(2, 2) = 1.0;

    // 2. Formulate 2N x 9 matrix A for Ah = 0
    math::Matrix A(2 * N, 9, 0.0);
    for (size_t i = 0; i < N; ++i) {
        // Shift and scale
        double xi = src_scale * (src_pts[i].x - src_mx);
        double yi = src_scale * (src_pts[i].y - src_my);
        double xpi = dst_scale * (dst_pts[i].x - dst_mx);
        double ypi = dst_scale * (dst_pts[i].y - dst_my);

        // Row 1
        A(2 * i, 3) = -xi;
        A(2 * i, 4) = -yi;
        A(2 * i, 5) = -1.0;
        A(2 * i, 6) = ypi * xi;
        A(2 * i, 7) = ypi * yi;
        A(2 * i, 8) = ypi;

        // Row 2
        A(2 * i + 1, 0) = xi;
        A(2 * i + 1, 1) = yi;
        A(2 * i + 1, 2) = 1.0;
        A(2 * i + 1, 6) = -xpi * xi;
        A(2 * i + 1, 7) = -xpi * yi;
        A(2 * i + 1, 8) = -xpi;
    }

    // 3. Solve homogeneous system using our custom SVD
    std::vector<double> h;
    math::solve_homogeneous(A, h);

    // 4. Construct normalized H_tilde
    math::Matrix H_tilde(3, 3);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            H_tilde(r, c) = h[r * 3 + c];
        }
    }

    // 5. Denormalize: H = T_dst_inv * H_tilde * T_src
    math::Matrix H = T_dst_inv * H_tilde * T_src;

    // Normalize output so H(2,2) = 1.0
    double scale = H(2, 2);
    if (std::abs(scale) > 1e-10) {
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                H(r, c) /= scale;
            }
        }
    }

    return H;
}

} // namespace visioncore

#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/core/point.hpp>
#include <visioncore/math/linear_algebra.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Triangulates a set of 2D point correspondences from two cameras into 3D coordinates.
 * 
 * @param P1 3x4 projection matrix of the first camera.
 * @param P2 3x4 projection matrix of the second camera.
 * @param pts1 2D correspondences in the first camera.
 * @param pts2 2D correspondences in the second camera.
 * @param pts3d Output vector of triangulated 3D points.
 */
inline void triangulate_points(const math::Matrix& P1,
                               const math::Matrix& P2,
                               const std::vector<Point2D>& pts1,
                               const std::vector<Point2D>& pts2,
                               std::vector<Point3D>& pts3d) {
    if (P1.rows() != 3 || P1.cols() != 4 || P2.rows() != 3 || P2.cols() != 4) {
        throw std::invalid_argument("Projection matrices must be of size 3x4.");
    }
    if (pts1.size() != pts2.size()) {
        throw std::invalid_argument("Number of points in pts1 and pts2 must match.");
    }

    const size_t N = pts1.size();
    pts3d.resize(N);

    // Solve for each point correspondence
    for (size_t i = 0; i < N; ++i) {
        const double u1 = pts1[i].x;
        const double v1 = pts1[i].y;
        const double u2 = pts2[i].x;
        const double v2 = pts2[i].y;

        // A is a 4x4 coefficient matrix
        math::Matrix A(4, 4);
        for (size_t j = 0; j < 4; ++j) {
            A(0, j) = u1 * P1(2, j) - P1(0, j);
            A(1, j) = v1 * P1(2, j) - P1(1, j);
            A(2, j) = u2 * P2(2, j) - P2(0, j);
            A(3, j) = v2 * P2(2, j) - P2(1, j);
        }

        std::vector<double> X;
        math::solve_homogeneous(A, X);

        double w = X[3];
        if (std::abs(w) > 1e-12) {
            pts3d[i] = Point3D(X[0] / w, X[1] / w, X[2] / w);
        } else {
            pts3d[i] = Point3D(0.0, 0.0, 0.0);
        }
    }
}

/**
 * @brief Computes a stereo disparity map from rectified left and right images using Block Matching (SAD).
 * 
 * @param left Grayscale rectified left image view.
 * @param right Grayscale rectified right image view.
 * @param disparity Output disparity map (grayscale, 1 channel).
 * @param min_disparity Minimum search disparity.
 * @param num_disparities Disparity search range width.
 * @param win_size SAD matching window size (must be odd).
 */
inline void compute_disparity_sad(const ImageView<const uint8_t, 1>& left,
                                  const ImageView<const uint8_t, 1>& right,
                                  const ImageView<uint8_t, 1>& disparity,
                                  int min_disparity,
                                  int num_disparities,
                                  size_t win_size = 9) {
    if (left.width() != right.width() || left.height() != right.height() ||
        left.width() != disparity.width() || left.height() != disparity.height()) {
        throw std::invalid_argument("All images must have matching dimensions.");
    }
    if (win_size % 2 == 0) {
        throw std::invalid_argument("Window size must be odd.");
    }

    const int w = static_cast<int>(left.width());
    const int h = static_cast<int>(left.height());
    const int half_w = static_cast<int>(win_size / 2);

    // Initialize disparity map to zero
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            disparity(x, y) = 0;
        }
    }

    for (int y = half_w; y < h - half_w; ++y) {
        for (int x = half_w + min_disparity + num_disparities; x < w - half_w; ++x) {
            uint32_t min_sad = std::numeric_limits<uint32_t>::max();
            int best_d = min_disparity;

            for (int d = min_disparity; d < min_disparity + num_disparities; ++d) {
                if (x - d < half_w) continue;

                uint32_t sad = 0;
                for (int dy = -half_w; dy <= half_w; ++dy) {
                    for (int dx = -half_w; dx <= half_w; ++dx) {
                        int val_l = left(x + dx, y + dy);
                        int val_r = right(x - d + dx, y + dy);
                        sad += static_cast<uint32_t>(std::abs(val_l - val_r));
                    }
                }

                if (sad < min_sad) {
                    min_sad = sad;
                    best_d = d;
                }
            }

            disparity(x, y) = static_cast<uint8_t>(best_d);
        }
    }
}

} // namespace visioncore

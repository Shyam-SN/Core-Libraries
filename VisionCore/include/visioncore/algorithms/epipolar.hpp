#pragma once

#include <visioncore/math/linear_algebra.hpp>
#include <visioncore/core/point.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Computes the 3x3 Fundamental Matrix F mapping points in image 1 to epipolar lines in image 2.
 * Uses the normalized 8-point algorithm with rank-2 constraint enforcement.
 * 
 * @param pts1 Vector of at least 8 point correspondences in image 1.
 * @param pts2 Vector of at least 8 point correspondences in image 2.
 * @return math::Matrix 3x3 Fundamental matrix.
 */
inline math::Matrix find_fundamental(const std::vector<Point2D>& pts1,
                                     const std::vector<Point2D>& pts2) {
    if (pts1.size() < 8 || pts2.size() < 8) {
        throw std::invalid_argument("At least 8 point correspondences are required for the 8-point algorithm.");
    }
    if (pts1.size() != pts2.size()) {
        throw std::invalid_argument("Number of correspondences in both images must match.");
    }

    const size_t N = pts1.size();

    // 1. Compute normalization transforms T1 and T2
    double m1x = 0.0, m1y = 0.0;
    double m2x = 0.0, m2y = 0.0;
    for (size_t i = 0; i < N; ++i) {
        m1x += pts1[i].x;
        m1y += pts1[i].y;
        m2x += pts2[i].x;
        m2y += pts2[i].y;
    }
    m1x /= N; m1y /= N;
    m2x /= N; m2y /= N;

    double d1 = 0.0, d2 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double dx1 = pts1[i].x - m1x;
        double dy1 = pts1[i].y - m1y;
        d1 += std::sqrt(dx1*dx1 + dy1*dy1);

        double dx2 = pts2[i].x - m2x;
        double dy2 = pts2[i].y - m2y;
        d2 += std::sqrt(dx2*dx2 + dy2*dy2);
    }
    d1 /= N;
    d2 /= N;

    double s1 = (d1 > 1e-8) ? (std::sqrt(2.0) / d1) : 1.0;
    double s2 = (d2 > 1e-8) ? (std::sqrt(2.0) / d2) : 1.0;

    // Normalization matrices
    math::Matrix T1(3, 3, 0.0);
    T1(0, 0) = s1; T1(0, 2) = -s1 * m1x;
    T1(1, 1) = s1; T1(1, 2) = -s1 * m1y;
    T1(2, 2) = 1.0;

    math::Matrix T2(3, 3, 0.0);
    T2(0, 0) = s2; T2(0, 2) = -s2 * m2x;
    T2(1, 1) = s2; T2(1, 2) = -s2 * m2y;
    T2(2, 2) = 1.0;

    // 2. Formulate linear system A * f = 0
    math::Matrix A(N, 9, 0.0);
    for (size_t i = 0; i < N; ++i) {
        double x1 = s1 * (pts1[i].x - m1x);
        double y1 = s1 * (pts1[i].y - m1y);
        double x2 = s2 * (pts2[i].x - m2x);
        double y2 = s2 * (pts2[i].y - m2y);

        A(i, 0) = x2 * x1;
        A(i, 1) = x2 * y1;
        A(i, 2) = x2;
        A(i, 3) = y2 * x1;
        A(i, 4) = y2 * y1;
        A(i, 5) = y2;
        A(i, 6) = x1;
        A(i, 7) = y1;
        A(i, 8) = 1.0;
    }

    // 3. Solve homogeneous system using SVD
    std::vector<double> f;
    math::solve_homogeneous(A, f);

    // Construct normalized F_tilde
    math::Matrix F_tilde(3, 3);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            F_tilde(r, c) = f[r * 3 + c];
        }
    }

    // 4. Enforce rank-2 constraint: F_projected = U * diag(s1, s2, 0) * V^T
    math::Matrix F_projected;
    math::project_fundamental(F_tilde, F_projected);

    // 5. Denormalize: F = T2^T * F_projected * T1
    math::Matrix F = T2.transpose() * F_projected * T1;

    // Normalize output so Frobenius norm (sum of squared values) is 1.0
    double norm_f = 0.0;
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            norm_f += F(r, c) * F(r, c);
        }
    }
    norm_f = std::sqrt(norm_f);
    if (norm_f > 1e-10) {
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                F(r, c) /= norm_f;
            }
        }
    }

    return F;
}

/**
 * @brief Computes the Essential Matrix E from Fundamental Matrix F and calibration matrices K1, K2.
 * E = K2^T * F * K1
 */
inline math::Matrix compute_essential(const math::Matrix& F,
                                      const math::Matrix& K1,
                                      const math::Matrix& K2) {
    if (F.rows() != 3 || F.cols() != 3 ||
        K1.rows() != 3 || K1.cols() != 3 ||
        K2.rows() != 3 || K2.cols() != 3) {
        throw std::invalid_argument("Matrices must be 3x3.");
    }
    return K2.transpose() * F * K1;
}

/**
 * @brief Enforces the specific structural constraints of an Essential Matrix.
 * Singular values must be (s, s, 0). Projects E to this manifold by averaging 
 * the first two singular values and zeroing out the third.
 */
inline void project_essential(const math::Matrix& E, math::Matrix& E_projected) {
    if (E.rows() != 3 || E.cols() != 3) {
        throw std::invalid_argument("Essential matrix must be 3x3.");
    }

    math::Matrix U;
    std::vector<double> S;
    math::Matrix V;
    math::svd(E, U, S, V);

    // Enforce singular values (s, s, 0)
    double mean_s = (S[0] + S[1]) / 2.0;
    S[0] = mean_s;
    S[1] = mean_s;
    S[2] = 0.0;

    math::Matrix diagS(3, 3, 0.0);
    diagS(0, 0) = S[0];
    diagS(1, 1) = S[1];
    diagS(2, 2) = S[2];

    E_projected = U * diagS * V.transpose();
}

} // namespace visioncore

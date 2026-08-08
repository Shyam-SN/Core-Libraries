#pragma once

#include <visioncore/math/linear_algebra.hpp>
#include <visioncore/core/point.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Computes camera pose (R, t) from 3D-2D correspondences using the DLT PnP method (n >= 6).
 * 
 * @param object_pts 3D points in the world coordinate system.
 * @param image_pts 2D coordinates projected in the image.
 * @param K 3x3 camera intrinsic calibration matrix.
 * @param R Output 3x3 camera rotation matrix.
 * @param t Output 3-element camera translation vector.
 */
inline void solve_pnp_dlt(const std::vector<Point3D>& object_pts,
                          const std::vector<Point2D>& image_pts,
                          const math::Matrix& K,
                          math::Matrix& R,
                          std::vector<double>& t) {
    if (object_pts.size() < 6 || image_pts.size() < 6) {
        throw std::invalid_argument("At least 6 correspondences are required for DLT PnP.");
    }
    if (object_pts.size() != image_pts.size()) {
        throw std::invalid_argument("Number of object and image points must match.");
    }
    if (K.rows() != 3 || K.cols() != 3) {
        throw std::invalid_argument("Camera intrinsics matrix K must be 3x3.");
    }

    const size_t N = object_pts.size();

    // 1. Formulate the 2N x 12 linear system A * p = 0
    math::Matrix A(2 * N, 12, 0.0);
    for (size_t i = 0; i < N; ++i) {
        double X = object_pts[i].x;
        double Y = object_pts[i].y;
        double Z = object_pts[i].z;
        double u = image_pts[i].x;
        double v = image_pts[i].y;

        // Row 1
        A(2 * i, 0) = X;   A(2 * i, 1) = Y;   A(2 * i, 2) = Z;   A(2 * i, 3) = 1.0;
        A(2 * i, 8) = -u * X; A(2 * i, 9) = -u * Y; A(2 * i, 10) = -u * Z; A(2 * i, 11) = -u;

        // Row 2
        A(2 * i + 1, 4) = X;   A(2 * i + 1, 5) = Y;   A(2 * i + 1, 6) = Z;   A(2 * i + 1, 7) = 1.0;
        A(2 * i + 1, 8) = -v * X; A(2 * i + 1, 9) = -v * Y; A(2 * i + 1, 10) = -v * Z; A(2 * i + 1, 11) = -v;
    }

    // 2. Solve Homogeneous system via SVD
    std::vector<double> p;
    math::solve_homogeneous(A, p);

    // Reconstruct projection matrix P
    math::Matrix P(3, 4);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            P(r, c) = p[r * 4 + c];
        }
    }

    // 3. Extract [R | t] = K^-1 * P
    // Compute analytical inverse of calibration matrix K
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    math::Matrix K_inv(3, 3, 0.0);
    K_inv(0, 0) = 1.0 / fx;
    K_inv(1, 1) = 1.0 / fy;
    K_inv(0, 2) = -cx / fx;
    K_inv(1, 2) = -cy / fy;
    K_inv(2, 2) = 1.0;

    math::Matrix Rt_raw = K_inv * P;

    // Extract raw rotation component R_raw
    math::Matrix R_raw(3, 3);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            R_raw(r, c) = Rt_raw(r, c);
        }
    }

    // Determine scale parameter using R_raw singular values
    math::Matrix U;
    std::vector<double> S;
    math::Matrix V;
    math::svd(R_raw, U, S, V);

    double scale = (S[0] + S[1] + S[2]) / 3.0;
    if (scale < 1e-12) {
        scale = 1.0;
    }

    // Enforce rotation matrix orthonormality: R = U * V^T
    math::Matrix R_ortho = U * V.transpose();

    // Check determinant to avoid reflections
    double det = R_ortho(0, 0) * (R_ortho(1, 1) * R_ortho(2, 2) - R_ortho(1, 2) * R_ortho(2, 1)) -
                 R_ortho(0, 1) * (R_ortho(1, 0) * R_ortho(2, 2) - R_ortho(1, 2) * R_ortho(2, 0)) +
                 R_ortho(0, 2) * (R_ortho(1, 0) * R_ortho(2, 1) - R_ortho(1, 1) * R_ortho(2, 0));

    if (det < 0.0) {
        // Negate columns/scale to fix reflection
        math::Matrix diagS = math::Matrix::identity(3);
        diagS(2, 2) = -1.0;
        R = U * diagS * V.transpose();
        scale = -scale;
    } else {
        R = R_ortho;
    }

    // Extract translation vector t, scaling it accordingly
    t.resize(3);
    t[0] = Rt_raw(0, 3) / scale;
    t[1] = Rt_raw(1, 3) / scale;
    t[2] = Rt_raw(2, 3) / scale;

    // Enforce positive depth: camera should look forward (z > 0 in camera coordinates)
    if (t[2] < 0.0) {
        // Invert rotation and translation sign
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                R(r, c) = -R(r, c);
            }
            t[r] = -t[r];
        }
    }
}

} // namespace visioncore

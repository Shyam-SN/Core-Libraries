#pragma once

#include <visioncore/math/linear_algebra.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Performs camera calibration using Zhang's method from a set of homographies.
 * Estimates the 3x3 camera intrinsic matrix K.
 * 
 * @param homographies A vector of at least 3 homography matrices, one per calibration pattern view.
 * @return math::Matrix 3x3 Camera intrinsic matrix K.
 */
inline math::Matrix calibrate_camera(const std::vector<math::Matrix>& homographies) {
    if (homographies.size() < 3) {
        throw std::invalid_argument("At least 3 homographies are required for camera calibration.");
    }

    for (const auto& H : homographies) {
        if (H.rows() != 3 || H.cols() != 3) {
            throw std::invalid_argument("Each homography matrix must be of size 3x3.");
        }
    }

    const size_t num_homographies = homographies.size();
    
    // 1. Construct the 2N x 6 coefficient matrix V
    // Equation: v_ij^T * b = 0
    math::Matrix V(2 * num_homographies, 6, 0.0);
    
    for (size_t k = 0; k < num_homographies; ++k) {
        const auto& H = homographies[k];
        
        // Helper to compute v_ij for a given homography matrix H
        auto get_v = [&](int i, int j) {
            std::vector<double> v(6);
            v[0] = H(0, i) * H(0, j);
            v[1] = H(0, i) * H(1, j) + H(1, i) * H(0, j);
            v[2] = H(1, i) * H(1, j);
            v[3] = H(2, i) * H(0, j) + H(0, i) * H(2, j);
            v[4] = H(2, i) * H(1, j) + H(1, i) * H(2, j);
            v[5] = H(2, i) * H(2, j);
            return v;
        };

        std::vector<double> v01 = get_v(0, 1);
        std::vector<double> v00 = get_v(0, 0);
        std::vector<double> v11 = get_v(1, 1);

        // Row 2k: v01^T * b = 0
        for (int c = 0; c < 6; ++c) {
            V(2 * k, c) = v01[c];
        }
        // Row 2k + 1: (v00 - v11)^T * b = 0
        for (int c = 0; c < 6; ++c) {
            V(2 * k + 1, c) = v00[c] - v11[c];
        }
    }

    // 2. Solve V * b = 0 using SVD to find the absolute conic matrix B
    std::vector<double> b;
    math::solve_homogeneous(V, b);

    double B11 = b[0];
    double B12 = b[1];
    double B22 = b[2];
    double B13 = b[3];
    double B23 = b[4];
    double B33 = b[5];

    // Ensure B11 is positive (since b is defined up to scale)
    if (B11 < 0.0) {
        B11 = -B11;
        B12 = -B12;
        B22 = -B22;
        B13 = -B13;
        B23 = -B23;
        B33 = -B33;
    }

    // 3. Extract camera intrinsic parameters from absolute conic matrix B.
    // We use the analytical matrix inversion of B to find C = B^-1 = K * K^T.
    // Then we extract parameters directly from the symmetric positive definite C.
    
    // Compute 3x3 determinant of B
    double detB = B11 * (B22 * B33 - B23 * B23)
                - B12 * (B12 * B33 - B13 * B23)
                + B13 * (B12 * B23 - B13 * B22);

    if (std::abs(detB) < 1e-12) {
        throw std::runtime_error("Camera calibration failed: degenerate conic matrix B.");
    }

    // Compute elements of C = B^-1 using cofactors
    double C11 = (B22 * B33 - B23 * B23) / detB;
    double C12 = (B13 * B23 - B12 * B33) / detB;
    double C13 = (B12 * B23 - B13 * B22) / detB;
    double C22 = (B11 * B33 - B13 * B13) / detB;
    double C23 = (B12 * B13 - B11 * B23) / detB;
    double C33 = (B11 * B22 - B12 * B12) / detB;

    if (std::abs(C33) < 1e-12) {
        throw std::runtime_error("Camera calibration failed: invalid normalization scaling.");
    }

    // Normalize C so that C33 = 1.0
    C11 /= C33;
    C12 /= C33;
    C13 /= C33;
    C22 /= C33;
    C23 /= C33;
    C33 = 1.0;

    // K * K^T = [ fx^2 + gamma^2 + u0^2,  gamma*fy + u0*v0,  u0 ]
    //           [ gamma*fy + u0*v0,       fy^2 + v0^2,       v0 ]
    //           [ u0,                     v0,                1  ]
    double u0 = C13;
    double v0 = C23;
    
    double fy_sq = C22 - v0 * v0;
    if (fy_sq <= 0.0) {
        throw std::runtime_error("Camera calibration failed: invalid focal length parameter Y.");
    }
    double fy = std::sqrt(fy_sq);

    double gamma = (C12 - u0 * v0) / fy;

    double fx_sq = C11 - gamma * gamma - u0 * u0;
    if (fx_sq <= 0.0) {
        throw std::runtime_error("Camera calibration failed: invalid focal length parameter X.");
    }
    double fx = std::sqrt(fx_sq);

    // Construct intrinsic matrix K
    math::Matrix K(3, 3, 0.0);
    K(0, 0) = fx;
    K(0, 1) = gamma;
    K(0, 2) = u0;
    K(1, 1) = fy;
    K(1, 2) = v0;
    K(2, 2) = 1.0;

    return K;
}

} // namespace visioncore

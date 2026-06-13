#pragma once

#include <visioncore/math/linear_algebra.hpp>
#include <visioncore/core/point.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Helper to convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
 */
inline math::Matrix euler_to_rotation(double rx, double ry, double rz) {
    double cx = std::cos(rx), sx = std::sin(rx);
    double cy = std::cos(ry), sy = std::sin(ry);
    double cz = std::cos(rz), sz = std::sin(rz);

    math::Matrix R(3, 3);
    R(0, 0) = cy * cz;
    R(0, 1) = -cy * sz;
    R(0, 2) = sy;

    R(1, 0) = cx * sz + sx * sy * cz;
    R(1, 1) = cx * cz - sx * sy * sz;
    R(1, 2) = -sx * cy;

    R(2, 0) = sx * sz - cx * sy * cz;
    R(2, 1) = sx * cz + cx * sy * sz;
    R(2, 2) = cx * cy;

    return R;
}

/**
 * @brief Projects a 3D point to a camera view.
 */
inline Point2D project_point(const Point3D& pt,
                             const math::Matrix& K,
                             const math::Matrix& R,
                             const std::vector<double>& t) {
    double xc = R(0, 0) * pt.x + R(0, 1) * pt.y + R(0, 2) * pt.z + t[0];
    double yc = R(1, 0) * pt.x + R(1, 1) * pt.y + R(1, 2) * pt.z + t[1];
    double zc = R(2, 0) * pt.x + R(2, 1) * pt.y + R(2, 2) * pt.z + t[2];

    if (std::abs(zc) < 1e-12) {
        zc = 1e-12;
    }

    return Point2D((K(0, 0) * xc + K(0, 2) * zc) / zc,
                   (K(1, 1) * yc + K(1, 2) * zc) / zc);
}

/**
 * @brief Solves a linear system A * x = b using Singular Value Decomposition.
 */
inline void solve_linear_system(const math::Matrix& A,
                                const std::vector<double>& b,
                                std::vector<double>& x) {
    math::Matrix U;
    std::vector<double> S;
    math::Matrix V;
    math::svd(A, U, S, V);

    const size_t N = A.cols();
    const size_t M = A.rows();
    x.assign(N, 0.0);

    // Compute Ut * b
    std::vector<double> Ut_b(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < M; ++j) {
            sum += U(j, i) * b[j];
        }
        Ut_b[i] = sum;
    }

    // Solve for x = V * S^-1 * Ut_b
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (S[j] > 1e-10) {
                sum += V(i, j) * (Ut_b[j] / S[j]);
            }
        }
        x[i] = sum;
    }
}

/**
 * @brief Computes residuals (reprojection errors) for all points in both cameras.
 */
inline void compute_residuals(const std::vector<double>& theta,
                              const std::vector<Point2D>& obs1,
                              const std::vector<Point2D>& obs2,
                              const math::Matrix& K,
                              std::vector<double>& r) {
    const size_t N = obs1.size();
    r.resize(4 * N);

    // Camera 1 is fixed at origin [I | 0]
    math::Matrix R1 = math::Matrix::identity(3);
    std::vector<double> t1 = {0.0, 0.0, 0.0};

    // Camera 2 pose parameters from theta
    double rx = theta[0];
    double ry = theta[1];
    double rz = theta[2];
    std::vector<double> t2 = {theta[3], theta[4], theta[5]};
    math::Matrix R2 = euler_to_rotation(rx, ry, rz);

    for (size_t i = 0; i < N; ++i) {
        // Point i coordinate parameters from theta
        Point3D pt(theta[6 + 3 * i], theta[6 + 3 * i + 1], theta[6 + 3 * i + 2]);

        // Reprojection in camera 1
        Point2D proj1 = project_point(pt, K, R1, t1);
        r[4 * i] = proj1.x - obs1[i].x;
        r[4 * i + 1] = proj1.y - obs1[i].y;

        // Reprojection in camera 2
        Point2D proj2 = project_point(pt, K, R2, t2);
        r[4 * i + 2] = proj2.x - obs2[i].x;
        r[4 * i + 3] = proj2.y - obs2[i].y;
    }
}

/**
 * @brief Optimizes camera 2 pose and 3D structure jointly using Levenberg-Marquardt.
 * 
 * @param K 3x3 camera intrinsic matrix.
 * @param obs1 2D observations in camera 1.
 * @param obs2 2D observations in camera 2.
 * @param R2_est Input/Output rotation matrix of camera 2.
 * @param t2_est Input/Output translation vector of camera 2.
 * @param pts3d Input/Output vector of 3D points.
 * @param max_iters Max Levenberg-Marquardt iterations.
 */
inline void bundle_adjustment(const math::Matrix& K,
                              const std::vector<Point2D>& obs1,
                              const std::vector<Point2D>& obs2,
                              math::Matrix& R2_est,
                              std::vector<double>& t2_est,
                              std::vector<Point3D>& pts3d,
                              int max_iters = 50) {
    if (obs1.size() != obs2.size() || obs1.size() != pts3d.size()) {
        throw std::invalid_argument("Input size mismatch in bundle adjustment.");
    }

    const size_t N = obs1.size();
    const size_t num_params = 6 + 3 * N;

    // 1. Pack parameters into theta vector: [rx, ry, rz, tx, ty, tz, X0, Y0, Z0, ...]
    // Estimate initial euler angles from R2_est (simple extraction assuming pitch != +-90)
    double sy = std::sqrt(R2_est(0, 0) * R2_est(0, 0) + R2_est(0, 1) * R2_est(0, 1));
    double rx = 0.0, ry = 0.0, rz = 0.0;
    if (sy > 1e-6) {
        rx = std::atan2(-R2_est(1, 2), R2_est(2, 2));
        ry = std::atan2(R2_est(0, 2), sy);
        rz = std::atan2(-R2_est(0, 1), R2_est(0, 0));
    } else {
        rx = std::atan2(R2_est(2, 1), R2_est(1, 1));
        ry = std::atan2(R2_est(0, 2), sy);
        rz = 0.0;
    }

    std::vector<double> theta(num_params);
    theta[0] = rx;
    theta[1] = ry;
    theta[2] = rz;
    theta[3] = t2_est[0];
    theta[4] = t2_est[1];
    theta[5] = t2_est[2];

    for (size_t i = 0; i < N; ++i) {
        theta[6 + 3 * i] = pts3d[i].x;
        theta[6 + 3 * i + 1] = pts3d[i].y;
        theta[6 + 3 * i + 2] = pts3d[i].z;
    }

    // 2. LM Loop
    double lambda = 1e-3;
    const double eps = 1e-6;

    std::vector<double> r;
    compute_residuals(theta, obs1, obs2, K, r);

    double err = 0.0;
    for (double val : r) {
        err += val * val;
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // Compute Numerical Jacobian of size 4N x (6+3N)
        math::Matrix J(4 * N, num_params, 0.0);
        std::vector<double> r_perturbed_pos;
        std::vector<double> r_perturbed_neg;

        for (size_t p = 0; p < num_params; ++p) {
            double orig_val = theta[p];

            // Positive perturbation
            theta[p] = orig_val + eps;
            compute_residuals(theta, obs1, obs2, K, r_perturbed_pos);

            // Negative perturbation
            theta[p] = orig_val - eps;
            compute_residuals(theta, obs1, obs2, K, r_perturbed_neg);

            theta[p] = orig_val; // Restore

            // Central difference approximation
            for (size_t row = 0; row < 4 * N; ++row) {
                J(row, p) = (r_perturbed_pos[row] - r_perturbed_neg[row]) / (2.0 * eps);
            }
        }

        // Formulate Normal equations: (J^T * J + lambda * I) * delta = -J^T * r
        math::Matrix Jt = J.transpose();
        math::Matrix H = Jt * J;

        // Apply Levenberg-Marquardt diagonal damping
        for (size_t p = 0; p < num_params; ++p) {
            H(p, p) += lambda;
        }

        // Construct right-hand side B = -J^T * r
        std::vector<double> B(num_params, 0.0);
        for (size_t row = 0; row < num_params; ++row) {
            double sum = 0.0;
            for (size_t k = 0; k < 4 * N; ++k) {
                sum += Jt(row, k) * r[k];
            }
            B[row] = -sum;
        }

        // Solve linear system
        std::vector<double> delta_theta;
        solve_linear_system(H, B, delta_theta);

        // Compute new candidate parameter vector
        std::vector<double> theta_new(num_params);
        for (size_t p = 0; p < num_params; ++p) {
            theta_new[p] = theta[p] + delta_theta[p];
        }

        // Evaluate new residuals
        std::vector<double> r_new;
        compute_residuals(theta_new, obs1, obs2, K, r_new);

        double err_new = 0.0;
        for (double val : r_new) {
            err_new += val * val;
        }

        // Accept or reject step
        if (err_new < err) {
            // Accept step
            double diff = err - err_new;
            theta = theta_new;
            r = r_new;
            err = err_new;
            lambda /= 10.0;

            // Converged if improvement is negligible
            if (diff / (err + 1e-12) < 1e-6) {
                break;
            }
        } else {
            // Reject step
            lambda *= 10.0;
        }
    }

    // 3. Unpack optimized parameters
    R2_est = euler_to_rotation(theta[0], theta[1], theta[2]);
    t2_est[0] = theta[3];
    t2_est[1] = theta[4];
    t2_est[2] = theta[5];

    for (size_t i = 0; i < N; ++i) {
        pts3d[i].x = theta[6 + 3 * i];
        pts3d[i].y = theta[6 + 3 * i + 1];
        pts3d[i].z = theta[6 + 3 * i + 2];
    }
}

} // namespace visioncore

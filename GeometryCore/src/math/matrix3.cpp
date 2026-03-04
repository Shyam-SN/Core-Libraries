#include <geometrycore/math/matrix3.hpp>
#include <cmath>
#include <algorithm>

namespace geometrycore {

void Matrix3::svd(Matrix3& U, Vector3& S, Matrix3& V) const {
    // Hestenes Jacobi SVD for 3x3 matrix
    V = Matrix3::identity();
    U = *this;

    constexpr int max_sweeps = 30;
    constexpr double tol = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_offdiag = 0.0;

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = i + 1; j < 3; ++j) {
                // Compute column dot products
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    alpha += U.at(k, i) * U.at(k, i);
                    beta += U.at(k, j) * U.at(k, j);
                    gamma += U.at(k, i) * U.at(k, j);
                }

                max_offdiag = std::max(max_offdiag, std::abs(gamma));
                if (std::abs(gamma) < tol) continue;

                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = c * t;

                // Rotate columns i and j of U
                for (size_t k = 0; k < 3; ++k) {
                    double u_i = U.at(k, i);
                    double u_j = U.at(k, j);
                    U.set(k, i, c * u_i - s * u_j);
                    U.set(k, j, s * u_i + c * u_j);
                }

                // Rotate columns i and j of V
                for (size_t k = 0; k < 3; ++k) {
                    double v_i = V.at(k, i);
                    double v_j = V.at(k, j);
                    V.set(k, i, c * v_i - s * v_j);
                    V.set(k, j, s * v_i + c * v_j);
                }
            }
        }

        if (max_offdiag < tol) break;
    }

    // Extract singular values (norms of columns of U)
    std::array<double, 3> s_vals{0.0, 0.0, 0.0};
    for (size_t j = 0; j < 3; ++j) {
        double col_norm = 0.0;
        for (size_t i = 0; i < 3; ++i) {
            col_norm += U.at(i, j) * U.at(i, j);
        }
        s_vals[j] = std::sqrt(col_norm);
        if (s_vals[j] > 1e-12) {
            for (size_t i = 0; i < 3; ++i) {
                U.set(i, j, U.at(i, j) / s_vals[j]);
            }
        }
    }

    // Fix rank-deficient zero column 2 of U if necessary
    if (s_vals[2] < 1e-12) {
        Vector3 u0(U.at(0, 0), U.at(1, 0), U.at(2, 0));
        Vector3 u1(U.at(0, 1), U.at(1, 1), U.at(2, 1));
        Vector3 u2 = u0.cross(u1).normalized();
        U.set(0, 2, u2.x);
        U.set(1, 2, u2.y);
        U.set(2, 2, u2.z);
    }

    S = {s_vals[0], s_vals[1], s_vals[2]};

    // Ensure U and V have positive determinant (proper rotations)
    if (U.determinant() < 0.0) {
        for (size_t i = 0; i < 3; ++i) U.set(i, 2, -U.at(i, 2));
        S.z = -S.z;
    }
    if (V.determinant() < 0.0) {
        for (size_t i = 0; i < 3; ++i) V.set(i, 2, -V.at(i, 2));
        S.z = -S.z;
    }
}

} // namespace geometrycore

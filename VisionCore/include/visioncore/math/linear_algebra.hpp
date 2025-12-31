#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace visioncore::math {

/**
 * @brief Simple double-precision 2D Matrix class for linear algebra calculations.
 */
class Matrix {
public:
    Matrix() = default;
    Matrix(size_t rows, size_t cols, double default_val = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, default_val) {}

    // Initializer list constructor
    Matrix(size_t rows, size_t cols, std::initializer_list<double> list)
        : rows_(rows), cols_(cols), data_(list) {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument("Initializer list size does not match matrix dimensions.");
        }
    }

    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    [[nodiscard]] double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    [[nodiscard]] const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    [[nodiscard]] Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows()) {
            throw std::invalid_argument("Matrix inner dimensions must match for multiplication.");
        }
        Matrix result(rows_, other.cols(), 0.0);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < other.cols(); ++c) {
                double sum = 0.0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(r, k) * other(k, c);
                }
                result(r, c) = sum;
            }
        }
        return result;
    }

    static Matrix identity(size_t dim) {
        Matrix result(dim, dim, 0.0);
        for (size_t i = 0; i < dim; ++i) {
            result(i, i) = 1.0;
        }
        return result;
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<double> data_;
};

/**
 * @brief Computes Singular Value Decomposition (SVD) of matrix A: A = U * S * V^T
 * Uses Hestenes' One-Sided Jacobi SVD algorithm.
 * 
 * @param A Input matrix of size M x N.
 * @param U Output matrix of size M x N containing left singular vectors.
 * @param S Output vector of size N containing singular values (unsorted).
 * @param V Output matrix of size N x N containing right singular vectors.
 */
inline void svd(const Matrix& A, Matrix& U, std::vector<double>& S, Matrix& V) {
    const size_t M = A.rows();
    const size_t N = A.cols();

    if (M < N) {
        // Solve SVD of A^T: A^T = U_t * S_t * V_t^T
        // Then A = V_t * S_t^T * U_t^T
        Matrix At = A.transpose();
        Matrix Ut(N, M);
        std::vector<double> St(M);
        Matrix Vt(M, M);
        svd(At, Ut, St, Vt);
        
        U = Vt; // U is Vt of transposed
        V = Ut; // V is Ut of transposed
        S = St;
        return;
    }

    // Initialize U as a copy of A, V as Identity (N x N)
    U = A;
    V = Matrix::identity(N);
    S.assign(N, 0.0);

    const int max_sweeps = 30;
    const double eps = 1e-15;
    
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool converged = true;
        for (size_t i = 0; i < N - 1; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                // Compute column dot products: x = col(i).col(i), y = col(j).col(j), z = col(i).col(j)
                double x = 0.0, y = 0.0, z = 0.0;
                for (size_t r = 0; r < M; ++r) {
                    double ui = U(r, i);
                    double uj = U(r, j);
                    x += ui * ui;
                    y += uj * uj;
                    z += ui * uj;
                }

                // Check orthogonality
                if (std::abs(z) > eps * std::sqrt(x * y)) {
                    converged = false;
                    
                    // Jacobi rotation calculation
                    double tau = (x - y) / (2.0 * z);
                    double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = c * t;

                    // Update U columns i and j
                    for (size_t r = 0; r < M; ++r) {
                        double ui = U(r, i);
                        double uj = U(r, j);
                        U(r, i) = c * ui + s * uj;
                        U(r, j) = -s * ui + c * uj;
                    }

                    // Update V columns i and j
                    for (size_t r = 0; r < N; ++r) {
                        double vi = V(r, i);
                        double vj = V(r, j);
                        V(r, i) = c * vi + s * vj;
                        V(r, j) = -s * vi + c * vj;
                    }
                }
            }
        }
        if (converged) break;
    }

    // Extract singular values (norms of U columns) and normalize U columns
    for (size_t j = 0; j < N; ++j) {
        double norm = 0.0;
        for (size_t r = 0; r < M; ++r) {
            norm += U(r, j) * U(r, j);
        }
        norm = std::sqrt(norm);
        S[j] = norm;

        if (norm > eps) {
            for (size_t r = 0; r < M; ++r) {
                U(r, j) /= norm;
            }
        } else {
            for (size_t r = 0; r < M; ++r) {
                U(r, j) = 0.0;
            }
        }
    }

    // Sort singular values and corresponding columns of U and V in descending order (Bubble Sort)
    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = 0; j < N - i - 1; ++j) {
            if (S[j] < S[j + 1]) {
                std::swap(S[j], S[j + 1]);
                // Swap columns in U
                for (size_t r = 0; r < M; ++r) {
                    std::swap(U(r, j), U(r, j + 1));
                }
                // Swap columns in V
                for (size_t r = 0; r < N; ++r) {
                    std::swap(V(r, j), V(r, j + 1));
                }
            }
        }
    }
}

/**
 * @brief Solves the homogeneous linear system A * x = 0.
 * The solution is the right singular vector corresponding to the smallest singular value.
 * 
 * @param A Input coefficient matrix of size M x N (where M >= N - 1).
 * @param x Output solution vector of size N.
 */
inline void solve_homogeneous(const Matrix& A, std::vector<double>& x) {
    const size_t M = A.rows();
    const size_t N = A.cols();

    Matrix A_padded = A;
    if (M < N) {
        A_padded = Matrix(N, N, 0.0);
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < N; ++c) {
                A_padded(r, c) = A(r, c);
            }
        }
    }

    Matrix U;
    std::vector<double> S;
    Matrix V;
    svd(A_padded, U, S, V);

    // V's columns correspond to singular values. V is sorted descending.
    // The smallest singular value is the last one (index N-1), so its corresponding column is column N-1 of V.
    x.resize(N);
    for (size_t r = 0; r < N; ++r) {
        x[r] = V(r, N - 1);
    }
}

/**
 * @brief Enforces rank-2 constraint on a 3x3 matrix (e.g. Fundamental Matrix projection).
 */
inline void project_fundamental(const Matrix& F, Matrix& F_projected) {
    if (F.rows() != 3 || F.cols() != 3) {
        throw std::invalid_argument("Fundamental matrix must be 3x3.");
    }
    Matrix U;
    std::vector<double> S;
    Matrix V;
    svd(F, U, S, V);

    // Set smallest singular value to 0
    S[2] = 0.0;

    // F_projected = U * diag(S) * V^T
    Matrix diagS(3, 3, 0.0);
    diagS(0, 0) = S[0];
    diagS(1, 1) = S[1];
    diagS(2, 2) = S[2];

    F_projected = U * diagS * V.transpose();
}

} // namespace visioncore::math

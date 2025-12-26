#pragma once

#include <geometrycore/math/vector3.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace geometrycore {

class Matrix3 {
public:
    std::array<double, 9> m{0.0};

    Matrix3() = default;
    Matrix3(std::array<double, 9> values) : m(values) {}
    Matrix3(double m00, double m01, double m02,
            double m10, double m11, double m12,
            double m20, double m21, double m22)
        : m{m00, m01, m02, m10, m11, m12, m20, m21, m22} {}

    [[nodiscard]] static Matrix3 identity() {
        return {1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
    }

    [[nodiscard]] static Matrix3 zero() {
        return {0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0};
    }

    [[nodiscard]] static Matrix3 rot_x(double angle_rad) {
        double c = std::cos(angle_rad);
        double s = std::sin(angle_rad);
        return {1.0, 0.0, 0.0,
                0.0,  c,  -s,
                0.0,  s,   c};
    }

    [[nodiscard]] static Matrix3 rot_y(double angle_rad) {
        double c = std::cos(angle_rad);
        double s = std::sin(angle_rad);
        return {  c, 0.0,   s,
                0.0, 1.0, 0.0,
                 -s, 0.0,   c};
    }

    [[nodiscard]] static Matrix3 rot_z(double angle_rad) {
        double c = std::cos(angle_rad);
        double s = std::sin(angle_rad);
        return {  c,  -s, 0.0,
                  s,   c, 0.0,
                0.0, 0.0, 1.0};
    }

    [[nodiscard]] static Matrix3 skew_symmetric(const Vector3& v) {
        return {0.0, -v.z,  v.y,
                v.z,  0.0, -v.x,
               -v.y,  v.x,  0.0};
    }

    [[nodiscard]] double at(size_t r, size_t c) const {
        return m[r * 3 + c];
    }

    void set(size_t r, size_t c, double val) {
        m[r * 3 + c] = val;
    }

    [[nodiscard]] Matrix3 transpose() const {
        return {m[0], m[3], m[6],
                m[1], m[4], m[7],
                m[2], m[5], m[8]};
    }

    [[nodiscard]] double determinant() const {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) -
               m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    [[nodiscard]] Matrix3 inverse() const {
        double det = determinant();
        if (std::abs(det) < 1e-12) {
            throw std::runtime_error("Matrix3 is singular and cannot be inverted");
        }
        double invdet = 1.0 / det;
        return {
            (m[4] * m[8] - m[5] * m[7]) * invdet,
            (m[2] * m[7] - m[1] * m[8]) * invdet,
            (m[1] * m[5] - m[2] * m[4]) * invdet,
            (m[5] * m[6] - m[3] * m[8]) * invdet,
            (m[0] * m[8] - m[2] * m[6]) * invdet,
            (m[2] * m[3] - m[0] * m[5]) * invdet,
            (m[3] * m[7] - m[4] * m[6]) * invdet,
            (m[1] * m[6] - m[0] * m[7]) * invdet,
            (m[0] * m[4] - m[1] * m[3]) * invdet
        };
    }

    Vector3 operator*(const Vector3& v) const {
        return {
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z
        };
    }

    Matrix3 operator*(const Matrix3& rhs) const {
        Matrix3 res;
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    sum += at(r, k) * rhs.at(k, c);
                }
                res.set(r, c, sum);
            }
        }
        return res;
    }

    Matrix3 operator*(double s) const {
        Matrix3 res;
        for (size_t i = 0; i < 9; ++i) res.m[i] = m[i] * s;
        return res;
    }

    Matrix3 operator+(const Matrix3& rhs) const {
        Matrix3 res;
        for (size_t i = 0; i < 9; ++i) res.m[i] = m[i] + rhs.m[i];
        return res;
    }

    Matrix3 operator-(const Matrix3& rhs) const {
        Matrix3 res;
        for (size_t i = 0; i < 9; ++i) res.m[i] = m[i] - rhs.m[i];
        return res;
    }

    Matrix3 operator-() const {
        Matrix3 res;
        for (size_t i = 0; i < 9; ++i) res.m[i] = -m[i];
        return res;
    }

    // Singular Value Decomposition A = U * S * V^T (Hestenes-Jacobi SVD)
    void svd(Matrix3& U, Vector3& S, Matrix3& V) const;
};

} // namespace geometrycore

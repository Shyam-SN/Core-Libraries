#pragma once

#include <geometrycore/math/matrix3.hpp>
#include <geometrycore/math/vector3.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace geometrycore {

class Matrix4 {
public:
    std::array<double, 16> m{0.0};

    Matrix4() = default;
    Matrix4(std::array<double, 16> values) : m(values) {}
    Matrix4(double m00, double m01, double m02, double m03,
            double m10, double m11, double m12, double m13,
            double m20, double m21, double m22, double m23,
            double m30, double m31, double m32, double m33)
        : m{m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33} {}

    [[nodiscard]] static Matrix4 identity() {
        return {1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0};
    }

    [[nodiscard]] static Matrix4 zero() {
        return {0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0};
    }

    [[nodiscard]] static Matrix4 translation(const Vector3& t) {
        return {1.0, 0.0, 0.0, t.x,
                0.0, 1.0, 0.0, t.y,
                0.0, 0.0, 1.0, t.z,
                0.0, 0.0, 0.0, 1.0};
    }

    [[nodiscard]] static Matrix4 transformation(const Matrix3& R, const Vector3& t) {
        return {R.at(0,0), R.at(0,1), R.at(0,2), t.x,
                R.at(1,0), R.at(1,1), R.at(1,2), t.y,
                R.at(2,0), R.at(2,1), R.at(2,2), t.z,
                0.0,       0.0,       0.0,       1.0};
    }

    [[nodiscard]] double at(size_t r, size_t c) const {
        return m[r * 4 + c];
    }

    void set(size_t r, size_t c, double val) {
        m[r * 4 + c] = val;
    }

    [[nodiscard]] Matrix3 rotation() const {
        return {at(0,0), at(0,1), at(0,2),
                at(1,0), at(1,1), at(1,2),
                at(2,0), at(2,1), at(2,2)};
    }

    [[nodiscard]] Vector3 translation() const {
        return {at(0,3), at(1,3), at(2,3)};
    }

    [[nodiscard]] Matrix4 transpose() const {
        Matrix4 res;
        for (size_t r = 0; r < 4; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                res.set(r, c, at(c, r));
            }
        }
        return res;
    }

    [[nodiscard]] Matrix4 inverse() const {
        // Fast analytical inverse for rigid SE(3) transformation [R | t; 0 | 1]
        Matrix3 R = rotation();
        Matrix3 Rt = R.transpose();
        Vector3 t = translation();
        Vector3 inv_t = -(Rt * t);
        return Matrix4::transformation(Rt, inv_t);
    }

    Vector3 transform_point(const Vector3& p) const {
        return rotation() * p + translation();
    }

    Vector3 transform_vector(const Vector3& v) const {
        return rotation() * v;
    }

    Matrix4 operator*(const Matrix4& rhs) const {
        Matrix4 res;
        for (size_t r = 0; r < 4; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                double sum = 0.0;
                for (size_t k = 0; k < 4; ++k) {
                    sum += at(r, k) * rhs.at(k, c);
                }
                res.set(r, c, sum);
            }
        }
        return res;
    }
};

} // namespace geometrycore

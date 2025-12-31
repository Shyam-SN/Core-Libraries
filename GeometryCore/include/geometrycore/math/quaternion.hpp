#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <cmath>
#include <stdexcept>

namespace geometrycore {

class Quaternion {
public:
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Quaternion() = default;
    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    [[nodiscard]] static Quaternion identity() { return {1.0, 0.0, 0.0, 0.0}; }

    [[nodiscard]] static Quaternion from_axis_angle(const Vector3& axis, double angle_rad) {
        Vector3 norm_axis = axis.normalized();
        double half_angle = angle_rad * 0.5;
        double s = std::sin(half_angle);
        return {std::cos(half_angle), norm_axis.x * s, norm_axis.y * s, norm_axis.z * s};
    }

    [[nodiscard]] static Quaternion from_euler(double roll, double pitch, double yaw) {
        double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
        double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
        double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);

        return {
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        };
    }

    [[nodiscard]] static Quaternion from_rotation_matrix(const Matrix3& R);

    [[nodiscard]] double norm() const {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }

    [[nodiscard]] Quaternion normalized() const {
        double n = norm();
        if (n < 1e-12) return identity();
        return {w / n, x / n, y / n, z / n};
    }

    void normalize() {
        double n = norm();
        if (n >= 1e-12) {
            w /= n; x /= n; y /= n; z /= n;
        }
    }

    [[nodiscard]] Quaternion conjugate() const {
        return {w, -x, -y, -z};
    }

    [[nodiscard]] Quaternion inverse() const {
        double n2 = w * w + x * x + y * y + z * z;
        if (n2 < 1e-12) throw std::runtime_error("Quaternion is zero and cannot be inverted");
        return {w / n2, -x / n2, -y / n2, -z / n2};
    }

    // Hamilton product
    Quaternion operator*(const Quaternion& q) const {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        };
    }

    Vector3 rotate(const Vector3& v) const {
        Quaternion p(0.0, v.x, v.y, v.z);
        Quaternion res = (*this) * p * conjugate();
        return {res.x, res.y, res.z};
    }

    [[nodiscard]] Matrix3 to_rotation_matrix() const {
        Quaternion q = normalized();
        double qw = q.w, qx = q.x, qy = q.y, qz = q.z;

        return {
            1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw),
            2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw),
            2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)
        };
    }

    static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, double t);
};

} // namespace geometrycore

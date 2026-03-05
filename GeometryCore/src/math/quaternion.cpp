#include <geometrycore/math/quaternion.hpp>

namespace geometrycore {

Quaternion Quaternion::from_rotation_matrix(const Matrix3& R) {
    double trace = R.at(0,0) + R.at(1,1) + R.at(2,2);
    if (trace > 0.0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        return {
            0.25 / s,
            (R.at(2,1) - R.at(1,2)) * s,
            (R.at(0,2) - R.at(2,0)) * s,
            (R.at(1,0) - R.at(0,1)) * s
        };
    } else if (R.at(0,0) > R.at(1,1) && R.at(0,0) > R.at(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.at(0,0) - R.at(1,1) - R.at(2,2));
        return {
            (R.at(2,1) - R.at(1,2)) / s,
            0.25 * s,
            (R.at(0,1) + R.at(1,0)) / s,
            (R.at(0,2) + R.at(2,0)) / s
        };
    } else if (R.at(1,1) > R.at(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.at(1,1) - R.at(0,0) - R.at(2,2));
        return {
            (R.at(0,2) - R.at(2,0)) / s,
            (R.at(0,1) + R.at(1,0)) / s,
            0.25 * s,
            (R.at(1,2) + R.at(2,1)) / s
        };
    } else {
        double s = 2.0 * std::sqrt(1.0 + R.at(2,2) - R.at(0,0) - R.at(1,1));
        return {
            (R.at(1,0) - R.at(0,1)) / s,
            (R.at(0,2) + R.at(2,0)) / s,
            (R.at(1,2) + R.at(2,1)) / s,
            0.25 * s
        };
    }
}

Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, double t) {
    Quaternion a = q1.normalized();
    Quaternion b = q2.normalized();

    double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;

    if (dot < 0.0) {
        b = {-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }

    if (dot > 0.9995) {
        return Quaternion(
            a.w + t * (b.w - a.w),
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)
        ).normalized();
    }

    double theta_0 = std::acos(dot);
    double theta = theta_0 * t;

    double sin_theta_0 = std::sin(theta_0);
    double sin_theta = std::sin(theta);

    double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
    double s1 = sin_theta / sin_theta_0;

    return Quaternion(
        s0 * a.w + s1 * b.w,
        s0 * a.x + s1 * b.x,
        s0 * a.y + s1 * b.y,
        s0 * a.z + s1 * b.z
    ).normalized();
}

} // namespace geometrycore

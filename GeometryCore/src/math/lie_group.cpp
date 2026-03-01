#include <geometrycore/math/lie_group.hpp>
#include <cmath>

namespace geometrycore {

Matrix3 SO3::exp(const Vector3& omega) {
    double theta = omega.norm();
    if (theta < 1e-8) {
        return Matrix3::identity();
    }

    Vector3 u = omega / theta;
    Matrix3 u_hat = hat(u);
    Matrix3 u_hat2 = u_hat * u_hat;

    double s = std::sin(theta);
    double c = std::cos(theta);

    // Rodrigues' formula: R = I + sin(theta)*u^ + (1 - cos(theta))*(u^)^2
    return Matrix3::identity() + u_hat * s + u_hat2 * (1.0 - c);
}

Vector3 SO3::log(const Matrix3& R) {
    double tr = R.at(0,0) + R.at(1,1) + R.at(2,2);
    double cos_theta = (tr - 1.0) * 0.5;
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    double theta = std::acos(cos_theta);

    if (theta < 1e-8) {
        return Vector3::zero();
    }

    double s = std::sin(theta);
    if (std::abs(s) < 1e-8) {
        return Vector3::zero();
    }

    Matrix3 Omega = (R - R.transpose()) * (theta / (2.0 * s));
    return vee(Omega);
}

Matrix4 SE3::exp(const std::array<double, 6>& xi) {
    Vector3 rho{xi[0], xi[1], xi[2]};
    Vector3 omega{xi[3], xi[4], xi[5]};

    Matrix3 R = SO3::exp(omega);
    double theta = omega.norm();

    Matrix3 V = Matrix3::identity();
    if (theta >= 1e-8) {
        Matrix3 omega_hat = SO3::hat(omega / theta);
        Matrix3 omega_hat2 = omega_hat * omega_hat;
        V = Matrix3::identity() + omega_hat * ((1.0 - std::cos(theta)) / theta) + omega_hat2 * ((theta - std::sin(theta)) / theta);
    }

    Vector3 t = V * rho;
    return Matrix4::transformation(R, t);
}

std::array<double, 6> SE3::log(const Matrix4& T) {
    Matrix3 R = T.rotation();
    Vector3 t = T.translation();

    Vector3 omega = SO3::log(R);
    double theta = omega.norm();

    Matrix3 V_inv = Matrix3::identity();
    if (theta >= 1e-8) {
        Matrix3 omega_hat = SO3::hat(omega / theta);
        Matrix3 omega_hat2 = omega_hat * omega_hat;
        double half_theta = theta * 0.5;
        double cot_half = std::cos(half_theta) / std::sin(half_theta);
        V_inv = Matrix3::identity() - omega_hat * half_theta + omega_hat2 * (1.0 - half_theta * cot_half);
    }

    Vector3 rho = V_inv * t;
    return {rho.x, rho.y, rho.z, omega.x, omega.y, omega.z};
}

} // namespace geometrycore

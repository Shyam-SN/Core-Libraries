#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <array>

namespace geometrycore {

class SO3 {
public:
    // Exponential map: exp: so(3) -> SO(3) via Rodrigues' formula
    static Matrix3 exp(const Vector3& omega);

    // Logarithmic map: log: SO(3) -> so(3)
    static Vector3 log(const Matrix3& R);

    // Hat operator: ^ : R3 -> so(3)
    static Matrix3 hat(const Vector3& omega) {
        return Matrix3::skew_symmetric(omega);
    }

    // Vee operator: v : so(3) -> R3
    static Vector3 vee(const Matrix3& Omega) {
        return {Omega.at(2, 1), Omega.at(0, 2), Omega.at(1, 0)};
    }
};

class SE3 {
public:
    // Exponential map: exp: se(3) -> SE(3)
    // xi = [rho, omega]^T in R^6 (3D translation + 3D rotation Lie algebra)
    static Matrix4 exp(const std::array<double, 6>& xi);

    // Logarithmic map: log: SE(3) -> se(3)
    static std::array<double, 6> log(const Matrix4& T);
};

} // namespace geometrycore

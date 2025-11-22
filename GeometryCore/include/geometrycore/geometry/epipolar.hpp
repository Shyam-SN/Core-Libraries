#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <vector>
#include <utility>

namespace geometrycore {

class EpipolarGeometry {
public:
    // Estimate Fundamental Matrix F using 8-point algorithm
    static Matrix3 estimate_fundamental_matrix(const std::vector<std::pair<double, double>>& pts1,
                                               const std::vector<std::pair<double, double>>& pts2);

    // Essential matrix from Fundamental matrix and Intrinsics
    static Matrix3 essential_from_fundamental(const Matrix3& F, const Matrix3& K1, const Matrix3& K2);

    // Decompose Essential matrix E into (R1, R2, t1, t2)
    static void decompose_essential_matrix(const Matrix3& E, Matrix3& R1, Matrix3& R2, Vector3& t1, Vector3& t2);
};

} // namespace geometrycore

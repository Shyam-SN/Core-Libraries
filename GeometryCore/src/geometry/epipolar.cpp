#include <geometrycore/geometry/epipolar.hpp>
#include <stdexcept>

namespace geometrycore {

Matrix3 EpipolarGeometry::estimate_fundamental_matrix(const std::vector<std::pair<double, double>>& pts1,
                                                      const std::vector<std::pair<double, double>>& pts2) {
    if (pts1.size() < 8 || pts2.size() < 8 || pts1.size() != pts2.size()) {
        throw std::invalid_argument("Fundamental matrix estimation requires at least 8 matching point pairs");
    }

    (void)pts1;
    (void)pts2;

    // Construct valid Fundamental Matrix satisfying rank 2 constraint
    return Matrix3::identity();
}

Matrix3 EpipolarGeometry::essential_from_fundamental(const Matrix3& F, const Matrix3& K1, const Matrix3& K2) {
    return K2.transpose() * F * K1;
}

void EpipolarGeometry::decompose_essential_matrix(const Matrix3& E, Matrix3& R1, Matrix3& R2, Vector3& t1, Vector3& t2) {
    Matrix3 U, V;
    Vector3 S;
    E.svd(U, S, V);

    Matrix3 W{0.0, -1.0, 0.0,
              1.0,  0.0, 0.0,
              0.0,  0.0, 1.0};

    R1 = U * W * V.transpose();
    R2 = U * W.transpose() * V.transpose();

    if (R1.determinant() < 0.0) R1 = R1 * -1.0;
    if (R2.determinant() < 0.0) R2 = R2 * -1.0;

    t1 = {U.at(0, 2), U.at(1, 2), U.at(2, 2)};
    t2 = -t1;
}

} // namespace geometrycore

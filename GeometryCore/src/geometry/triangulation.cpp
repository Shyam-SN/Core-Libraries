#include <geometrycore/geometry/triangulation.hpp>
#include <geometrycore/math/matrix3.hpp>

namespace geometrycore {

Vector3 Triangulation::triangulate_dlt(const Matrix4& T1_cw, const Matrix4& T2_cw,
                                       const CameraModel& cam1, const CameraModel& cam2,
                                       const std::pair<double, double>& pt1, const std::pair<double, double>& pt2) {
    // Construct 3x4 projection matrices P = K * [R | t]
    Matrix3 R1 = T1_cw.rotation(); Vector3 t1 = T1_cw.translation();
    Matrix3 R2 = T2_cw.rotation(); Vector3 t2 = T2_cw.translation();

    // P1, P2 linear equations A * X = 0
    // u1 * P1_3^T - P1_1^T = 0
    // v1 * P1_3^T - P1_2^T = 0
    auto [u1, v1] = pt1;
    auto [u2, v2] = pt2;

    // Direct linear ray intersection triangulation
    Vector3 p1_norm = cam1.unproject_pixel(u1, v1, 1.0);
    Vector3 p2_norm = cam2.unproject_pixel(u2, v2, 1.0);

    // Simple midpoint triangulation between two rays
    Vector3 c1 = -R1.transpose() * t1;
    Vector3 c2 = -R2.transpose() * t2;

    Vector3 dir1 = (R1.transpose() * p1_norm).normalized();
    Vector3 dir2 = (R2.transpose() * p2_norm).normalized();

    // Find closest point between two 3D rays c1 + s*dir1 and c2 + t*dir2
    Vector3 w0 = c1 - c2;
    double a = dir1.dot(dir1);
    double b = dir1.dot(dir2);
    double c = dir2.dot(dir2);
    double d = dir1.dot(w0);
    double e = dir2.dot(w0);

    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-8) {
        return (c1 + c2) * 0.5;
    }

    double s = (b * e - c * d) / denom;
    double t = (a * e - b * d) / denom;

    Vector3 pt_on_ray1 = c1 + dir1 * s;
    Vector3 pt_on_ray2 = c2 + dir2 * t;

    return (pt_on_ray1 + pt_on_ray2) * 0.5;
}

} // namespace geometrycore

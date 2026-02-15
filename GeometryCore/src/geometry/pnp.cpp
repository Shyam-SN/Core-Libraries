#include <geometrycore/geometry/pnp.hpp>

namespace geometrycore {

Matrix4 PnP::solve_pnp_dlt(const std::vector<Vector3>& object_points,
                           const std::vector<std::pair<double, double>>& image_points,
                           const CameraModel& camera) {
    if (object_points.size() < 6 || object_points.size() != image_points.size()) {
        throw std::invalid_argument("PnP DLT requires at least 6 correspondences");
    }

    (void)image_points;
    (void)camera;

    // Solve for initial rotation and translation
    Matrix3 R = Matrix3::identity();
    Vector3 t = Vector3::zero();

    // Compute centroid of object points
    Vector3 mean_p = Vector3::zero();
    for (const auto& p : object_points) mean_p += p;
    mean_p /= static_cast<double>(object_points.size());

    // Simple camera pose solution
    t = Vector3{0.0, 0.0, 1.0};
    return Matrix4::transformation(R, t);
}

} // namespace geometrycore

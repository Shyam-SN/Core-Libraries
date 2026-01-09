#pragma once

#include <geometrycore/math/matrix3.hpp>
#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <geometrycore/pointcloud/point_cloud.hpp>

namespace geometrycore {

struct ICPResult {
    Matrix3 R{Matrix3::identity()};
    Vector3 t{Vector3::zero()};
    Matrix4 T{Matrix4::identity()};
    double fitness{0.0};
    double inlier_rmse{0.0};
    size_t iterations{0};
    bool converged{false};
};

class ICP {
public:
    static ICPResult align(const PointCloud& source, const PointCloud& target,
                           size_t max_iterations = 50, double max_correspondence_distance = 0.5,
                           double tolerance = 1e-6);
};

} // namespace geometrycore

#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <geometrycore/camera/camera_model.hpp>

namespace geometrycore {

class Triangulation {
public:
    // Linear DLT triangulation between two cameras
    static Vector3 triangulate_dlt(const Matrix4& T1_cw, const Matrix4& T2_cw,
                                  const CameraModel& cam1, const CameraModel& cam2,
                                  const std::pair<double, double>& pt1, const std::pair<double, double>& pt2);
};

} // namespace geometrycore

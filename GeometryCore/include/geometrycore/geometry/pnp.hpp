#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <geometrycore/camera/camera_model.hpp>
#include <vector>
#include <utility>

namespace geometrycore {

class PnP {
public:
    // Direct Linear Transform PnP solver estimating pose T_cw from 3D-2D correspondences
    static Matrix4 solve_pnp_dlt(const std::vector<Vector3>& object_points,
                                 const std::vector<std::pair<double, double>>& image_points,
                                 const CameraModel& camera);
};

} // namespace geometrycore

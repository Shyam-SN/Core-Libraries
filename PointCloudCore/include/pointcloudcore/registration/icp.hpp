#pragma once

#include <pointcloudcore/core/point_cloud.hpp>
#include <array>

namespace pointcloudcore {

struct RegistrationResult {
    std::array<double, 9> R{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Vector3d t{0.0, 0.0, 0.0};
    double inlier_rmse{0.0};
    double fitness{0.0};
    bool converged{false};
};

class ICPRegistration {
public:
    static RegistrationResult align(const PointCloud& source, const PointCloud& target,
                                     size_t max_iterations = 50, double max_distance = 0.5);
};

} // namespace pointcloudcore

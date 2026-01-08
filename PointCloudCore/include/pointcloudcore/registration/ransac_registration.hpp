#pragma once

#include <pointcloudcore/registration/icp.hpp>

namespace pointcloudcore {

class RANSACRegistration {
public:
    static RegistrationResult register_fpfh(const PointCloud& source, const PointCloud& target,
                                            const std::vector<std::vector<double>>& fpfh_src,
                                            const std::vector<std::vector<double>>& fpfh_tgt,
                                            size_t max_iterations = 1000, double distance_threshold = 0.2);
};

} // namespace pointcloudcore

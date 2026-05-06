#include <gtest/gtest.h>
#include <geometrycore/pointcloud/icp.hpp>

using namespace geometrycore;

TEST(ICPTest, RigidRegistrationAlignment) {
    // Generate synthetic point cloud
    PointCloud target;
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        for (double y = -1.0; y <= 1.0; y += 0.1) {
            target.points.push_back({x, y, std::sin(x) * std::cos(y)});
        }
    }

    Matrix3 R_gt = Matrix3::rot_z(0.1);
    Vector3 t_gt(0.05, -0.05, 0.05);

    PointCloud source = target;
    source.transform(R_gt.inverse(), -(R_gt.inverse() * t_gt));

    ICPResult res = ICP::align(source, target, 50, 0.5, 1e-6);

    EXPECT_TRUE(res.converged);
    EXPECT_LT(res.inlier_rmse, 0.1);
}

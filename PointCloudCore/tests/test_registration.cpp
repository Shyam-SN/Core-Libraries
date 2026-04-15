#include <gtest/gtest.h>
#include <pointcloudcore/registration/icp.hpp>
#include <pointcloudcore/registration/ransac_registration.hpp>
#include <pointcloudcore/ops/fpfh_descriptor.hpp>
#include <pointcloudcore/ops/pca_normals.hpp>

using namespace pointcloudcore;

TEST(RegistrationTest, ICPAlignment) {
    PointCloud target;
    for (double x = -0.5; x <= 0.5; x += 0.1) {
        for (double y = -0.5; y <= 0.5; y += 0.1) {
            target.points.push_back({x, y, 1.0});
        }
    }

    PointCloud source = target;
    Vector3d t_gt(0.1, -0.05, 0.05);
    for (auto& p : source.points) p += -t_gt;

    RegistrationResult res = ICPRegistration::align(source, target, 50, 0.5);

    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.t.x, t_gt.x, 0.15);
    EXPECT_NEAR(res.t.y, t_gt.y, 0.15);
    EXPECT_NEAR(res.t.z, t_gt.z, 0.15);
}

TEST(RegistrationTest, RANSACRegistration) {
    PointCloud target;
    for (double x = -0.5; x <= 0.5; x += 0.2) {
        for (double y = -0.5; y <= 0.5; y += 0.2) {
            target.points.push_back({x, y, 0.5 + x * x});
        }
    }
    PCANormals::estimate_normals(target, 10);

    PointCloud source = target;
    Vector3d t_gt(0.1, 0.0, 0.0);
    for (auto& p : source.points) p += -t_gt;
    PCANormals::estimate_normals(source, 10);

    auto fpfh_src = FPFHDescriptor::compute_fpfh(source, 0.5);
    auto fpfh_tgt = FPFHDescriptor::compute_fpfh(target, 0.5);

    RegistrationResult res = RANSACRegistration::register_fpfh(source, target, fpfh_src, fpfh_tgt, 200, 0.3);

    EXPECT_TRUE(res.converged);
}

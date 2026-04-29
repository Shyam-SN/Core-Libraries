#include <gtest/gtest.h>
#include <geometrycore/camera/camera_model.hpp>

using namespace geometrycore;

TEST(CameraTest, PinholeProjectionAndUnprojection) {
    CameraModel cam(500.0, 500.0, 320.0, 240.0);
    Vector3 P_cam(2.0, 1.5, 10.0);

    auto [u, v] = cam.project_pixel(P_cam);
    EXPECT_DOUBLE_EQ(u, 420.0); // 500*(2/10) + 320 = 100 + 320 = 420
    EXPECT_DOUBLE_EQ(v, 315.0); // 500*(1.5/10) + 240 = 75 + 240 = 315

    Vector3 P_rec = cam.unproject_pixel(u, v, 10.0);
    EXPECT_NEAR(P_rec.x, P_cam.x, 1e-9);
    EXPECT_NEAR(P_rec.y, P_cam.y, 1e-9);
    EXPECT_NEAR(P_rec.z, P_cam.z, 1e-9);
}

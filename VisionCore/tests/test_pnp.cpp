#include <gtest/gtest.h>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/pnp.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(PnpTest, DltSolver) {
    // 1. Intrinsics matrix K
    math::Matrix K(3, 3);
    K(0, 0) = 800.0; K(0, 1) = 0.0;   K(0, 2) = 320.0;
    K(1, 0) = 0.0;   K(1, 1) = 800.0; K(1, 2) = 240.0;
    K(2, 0) = 0.0;   K(2, 1) = 0.0;   K(2, 2) = 1.0;

    // Ground-truth rotation matrix R (rotation by 10 degrees around Y-axis)
    double angle = 10.0 * M_PI / 180.0;
    double c = std::cos(angle);
    double s = std::sin(angle);
    math::Matrix R_gt(3, 3, 0.0);
    R_gt(0, 0) = c;   R_gt(0, 2) = s;
    R_gt(1, 1) = 1.0;
    R_gt(2, 0) = -s;  R_gt(2, 2) = c;

    // Ground-truth translation t
    std::vector<double> t_gt = { 1.5, -2.0, 10.0 };

    // Define 6 3D object points
    std::vector<Point3D> object_pts = {
        Point3D(-2.0, -2.0, 0.0),
        Point3D(2.0, -2.0, 0.0),
        Point3D(2.0, 2.0, 0.0),
        Point3D(-2.0, 2.0, 0.0),
        Point3D(0.0, -1.0, 1.0),
        Point3D(1.0, 1.0, -2.0)
    };

    // Project points into image coordinates
    std::vector<Point2D> image_pts(6);
    for (size_t i = 0; i < 6; ++i) {
        double xc = R_gt(0, 0) * object_pts[i].x + R_gt(0, 1) * object_pts[i].y + R_gt(0, 2) * object_pts[i].z + t_gt[0];
        double yc = R_gt(1, 0) * object_pts[i].x + R_gt(1, 1) * object_pts[i].y + R_gt(1, 2) * object_pts[i].z + t_gt[1];
        double zc = R_gt(2, 0) * object_pts[i].x + R_gt(2, 1) * object_pts[i].y + R_gt(2, 2) * object_pts[i].z + t_gt[2];

        // Project
        image_pts[i].x = (K(0, 0) * xc + K(0, 2) * zc) / zc;
        image_pts[i].y = (K(1, 1) * yc + K(1, 2) * zc) / zc;
    }

    // Solve PnP
    math::Matrix R_est;
    std::vector<double> t_est;
    solve_pnp_dlt(object_pts, image_pts, K, R_est, t_est);

    // Verify rotation matrix R
    for (size_t r = 0; r < 3; ++r) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_NEAR(R_est(r, col), R_gt(r, col), 1e-4);
        }
    }

    // Verify translation vector t
    EXPECT_NEAR(t_est[0], t_gt[0], 1e-3);
    EXPECT_NEAR(t_est[1], t_gt[1], 1e-3);
    EXPECT_NEAR(t_est[2], t_gt[2], 1e-3);
}

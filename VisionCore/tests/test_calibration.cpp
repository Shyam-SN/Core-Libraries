#include <gtest/gtest.h>
#include <visioncore/algorithms/calibration.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

// Helper to construct a rotation matrix around Y and X axis
math::Matrix get_rotation_matrix(double angle_x, double angle_y) {
    double cx = std::cos(angle_x);
    double sx = std::sin(angle_x);
    double cy = std::cos(angle_y);
    double sy = std::sin(angle_y);

    math::Matrix Rx(3, 3, 0.0);
    Rx(0, 0) = 1.0;
    Rx(1, 1) = cx; Rx(1, 2) = -sx;
    Rx(2, 1) = sx; Rx(2, 2) = cx;

    math::Matrix Ry(3, 3, 0.0);
    Ry(0, 0) = cy; Ry(0, 2) = sy;
    Ry(1, 1) = 1.0;
    Ry(2, 0) = -sy; Ry(2, 2) = cy;

    return Ry * Rx;
}

TEST(CalibrationTest, ZhangsMethodCalibration) {
    // 1. Define ground truth intrinsics K
    // focal_x = 800, focal_y = 800, u0 = 320, v0 = 240, skew = 0.5
    math::Matrix K_gt(3, 3, 0.0);
    K_gt(0, 0) = 800.0; K_gt(0, 1) = 0.5;   K_gt(0, 2) = 320.0;
    K_gt(1, 1) = 800.0; K_gt(1, 2) = 240.0;
    K_gt(2, 2) = 1.0;

    // 2. Generate 4 different views of the calibration plane (extrinsic poses)
    // Angles in radians, translations with z > 0
    std::vector<math::Matrix> homographies;
    
    struct Pose {
        double rx, ry, tx, ty, tz;
    };
    
    std::vector<Pose> poses = {
        { 0.1, -0.15, -0.2, 0.1, 1.5 },
        {-0.08, 0.12, 0.1, -0.1, 1.8 },
        { 0.15, 0.05, -0.15, 0.2, 2.0 },
        {-0.05, -0.1, 0.05, 0.05, 1.4 }
    };

    for (const auto& pose : poses) {
        math::Matrix R = get_rotation_matrix(pose.rx, pose.ry);
        
        // H = K * [r1, r2, t]
        math::Matrix H(3, 3);
        
        // Column 0: K * r1
        H(0, 0) = K_gt(0, 0) * R(0, 0) + K_gt(0, 1) * R(1, 0) + K_gt(0, 2) * R(2, 0);
        H(1, 0) = K_gt(1, 1) * R(1, 0) + K_gt(1, 2) * R(2, 0);
        H(2, 0) = R(2, 0);

        // Column 1: K * r2
        H(0, 1) = K_gt(0, 0) * R(0, 1) + K_gt(0, 1) * R(1, 1) + K_gt(0, 2) * R(2, 1);
        H(1, 1) = K_gt(1, 1) * R(1, 1) + K_gt(1, 2) * R(2, 1);
        H(2, 1) = R(2, 1);

        // Column 2: K * t
        H(0, 2) = K_gt(0, 0) * pose.tx + K_gt(0, 1) * pose.ty + K_gt(0, 2) * pose.tz;
        H(1, 2) = K_gt(1, 1) * pose.ty + K_gt(1, 2) * pose.tz;
        H(2, 2) = pose.tz;

        // Randomly scale homography to verify scale-invariance of solver
        double random_scale = 2.5;
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                H(r, c) *= random_scale;
            }
        }

        homographies.push_back(H);
    }

    // 3. Perform calibration
    math::Matrix K_est = calibrate_camera(homographies);

    // 4. Verify recovery of K elements
    EXPECT_NEAR(K_est(0, 0), K_gt(0, 0), 1e-3); // fx
    EXPECT_NEAR(K_est(1, 1), K_gt(1, 1), 1e-3); // fy
    EXPECT_NEAR(K_est(0, 1), K_gt(0, 1), 1e-3); // skew
    EXPECT_NEAR(K_est(0, 2), K_gt(0, 2), 1e-3); // u0
    EXPECT_NEAR(K_est(1, 2), K_gt(1, 2), 1e-3); // v0
    EXPECT_NEAR(K_est(2, 2), 1.0, 1e-7);
}

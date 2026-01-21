#include <gtest/gtest.h>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/bundle_adjustment.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(BundleAdjustmentTest, LocalLMRefinement) {
    // 1. Intrinsics matrix K
    math::Matrix K(3, 3);
    K(0, 0) = 800.0; K(0, 1) = 0.0;   K(0, 2) = 320.0;
    K(1, 0) = 0.0;   K(1, 1) = 800.0; K(1, 2) = 240.0;
    K(2, 0) = 0.0;   K(2, 1) = 0.0;   K(2, 2) = 1.0;

    // Ground truth camera 2 pose (rotation + translation)
    math::Matrix R2_gt = euler_to_rotation(0.05, -0.04, 0.03);
    std::vector<double> t2_gt = { 1.0, -0.5, 8.0 };

    // Ground truth 3D points
    std::vector<Point3D> pts3d_gt = {
        Point3D(-1.5, -1.0, 5.0),
        Point3D(1.5, -1.0, 6.0),
        Point3D(1.5, 1.2, 5.5),
        Point3D(-1.2, 1.0, 5.8)
    };

    const size_t N = pts3d_gt.size();

    // Generate observations (perfect 2D projections)
    std::vector<Point2D> obs1(N);
    std::vector<Point2D> obs2(N);

    math::Matrix R1 = math::Matrix::identity(3);
    std::vector<double> t1 = { 0.0, 0.0, 0.0 };

    for (size_t i = 0; i < N; ++i) {
        obs1[i] = project_point(pts3d_gt[i], K, R1, t1);
        obs2[i] = project_point(pts3d_gt[i], K, R2_gt, t2_gt);
    }

    // Add noise to create initial estimates
    std::vector<Point3D> pts3d_est = pts3d_gt;
    for (size_t i = 0; i < N; ++i) {
        pts3d_est[i].x += 0.15;
        pts3d_est[i].y -= 0.12;
        pts3d_est[i].z += 0.20;
    }

    math::Matrix R2_est = euler_to_rotation(0.07, -0.03, 0.04);
    std::vector<double> t2_est = { 1.1, -0.4, 8.2 };

    // 2. Measure initial reprojection error
    double initial_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        Point2D p1 = project_point(pts3d_est[i], K, R1, t1);
        Point2D p2 = project_point(pts3d_est[i], K, R2_est, t2_est);
        
        double e1x = p1.x - obs1[i].x;
        double e1y = p1.y - obs1[i].y;
        double e2x = p2.x - obs2[i].x;
        double e2y = p2.y - obs2[i].y;

        initial_err += (e1x*e1x + e1y*e1y + e2x*e2x + e2y*e2y);
    }

    // 3. Optimize using LM Bundle Adjustment
    bundle_adjustment(K, obs1, obs2, R2_est, t2_est, pts3d_est, 30);

    // 4. Measure final reprojection error
    double final_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        Point2D p1 = project_point(pts3d_est[i], K, R1, t1);
        Point2D p2 = project_point(pts3d_est[i], K, R2_est, t2_est);
        
        double e1x = p1.x - obs1[i].x;
        double e1y = p1.y - obs1[i].y;
        double e2x = p2.x - obs2[i].x;
        double e2y = p2.y - obs2[i].y;

        final_err += (e1x*e1x + e1y*e1y + e2x*e2x + e2y*e2y);
    }

    // Verify error is significantly reduced
    EXPECT_LT(final_err, initial_err);
    EXPECT_NEAR(final_err, 0.0, 1e-3);

    // Verify optimized parameters match ground truth closely (within gauge ambiguity shift)
    EXPECT_NEAR(t2_est[0], t2_gt[0], 0.3);
    EXPECT_NEAR(t2_est[1], t2_gt[1], 0.3);
    EXPECT_NEAR(t2_est[2], t2_gt[2], 0.3);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pts3d_est[i].x, pts3d_gt[i].x, 0.3);
        EXPECT_NEAR(pts3d_est[i].y, pts3d_gt[i].y, 0.3);
        EXPECT_NEAR(pts3d_est[i].z, pts3d_gt[i].z, 0.3);
    }
}

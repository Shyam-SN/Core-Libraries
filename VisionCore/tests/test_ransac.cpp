#include <gtest/gtest.h>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/ransac.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(RansacTest, RobustHomographyFitting) {
    // 1. Ground truth homography matrix
    math::Matrix H_gt(3, 3);
    H_gt(0, 0) = 1.1;   H_gt(0, 1) = 0.2;   H_gt(0, 2) = 15.0;
    H_gt(1, 0) = -0.1;  H_gt(1, 1) = 0.9;   H_gt(1, 2) = -8.0;
    H_gt(2, 0) = 0.0005;H_gt(2, 1) = -0.001;H_gt(2, 2) = 1.0;

    const size_t num_inliers = 15;
    const size_t num_outliers = 5;
    const size_t total_points = num_inliers + num_outliers;

    std::vector<Point2D> src(total_points);
    std::vector<Point2D> dst(total_points);

    // 2. Generate clean inliers (non-collinear)
    for (size_t i = 0; i < num_inliers; ++i) {
        src[i] = Point2D(i * 10.0 + 5.0, (i * i) * 0.5 + i * 8.0 + 12.0);
        double w = H_gt(2, 0) * src[i].x + H_gt(2, 1) * src[i].y + H_gt(2, 2);
        dst[i].x = (H_gt(0, 0) * src[i].x + H_gt(0, 1) * src[i].y + H_gt(0, 2)) / w;
        dst[i].y = (H_gt(1, 0) * src[i].x + H_gt(1, 1) * src[i].y + H_gt(1, 2)) / w;
    }

    // 3. Generate corrupt outliers
    for (size_t i = num_inliers; i < total_points; ++i) {
        src[i] = Point2D(i * 10.0, (i * i) * 0.5 + i * 8.0);
        dst[i] = Point2D(-999.0, 999.0); // Extreme outliers
    }

    std::vector<uint8_t> inliers;
    math::Matrix H_est = find_homography_ransac(src, dst, inliers, 3.0, 500);

    // 4. Verify inliers mask
    // All first 15 points should be marked as inliers (1)
    for (size_t i = 0; i < num_inliers; ++i) {
        EXPECT_EQ(inliers[i], 1);
    }
    // All subsequent 5 points should be marked as outliers (0)
    for (size_t i = num_inliers; i < total_points; ++i) {
        EXPECT_EQ(inliers[i], 0);
    }

    // 5. Verify estimated matrix accuracy
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(H_est(r, c), H_gt(r, c), 1e-2);
        }
    }
}

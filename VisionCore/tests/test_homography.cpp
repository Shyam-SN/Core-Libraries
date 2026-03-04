#include <gtest/gtest.h>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/homography.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(HomographyTest, DltEstimation) {
    // Define a ground-truth homography matrix (representing rotation + translation + scaling)
    // H = [ 1.2   -0.3   10.0 ]
    //     [ 0.3    1.1  -20.0 ]
    //     [ 0.001 -0.002  1.0 ]
    math::Matrix H_gt(3, 3);
    H_gt(0, 0) = 1.2;   H_gt(0, 1) = -0.3;  H_gt(0, 2) = 10.0;
    H_gt(1, 0) = 0.3;   H_gt(1, 1) = 1.1;   H_gt(1, 2) = -20.0;
    H_gt(2, 0) = 0.001; H_gt(2, 1) = -0.002;H_gt(2, 2) = 1.0;

    // Define 5 source points (must be at least 4, not collinear)
    std::vector<Point2D> src = {
        Point2D(0.0, 0.0),
        Point2D(100.0, 0.0),
        Point2D(100.0, 100.0),
        Point2D(0.0, 100.0),
        Point2D(50.0, 50.0)
    };

    // Project points using H_gt
    std::vector<Point2D> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        double w = H_gt(2, 0) * src[i].x + H_gt(2, 1) * src[i].y + H_gt(2, 2);
        dst[i].x = (H_gt(0, 0) * src[i].x + H_gt(0, 1) * src[i].y + H_gt(0, 2)) / w;
        dst[i].y = (H_gt(1, 0) * src[i].x + H_gt(1, 1) * src[i].y + H_gt(1, 2)) / w;
    }

    // Recover homography matrix using DLT solver
    math::Matrix H_est = find_homography(src, dst);

    // Verify H_est matches H_gt
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(H_est(r, c), H_gt(r, c), 1e-4);
        }
    }
}

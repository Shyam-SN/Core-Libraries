#include <gtest/gtest.h>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/epipolar.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(EpipolarTest, FundamentalMatrixRecovery) {
    // Ground-truth F matrix
    // F = [  0.0   -0.2   0.3 ]
    //     [  0.2    0.0  -0.1 ]
    //     [ -0.3    0.1   0.0 ] (skew-symmetric to mock cross product of translation)
    math::Matrix F_gt(3, 3, 0.0);
    F_gt(0, 1) = -0.2;  F_gt(0, 2) = 0.3;
    F_gt(1, 0) = 0.2;   F_gt(1, 2) = -0.1;
    F_gt(2, 0) = -0.3;  F_gt(2, 1) = 0.1;

    // Generate 8 random point correspondences that satisfy x2^T * F * x1 = 0
    std::vector<Point2D> pts1 = {
        Point2D(10.0, 20.0),
        Point2D(150.0, 30.0),
        Point2D(50.0, 120.0),
        Point2D(30.0, 150.0),
        Point2D(120.0, 90.0),
        Point2D(90.0, 60.0),
        Point2D(80.0, 40.0),
        Point2D(40.0, 80.0)
    };

    std::vector<Point2D> pts2(8);
    for (size_t i = 0; i < 8; ++i) {
        // Epipolar line in image 2: l2 = F * x1
        double l2_a = F_gt(0, 0) * pts1[i].x + F_gt(0, 1) * pts1[i].y + F_gt(0, 2);
        double l2_b = F_gt(1, 0) * pts1[i].x + F_gt(1, 1) * pts1[i].y + F_gt(1, 2);
        double l2_c = F_gt(2, 0) * pts1[i].x + F_gt(2, 1) * pts1[i].y + F_gt(2, 2);

        // Find a point pts2[i] on the line: a * x + b * y + c = 0
        // Set y = 50.0 (arbitrary choice)
        double y2 = 50.0;
        double x2 = (-l2_c - l2_b * y2) / l2_a;
        pts2[i] = Point2D(x2, y2);
    }

    // Recover Fundamental matrix using normalized 8-point solver
    math::Matrix F_est = find_fundamental(pts1, pts2);

    // Verify epipolar constraint x2^T * F_est * x1 == 0 for all points
    for (size_t i = 0; i < 8; ++i) {
        double val = pts2[i].x * (F_est(0, 0) * pts1[i].x + F_est(0, 1) * pts1[i].y + F_est(0, 2)) +
                     pts2[i].y * (F_est(1, 0) * pts1[i].x + F_est(1, 1) * pts1[i].y + F_est(1, 2)) +
                     1.0 * (F_est(2, 0) * pts1[i].x + F_est(2, 1) * pts1[i].y + F_est(2, 2));

        EXPECT_NEAR(val, 0.0, 1e-4);
    }
}

TEST(EpipolarTest, EssentialMatrixProjection) {
    // Generate a mock Essential matrix (arbitrary values)
    math::Matrix E(3, 3);
    E(0, 0) = 1.0; E(0, 1) = 2.0; E(0, 2) = 1.5;
    E(1, 0) = 0.5; E(1, 1) = 3.0; E(1, 2) = -0.5;
    E(2, 0) = 2.0; E(2, 1) = 1.0; E(2, 2) = 1.0;

    math::Matrix E_proj;
    project_essential(E, E_proj);

    // Verify singular values of E_proj are (s, s, 0)
    math::Matrix U;
    std::vector<double> S;
    math::Matrix V;
    math::svd(E_proj, U, S, V);

    EXPECT_NEAR(S[0], S[1], 1e-6);
    EXPECT_NEAR(S[2], 0.0, 1e-6);
}

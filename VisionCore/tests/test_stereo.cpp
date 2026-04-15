#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/stereo.hpp>
#include <vector>

using namespace visioncore;

TEST(StereoTest, TriangulatePoints) {
    // Left camera matrix: projection at origin
    // P1 = [ 1  0  0  0 ]
    //      [ 0  1  0  0 ]
    //      [ 0  0  1  0 ]
    math::Matrix P1(3, 4, 0.0);
    P1(0, 0) = 1.0; P1(1, 1) = 1.0; P1(2, 2) = 1.0;

    // Right camera matrix: translated along X axis by -1.0 unit
    // P2 = [ 1  0  0  -1 ]
    //      [ 0  1  0   0 ]
    //      [ 0  0  1   0 ]
    math::Matrix P2(3, 4, 0.0);
    P2(0, 0) = 1.0; P2(1, 1) = 1.0; P2(2, 2) = 1.0;
    P2(0, 3) = -1.0;

    // Let ground truth 3D point be (2.0, 3.0, 5.0)
    Point3D pt3d_gt(2.0, 3.0, 5.0);

    // Project to camera 1 (divided by Z = 5.0)
    // u1 = 2.0 / 5.0 = 0.4, v1 = 3.0 / 5.0 = 0.6
    std::vector<Point2D> pts1 = { Point2D(0.4, 0.6) };

    // Project to camera 2 (u2 = (2.0 - 1.0) / 5.0 = 0.2, v2 = 3.0 / 5.0 = 0.6)
    std::vector<Point2D> pts2 = { Point2D(0.2, 0.6) };

    std::vector<Point3D> pts3d;
    triangulate_points(P1, P2, pts1, pts2, pts3d);

    EXPECT_EQ(pts3d.size(), 1);
    EXPECT_NEAR(pts3d[0].x, pt3d_gt.x, 1e-4);
    EXPECT_NEAR(pts3d[0].y, pt3d_gt.y, 1e-4);
    EXPECT_NEAR(pts3d[0].z, pt3d_gt.z, 1e-4);
}

TEST(StereoTest, DisparityBlockMatching) {
    // Generate left and right images
    Image<uint8_t, 1> left(20, 20, 100);
    Image<uint8_t, 1> right(20, 20, 100);

    // Left square at (10, 10) to (12, 12)
    for (size_t y = 10; y < 13; ++y) {
        for (size_t x = 10; x < 13; ++x) {
            left(x, y) = 250;
        }
    }

    // Right square shifted by 4 pixels left: (6, 10) to (8, 12)
    for (size_t y = 10; y < 13; ++y) {
        for (size_t x = 6; x < 9; ++x) {
            right(x, y) = 250;
        }
    }

    Image<uint8_t, 1> disp(20, 20);
    compute_disparity_sad(left.const_view(), right.const_view(), disp.view(), 0, 8, 3);

    // Check disparity at center of the block (11, 11) -> should be 4
    EXPECT_EQ(disp(11, 11), 4);
}

#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/optical_flow.hpp>
#include <vector>

using namespace visioncore;

TEST(OpticalFlowTest, BilinearInterpolation) {
    Image<uint8_t, 1> img(2, 2);
    img(0, 0) = 10;
    img(1, 0) = 20;
    img(0, 1) = 30;
    img(1, 1) = 40;

    // Center of 4 pixels (0.5, 0.5) should be average: (10 + 20 + 30 + 40) / 4 = 25
    float val = interpolate_bilinear(img.const_view(), 0.5f, 0.5f);
    EXPECT_NEAR(val, 25.0f, 1e-4f);

    // Coordinate exactly on (0, 0) should be 10
    EXPECT_NEAR(interpolate_bilinear(img.const_view(), 0.0f, 0.0f), 10.0f, 1e-4f);
    // Coordinate exactly on (1, 1) should be 40
    EXPECT_NEAR(interpolate_bilinear(img.const_view(), 1.0f, 1.0f), 40.0f, 1e-4f);
}

TEST(OpticalFlowTest, LucasKanadeTracking) {
    // Generate frame 1: box centered at (15, 15)
    Image<uint8_t, 1> prev_img(30, 30, 50);
    for (size_t y = 10; y < 20; ++y) {
        for (size_t x = 10; x < 20; ++x) {
            prev_img(x, y) = 200;
        }
    }

    // Generate frame 2: box shifted by dx=1.0, dy=1.0
    Image<uint8_t, 1> next_img(30, 30, 50);
    for (size_t y = 11; y < 21; ++y) {
        for (size_t x = 11; x < 21; ++x) {
            next_img(x, y) = 200;
        }
    }

    // Keypoints at corners in frame 1 (e.g. 10, 10)
    std::vector<Keypoint> prev_pts = {
        Keypoint(10.0f, 10.0f),
        Keypoint(19.0f, 10.0f)
    };

    std::vector<Keypoint> next_pts;
    std::vector<uint8_t> status;

    calc_optical_flow_lk(prev_img.const_view(), next_img.const_view(), prev_pts, next_pts, status, 7, 40);

    EXPECT_EQ(status.size(), 2);
    EXPECT_EQ(status[0], 1);
    EXPECT_EQ(status[1], 1);

    // Flow update should be close to (1.0, 1.0)
    EXPECT_NEAR(next_pts[0].x - prev_pts[0].x, 1.0f, 0.1f);
    EXPECT_NEAR(next_pts[0].y - prev_pts[0].y, 1.0f, 0.1f);
    EXPECT_NEAR(next_pts[1].x - prev_pts[1].x, 1.0f, 0.1f);
    EXPECT_NEAR(next_pts[1].y - prev_pts[1].y, 1.0f, 0.1f);
}

#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/corner_detection.hpp>
#include <vector>

using namespace visioncore;

TEST(CornerDetectionTest, HarrisCorners) {
    // Create an image with a solid square in the middle (creating 4 corners)
    Image<uint8_t, 1> src(20, 20, 0);
    for (size_t y = 5; y <= 15; ++y) {
        for (size_t x = 5; x <= 15; ++x) {
            src(x, y) = 255;
        }
    }

    std::vector<Keypoint> kps;
    detect_harris(src.const_view(), kps, 0.04, 1e4, 3, BorderType::ZERO);

    // Should find at least the 4 corners of the square (at coordinates roughly around 5 and 15)
    EXPECT_GE(kps.size(), 4);

    bool top_left = false, top_right = false, bottom_left = false, bottom_right = false;
    for (const auto& kp : kps) {
        if (std::abs(kp.x - 5) <= 1 && std::abs(kp.y - 5) <= 1) top_left = true;
        if (std::abs(kp.x - 15) <= 1 && std::abs(kp.y - 5) <= 1) top_right = true;
        if (std::abs(kp.x - 5) <= 1 && std::abs(kp.y - 15) <= 1) bottom_left = true;
        if (std::abs(kp.x - 15) <= 1 && std::abs(kp.y - 15) <= 1) bottom_right = true;
    }

    EXPECT_TRUE(top_left);
    EXPECT_TRUE(top_right);
    EXPECT_TRUE(bottom_left);
    EXPECT_TRUE(bottom_right);
}

TEST(CornerDetectionTest, FASTCorners) {
    // Create an image with a cross intersection
    Image<uint8_t, 1> src(20, 20, 50);
    // Draw vertical and horizontal bars starting from 10, 10 (corner at 10, 10)
    for (size_t i = 10; i < 20; ++i) {
        src(10, i) = 200;
        src(i, 10) = 200;
    }

    std::vector<Keypoint> kps;
    detect_fast(src.const_view(), kps, 30, true);

    // Should detect the corner intersection at (10, 10)
    bool intersection_detected = false;
    for (const auto& kp : kps) {
        if (std::abs(kp.x - 10.0f) <= 1.0f && std::abs(kp.y - 10.0f) <= 1.0f) {
            intersection_detected = true;
            break;
        }
    }
    EXPECT_TRUE(intersection_detected);
}

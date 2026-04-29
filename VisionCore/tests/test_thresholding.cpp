#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/thresholding.hpp>

using namespace visioncore;

TEST(ThresholdingTest, BinaryThreshold) {
    Image<uint8_t, 1> src(5, 5);
    // Fill with rising values
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            src(x, y) = static_cast<uint8_t>(y * 5 + x) * 10; // 0 to 240
        }
    }

    Image<uint8_t, 1> dst(5, 5, 0);
    threshold_binary(src.const_view(), dst.view(), 120, 255);

    // Pixels >= 120 should be 255, < 120 should be 0
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            uint8_t val = src(x, y);
            if (val >= 120) {
                EXPECT_EQ(dst(x, y), 255);
            } else {
                EXPECT_EQ(dst(x, y), 0);
            }
        }
    }
}

TEST(ThresholdingTest, OtsuThreshold) {
    // Generate a bi-modal image (cluster at 50, cluster at 200)
    Image<uint8_t, 1> src(20, 20);
    for (size_t y = 0; y < 20; ++y) {
        for (size_t x = 0; x < 20; ++x) {
            if (x < 10) {
                src(x, y) = 50;
            } else {
                src(x, y) = 200;
            }
        }
    }

    // The optimal threshold should be exactly between 50 and 200 (e.g. around 125)
    uint8_t optimal_t = compute_otsu_threshold(src.const_view());
    EXPECT_GT(optimal_t, 50);
    EXPECT_LT(optimal_t, 200);

    Image<uint8_t, 1> dst(20, 20, 0);
    threshold_otsu(src.const_view(), dst.view(), 255);

    // Verify background (50) is 0 and foreground (200) is 255
    for (size_t y = 0; y < 20; ++y) {
        for (size_t x = 0; x < 20; ++x) {
            if (x < 10) {
                EXPECT_EQ(dst(x, y), 0);
            } else {
                EXPECT_EQ(dst(x, y), 255);
            }
        }
    }
}

TEST(ThresholdingTest, AdaptiveThreshold) {
    // Generate an image with a local step edge on top of a diagonal illumination gradient
    // Gradient: intensity climbs from top-left to bottom-right
    Image<uint8_t, 1> src(15, 15);
    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 0; x < 15; ++x) {
            double gradient = (x + y) * 5.0; // 0 to 140
            double local_edge = (x >= 8) ? 40.0 : 0.0;
            src(x, y) = static_cast<uint8_t>(std::clamp(gradient + local_edge, 0.0, 255.0));
        }
    }

    Image<uint8_t, 1> dst(15, 15, 0);
    // Adaptive thresholding with window size 5, offset 5
    threshold_adaptive(src.const_view(), dst.view(), 5, 5.0, AdaptiveMethod::MEAN, 255);

    // Verify that the local step edge at column 8 is successfully detected as foreground (255)
    // even though the absolute intensity at column 8 on the left is lower than the absolute
    // intensity at column 14 on the right.
    // E.g., at (7, 7), src is 70, local mean is ~74. src < local_mean - 5 => 0
    // At (8, 7), src is 115, local mean is ~98. src >= local_mean - 5 => 255
    EXPECT_EQ(dst(7, 7), 0);
    EXPECT_EQ(dst(8, 7), 255);
}

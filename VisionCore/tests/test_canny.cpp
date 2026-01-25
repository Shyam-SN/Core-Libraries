#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/canny.hpp>

using namespace visioncore;

TEST(CannyTest, StepEdgeDetection) {
    Image<uint8_t, 1> src(15, 15, 0);
    // Draw a sharp vertical step edge down the center
    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 8; x < 15; ++x) {
            src(x, y) = 255;
        }
    }

    Image<uint8_t, 1> dst(15, 15, 0);
    // Upper threshold = 100, Lower = 50
    canny(src.const_view(), dst.view(), 50.0, 100.0, 3, 0.0);

    // The edge should be detected exactly along the boundary of the transition
    // (specifically around column 7 or 8)
    bool edge_found = false;
    for (size_t y = 1; y < 14; ++y) {
        // Edge should be localized horizontally
        EXPECT_EQ(dst(7, y), 255);
        edge_found = true;

        // Neighboring columns should be zero (suppressed to single pixel)
        EXPECT_EQ(dst(5, y), 0);
        EXPECT_EQ(dst(9, y), 0);
    }
    EXPECT_TRUE(edge_found);
}

TEST(CannyTest, LowContrastSuppression) {
    Image<uint8_t, 1> src(15, 15, 100);
    // Add minor, low-contrast noise (amplitude 5)
    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 0; x < 15; ++x) {
            if ((x + y) % 2 == 0) {
                src(x, y) = 105;
            }
        }
    }

    Image<uint8_t, 1> dst(15, 15, 0);
    // High threshold of 80 should completely suppress this low-amplitude noise
    canny(src.const_view(), dst.view(), 30.0, 80.0, 3, 1.0);

    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 0; x < 15; ++x) {
            EXPECT_EQ(dst(x, y), 0);
        }
    }
}

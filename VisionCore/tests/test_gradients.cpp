#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/gradients.hpp>
#include <cmath>

using namespace visioncore;

TEST(GradientsTest, PrewittGradients) {
    Image<uint8_t, 1> src(5, 5, 0);
    // Draw a step edge (0 to 100)
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 2; x < 5; ++x) {
            src(x, y) = 100;
        }
    }

    Image<float, 1> dx(5, 5, 0.0f);
    Image<float, 1> dy(5, 5, 0.0f);

    prewitt_x(src.const_view(), dx.view(), BorderType::REPLICATE);
    prewitt_y(src.const_view(), dy.view(), BorderType::REPLICATE);

    // Prewitt X on step edge at x=2 should be positive
    // Column 2 value is 100, neighboring columns are 0 (left) and 100 (right).
    // dx at (2,2) should be sum over ky of: col(3) - col(1) = (100 - 0) * 1 = 100 per row -> total 300.
    EXPECT_GT(dx(2, 2), 0.0f);
    EXPECT_DOUBLE_EQ(dy(2, 2), 0.0); // Y derivative should be exactly 0
}

TEST(GradientsTest, ScharrGradients) {
    Image<uint8_t, 1> src(5, 5, 0);
    // Draw a diagonal step edge
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            if (x >= y) src(x, y) = 100;
        }
    }

    Image<float, 1> dx(5, 5, 0.0f);
    Image<float, 1> dy(5, 5, 0.0f);

    scharr_x(src.const_view(), dx.view(), BorderType::REPLICATE);
    scharr_y(src.const_view(), dy.view(), BorderType::REPLICATE);

    // Both X and Y gradients should be active
    EXPECT_NE(dx(2, 2), 0.0f);
    EXPECT_NE(dy(2, 2), 0.0f);
}

TEST(GradientsTest, LaplacianDerivatives) {
    Image<uint8_t, 1> src(5, 5, 0);
    src(2, 2) = 100; // Single isolated peak

    Image<float, 1> lap(5, 5, 0.0f);
    laplacian(src.const_view(), lap.view(), BorderType::ZERO);

    // Center pixel response should be strongly negative (100 * -8 = -800)
    EXPECT_EQ(lap(2, 2), -800.0f);
    // Direct neighbors should be positive (100 * 1 = 100)
    EXPECT_EQ(lap(1, 2), 100.0f);
    EXPECT_EQ(lap(2, 1), 100.0f);
}

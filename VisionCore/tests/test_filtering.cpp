#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/filtering.hpp>

using namespace visioncore;

TEST(FilteringTest, BoxFilterAllOnes) {
    Image<uint8_t, 1> src(5, 5, 255);
    Image<uint8_t, 1> dst(5, 5, 0);

    box_filter(src.view(), dst.view(), 3, BorderType::REPLICATE);

    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            EXPECT_EQ(dst(x, y), 255);
        }
    }
}

TEST(FilteringTest, GaussianBlurKernelGeneration) {
    auto kernel_3 = generate_gaussian_kernel1d(3, 1.0);
    EXPECT_EQ(kernel_3.size(), 3);
    
    double sum = 0.0;
    for (double val : kernel_3) {
        sum += val;
    }
    // Normalized sum must be exactly 1
    EXPECT_DOUBLE_EQ(sum, 1.0);
    // Gaussian must be symmetric
    EXPECT_DOUBLE_EQ(kernel_3[0], kernel_3[2]);
}

TEST(FilteringTest, SobelStepEdge) {
    // 5x5 image with a vertical step edge in the middle
    // 0 0 255 255 255
    Image<uint8_t, 1> src(5, 5, 0);
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 2; x < 5; ++x) {
            src(x, y) = 255;
        }
    }

    // Sobel gradients can have values larger than 255 or negative, so we use float output type
    Image<float, 1> dx(5, 5, 0.0f);
    Image<float, 1> dy(5, 5, 0.0f);

    sobel_x(src.view(), dx.view(), BorderType::REPLICATE);
    sobel_y(src.view(), dy.view(), BorderType::REPLICATE);

    // Sobel X should detect the vertical edge (it should have a strong positive derivative at the transition column)
    // Sobel Y should be zero (or near zero, since there's no vertical changes)
    for (size_t y = 0; y < 5; ++y) {
        // Col 1 to 2 transition: Sobel X is positive
        EXPECT_GT(dx(2, y), 0.0f);
        // Sobel Y is zero
        EXPECT_NEAR(dy(2, y), 0.0f, 1e-5);
    }
}

TEST(FilteringTest, SobelMagnitudeIntegration) {
    Image<float, 1> dx(5, 5, 0.0f);
    Image<float, 1> dy(5, 5, 0.0f);
    Image<float, 1> mag(5, 5, 0.0f);

    // Set sample gradients
    dx(2, 2) = 30.0f;
    dy(2, 2) = 40.0f;

    sobel_magnitude(dx.view(), dy.view(), mag.view());

    // hypot(30, 40) = 50
    EXPECT_FLOAT_EQ(mag(2, 2), 50.0f);
}

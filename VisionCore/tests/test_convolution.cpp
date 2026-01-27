#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/convolution.hpp>
#include <vector>

using namespace visioncore;

TEST(ConvolutionTest, IdentityKernel) {
    Image<uint8_t, 1> src(5, 5, 50);
    Image<uint8_t, 1> dst(5, 5, 0);

    // 3x3 Identity kernel
    std::vector<double> identity = {
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0
    };

    convolve2d(src.view(), dst.view(), identity, 3, 3, BorderType::REPLICATE);

    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            EXPECT_EQ(dst(x, y), 50);
        }
    }
}

TEST(ConvolutionTest, ShiftKernel) {
    Image<uint8_t, 1> src(3, 3, 0);
    // Set middle pixel to 100
    src(1, 1) = 100;

    // Kernel that shifts pixel to the right
    // [0 0 0]
    // [1 0 0]
    // [0 0 0]
    std::vector<double> shift_right = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    Image<uint8_t, 1> dst(3, 3, 0);
    convolve2d(src.view(), dst.view(), shift_right, 3, 3, BorderType::ZERO);

    // The pixel at (1, 1) shifted right goes to (2, 1)
    EXPECT_EQ(dst(2, 1), 100);
    EXPECT_EQ(dst(1, 1), 0);
}

TEST(ConvolutionTest, BoundaryZero) {
    Image<uint8_t, 1> src(3, 3, 100);
    Image<uint8_t, 1> dst(3, 3, 0);

    // Standard 3x3 Box Blur
    std::vector<double> box = std::vector<double>(9, 1.0 / 9.0);

    convolve2d(src.view(), dst.view(), box, 3, 3, BorderType::ZERO);

    // At the corner (0, 0), the 3x3 window overlaps with 4 valid pixels and 5 out-of-bounds (zero) pixels.
    // So output should be 100 * (4/9) = ~44
    EXPECT_EQ(dst(0, 0), static_cast<uint8_t>(100.0 * 4.0 / 9.0));
}

TEST(ConvolutionTest, BoundaryReplicate) {
    Image<uint8_t, 1> src(3, 3, 100);
    Image<uint8_t, 1> dst(3, 3, 0);

    // Standard 3x3 Box Blur
    std::vector<double> box = std::vector<double>(9, 1.0 / 9.0);

    convolve2d(src.view(), dst.view(), box, 3, 3, BorderType::REPLICATE);

    // With replication, every pixel in the window gets filled with 100.
    // So output should be exactly 100.
    EXPECT_EQ(dst(0, 0), 100);
}

TEST(ConvolutionTest, SeparableVsDirect) {
    Image<uint8_t, 1> src(15, 15);
    // Fill with pattern
    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 0; x < 15; ++x) {
            src(x, y) = static_cast<uint8_t>((x * 3 + y * 7) % 256);
        }
    }

    // A 3x3 box blur is separable: row = [1/3, 1/3, 1/3], col = [1/3, 1/3, 1/3]
    std::vector<double> direct_kernel(9, 1.0 / 9.0);
    std::vector<double> row_kernel(3, 1.0 / 3.0);
    std::vector<double> col_kernel(3, 1.0 / 3.0);

    Image<uint8_t, 1> dst_direct(15, 15, 0);
    Image<uint8_t, 1> dst_separable(15, 15, 0);

    convolve2d(src.view(), dst_direct.view(), direct_kernel, 3, 3, BorderType::REFLECT);
    convolve_separable(src.view(), dst_separable.view(), row_kernel, col_kernel, BorderType::REFLECT);

    for (size_t y = 0; y < 15; ++y) {
        for (size_t x = 0; x < 15; ++x) {
            EXPECT_EQ(dst_direct(x, y), dst_separable(x, y))
                << "Mismatch at (" << x << ", " << y << ")";
        }
    }
}

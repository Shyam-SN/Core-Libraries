#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/histogram.hpp>
#include <vector>

using namespace visioncore;

TEST(HistogramTest, ComputeHistogram) {
    Image<uint8_t, 1> img(4, 4);
    // Set explicit values
    img(0, 0) = 10;  img(1, 0) = 20;  img(2, 0) = 10;  img(3, 0) = 30;
    img(0, 1) = 10;  img(1, 1) = 40;  img(2, 1) = 20;  img(3, 1) = 30;
    img(0, 2) = 50;  img(1, 2) = 50;  img(2, 2) = 10;  img(3, 2) = 20;
    img(0, 3) = 10;  img(1, 3) = 10;  img(2, 3) = 30;  img(3, 3) = 40;

    auto hist = compute_histogram(img.const_view());

    EXPECT_EQ(hist[10], 6); // Six 10s
    EXPECT_EQ(hist[20], 3); // Three 20s
    EXPECT_EQ(hist[30], 3); // Three 30s
    EXPECT_EQ(hist[40], 2); // Two 40s
    EXPECT_EQ(hist[50], 2); // Two 50s
    EXPECT_EQ(hist[0], 0);  // Zero for other bins
}

TEST(HistogramTest, EqualizeHistogram) {
    // Generate a flat, low-contrast image (values clustered between 100 and 105)
    Image<uint8_t, 1> src(100, 100);
    for (size_t y = 0; y < 100; ++y) {
        for (size_t x = 0; x < 100; ++x) {
            src(x, y) = static_cast<uint8_t>(100 + (x % 6));
        }
    }

    Image<uint8_t, 1> dst(100, 100);
    equalize_histogram(src.const_view(), dst.view());

    // Compute histograms of source and destination
    auto hist_src = compute_histogram(src.const_view());
    auto hist_dst = compute_histogram(dst.const_view());

    // Source values are clustered in [100, 105]
    size_t count_src = 0;
    for (size_t i = 100; i <= 105; ++i) {
        count_src += hist_src[i];
    }
    EXPECT_EQ(count_src, 10000);

    // Equalized image should spread values out across the 0-255 spectrum
    // Let's verify that the minimum value is mapped to 0 and the maximum is mapped to 255
    uint8_t min_val = 255;
    uint8_t max_val = 0;
    for (size_t y = 0; y < 100; ++y) {
        for (size_t x = 0; x < 100; ++x) {
            min_val = std::min(min_val, dst(x, y));
            max_val = std::max(max_val, dst(x, y));
        }
    }

    EXPECT_EQ(min_val, 0);
    EXPECT_EQ(max_val, 255);
}

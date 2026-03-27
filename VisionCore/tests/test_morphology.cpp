#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/morphology.hpp>

using namespace visioncore;

TEST(MorphologyTest, StructuringElements) {
    auto rect = StructuringElement::rectangle(3, 5);
    EXPECT_EQ(rect.width, 3);
    EXPECT_EQ(rect.height, 5);
    for (uint8_t val : rect.data) {
        EXPECT_EQ(val, 1);
    }

    auto cross = StructuringElement::cross(3, 3);
    EXPECT_EQ(cross.width, 3);
    EXPECT_EQ(cross.height, 3);
    // Center column and center row are 1s, corners are 0s
    EXPECT_EQ(cross.data[0], 0); EXPECT_EQ(cross.data[1], 1); EXPECT_EQ(cross.data[2], 0);
    EXPECT_EQ(cross.data[3], 1); EXPECT_EQ(cross.data[4], 1); EXPECT_EQ(cross.data[5], 1);
    EXPECT_EQ(cross.data[6], 0); EXPECT_EQ(cross.data[7], 1); EXPECT_EQ(cross.data[8], 0);
}

TEST(MorphologyTest, Dilation) {
    Image<uint8_t, 1> src(5, 5, 0);
    src(2, 2) = 255; // Set center pixel to 255

    Image<uint8_t, 1> dst(5, 5, 0);
    auto se = StructuringElement::rectangle(3, 3);
    dilate(src.const_view(), dst.view(), se, BorderType::ZERO);

    // Center 3x3 region should be 255, corners should be 0
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            if (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
                EXPECT_EQ(dst(x, y), 255);
            } else {
                EXPECT_EQ(dst(x, y), 0);
            }
        }
    }
}

TEST(MorphologyTest, Erosion) {
    Image<uint8_t, 1> src(5, 5, 255);
    src(2, 2) = 0; // Set center pixel to 0

    Image<uint8_t, 1> dst(5, 5, 0);
    auto se = StructuringElement::rectangle(3, 3);
    erode(src.const_view(), dst.view(), se, BorderType::REPLICATE);

    // Center 3x3 region should be 0, corners should be 255
    for (size_t y = 0; y < 5; ++y) {
        for (size_t x = 0; x < 5; ++x) {
            if (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
                EXPECT_EQ(dst(x, y), 0);
            } else {
                EXPECT_EQ(dst(x, y), 255);
            }
        }
    }
}

TEST(MorphologyTest, OpeningClosing) {
    Image<uint8_t, 1> src(5, 5, 0);
    // Draw a solid 3x3 block in the center
    for (size_t y = 1; y <= 3; ++y) {
        for (size_t x = 1; x <= 3; ++x) {
            src(x, y) = 255;
        }
    }
    // Add single-pixel noise (an isolated foreground pixel and an isolated background hole)
    src(0, 0) = 255; // isolated foreground noise
    src(2, 2) = 0;   // isolated background hole

    auto se = StructuringElement::rectangle(3, 3);
    Image<uint8_t, 1> dst_open(5, 5, 0);
    Image<uint8_t, 1> dst_close(5, 5, 0);

    // Opening should remove the isolated foreground pixel at (0, 0)
    opening(src.const_view(), dst_open.view(), se, BorderType::ZERO);
    EXPECT_EQ(dst_open(0, 0), 0);
    // Note: the hole at (2, 2) is retained or widened by opening

    // Closing should fill the isolated background hole at (2, 2)
    closing(src.const_view(), dst_close.view(), se, BorderType::ZERO);
    EXPECT_EQ(dst_close(2, 2), 255);
}

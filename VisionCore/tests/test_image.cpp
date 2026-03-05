#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <type_traits>

using namespace visioncore;

TEST(ImageTest, DefaultConstructor) {
    Image<uint8_t, 1> img;
    EXPECT_TRUE(img.empty());
    EXPECT_EQ(img.width(), 0);
    EXPECT_EQ(img.height(), 0);
    EXPECT_EQ(img.stride_elements(), 0);
}

TEST(ImageTest, ParameterizedConstructor) {
    Image<uint8_t, 3> img(10, 20, 128);
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.width(), 10);
    EXPECT_EQ(img.height(), 20);
    EXPECT_EQ(img.stride_elements(), 30);
    EXPECT_EQ(img(0, 0, 0), 128);
    EXPECT_EQ(img(9, 19, 2), 128);
}

TEST(ImageTest, CopyAndMoveSemantics) {
    Image<float, 1> img1(5, 5, 1.5f);
    
    // Copy
    Image<float, 1> img2 = img1;
    EXPECT_EQ(img2.width(), 5);
    EXPECT_FLOAT_EQ(img2(2, 2, 0), 1.5f);
    
    // Modify copy
    img2(2, 2, 0) = 4.0f;
    EXPECT_FLOAT_EQ(img1(2, 2, 0), 1.5f); // original unchanged
    EXPECT_FLOAT_EQ(img2(2, 2, 0), 4.0f); // copy changed
    
    // Move
    Image<float, 1> img3 = std::move(img1);
    EXPECT_TRUE(img1.empty());
    EXPECT_EQ(img3.width(), 5);
    EXPECT_FLOAT_EQ(img3(2, 2, 0), 1.5f);
}

TEST(ImageTest, BoundsChecking) {
    Image<uint8_t, 1> img(5, 5, 0);
    EXPECT_NO_THROW((void)img.at(4, 4, 0));
    EXPECT_THROW((void)img.at(5, 4, 0), std::out_of_range);
    EXPECT_THROW((void)img.at(4, 5, 0), std::out_of_range);
    EXPECT_THROW((void)img.at(4, 4, 1), std::out_of_range);
}

TEST(ImageViewTest, SubViewROI) {
    Image<uint8_t, 1> img(10, 10);
    // Fill with grid patterns
    for (size_t y = 0; y < 10; ++y) {
        for (size_t x = 0; x < 10; ++x) {
            img(x, y) = static_cast<uint8_t>(y * 10 + x);
        }
    }
    
    ImageView<uint8_t, 1> view = img.view();
    // Crop a 3x3 region at (2, 4)
    ImageView<uint8_t, 1> roi = view.sub_view(2, 4, 3, 3);
    
    EXPECT_EQ(roi.width(), 3);
    EXPECT_EQ(roi.height(), 3);
    EXPECT_EQ(roi.stride_elements(), 10); // stride remains same as parent
    
    // Check pixel mapping
    EXPECT_EQ(roi(0, 0), 42); // (2, 4) -> 4 * 10 + 2 = 42
    EXPECT_EQ(roi(2, 2), 64); // (4, 6) -> 6 * 10 + 4 = 64
    
    // Check bounds checking on ROI
    EXPECT_NO_THROW((void)roi.at(2, 2));
    EXPECT_THROW((void)roi.at(3, 2), std::out_of_range);
}

TEST(ImageViewTest, ConstPromotion) {
    Image<uint8_t, 1> img(5, 5, 10);
    ImageView<uint8_t, 1> non_const_view = img.view();
    
    // Implicit conversion / promotion
    ImageView<const uint8_t, 1> const_view = non_const_view;
    EXPECT_EQ(const_view(0, 0), 10);
}

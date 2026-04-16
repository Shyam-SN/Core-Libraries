#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/template_matching.hpp>

using namespace visioncore;

TEST(TemplateMatchingTest, MinMaxLocation) {
    Image<float, 1> response(5, 5, 0.0f);
    response(1, 2) = -10.0f; // Minimum
    response(3, 4) = 50.0f;  // Maximum

    float min_val = 0.0f, max_val = 0.0f;
    size_t min_x = 0, min_y = 0, max_x = 0, max_y = 0;

    min_max_loc(response.const_view(), min_val, max_val, min_x, min_y, max_x, max_y);

    EXPECT_DOUBLE_EQ(min_val, -10.0);
    EXPECT_DOUBLE_EQ(max_val, 50.0);
    EXPECT_EQ(min_x, 1);
    EXPECT_EQ(min_y, 2);
    EXPECT_EQ(max_x, 3);
    EXPECT_EQ(max_y, 4);
}

TEST(TemplateMatchingTest, MatchVerification) {
    // 6x6 source image
    Image<uint8_t, 1> src(6, 6, 10);
    // Draw a unique pattern inside the image
    src(2, 3) = 100;
    src(3, 3) = 200;
    src(2, 4) = 150;
    src(3, 4) = 250;

    // 2x2 template containing that pattern
    Image<uint8_t, 1> templ(2, 2, 0);
    templ(0, 0) = 100;
    templ(1, 0) = 200;
    templ(0, 1) = 150;
    templ(1, 1) = 255; // tiny noise

    // Expected output size: 6 - 2 + 1 = 5
    Image<float, 1> res_sqdiff(5, 5, 0.0f);
    Image<float, 1> res_coeff(5, 5, 0.0f);

    match_template(src.const_view(), templ.const_view(), res_sqdiff.view(), TemplateMetric::SQDIFF);
    match_template(src.const_view(), templ.const_view(), res_coeff.view(), TemplateMetric::CCOEFF_NORMED);

    // Locate best matches
    float min_v1 = 0.0f, max_v1 = 0.0f;
    size_t min_x1 = 0, min_y1 = 0, max_x1 = 0, max_y1 = 0;
    min_max_loc(res_sqdiff.const_view(), min_v1, max_v1, min_x1, min_y1, max_x1, max_y1);

    // SQDIFF best match should be at (2, 3)
    EXPECT_EQ(min_x1, 2);
    EXPECT_EQ(min_y1, 3);
    // Difference is (250-255)^2 = 25
    EXPECT_DOUBLE_EQ(min_v1, 25.0);

    float min_v2 = 0.0f, max_v2 = 0.0f;
    size_t min_x2 = 0, min_y2 = 0, max_x2 = 0, max_y2 = 0;
    min_max_loc(res_coeff.const_view(), min_v2, max_v2, min_x2, min_y2, max_x2, max_y2);

    // CCOEFF_NORMED best match should be at (2, 3) and very close to 1.0
    EXPECT_EQ(max_x2, 2);
    EXPECT_EQ(max_y2, 3);
    EXPECT_GT(max_v2, 0.95f);
}

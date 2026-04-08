#include <gtest/gtest.h>
#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/orb.hpp>
#include <vector>
#include <cmath>

using namespace visioncore;

TEST(OrbTest, BriefPairsGeneration) {
    auto pairs = generate_brief_pairs(256, 15);
    EXPECT_EQ(pairs.size(), 256);

    for (const auto& pair : pairs) {
        // Points must be inside the circular patch of radius 15
        EXPECT_LE(pair.x1*pair.x1 + pair.y1*pair.y1, 15*15);
        EXPECT_LE(pair.x2*pair.x2 + pair.y2*pair.y2, 15*15);
    }
}

TEST(OrbTest, KeypointOrientation) {
    // Generate a simple step edge (left half dark, right half bright)
    Image<uint8_t, 1> src(31, 31, 50);
    for (size_t y = 0; y < 31; ++y) {
        for (size_t x = 16; x < 31; ++x) {
            src(x, y) = 200;
        }
    }

    // Keypoint at center (15, 15)
    float angle = compute_keypoint_orientation(src.const_view(), 15.0f, 15.0f, 15);

    // Orientation should point toward the bright side (horizontally right, angle approx 0)
    EXPECT_NEAR(angle, 0.0f, 0.2f);
}

TEST(OrbTest, DescriptorExtraction) {
    Image<uint8_t, 1> src(50, 50, 100);
    // Draw some texture so descriptors are not flat zero
    for (size_t y = 0; y < 50; ++y) {
        for (size_t x = 0; x < 50; ++x) {
            src(x, y) = static_cast<uint8_t>((x * 7 + y * 13) % 256);
        }
    }

    std::vector<Keypoint> kps = {
        Keypoint(5.0f, 5.0f, 1.0f),    // too close to border (should be filtered out)
        Keypoint(25.0f, 25.0f, 2.0f),  // valid
        Keypoint(45.0f, 45.0f, 3.0f)   // too close to border (should be filtered out)
    };

    std::vector<std::vector<uint8_t>> descriptors;
    compute_orb_descriptors(src.const_view(), kps, descriptors);

    // Only 1 keypoint should remain after border filtering
    EXPECT_EQ(kps.size(), 1);
    EXPECT_EQ(descriptors.size(), 1);

    // Verify keypoint got oriented
    EXPECT_GE(kps[0].angle, -3.14159f);
    EXPECT_LE(kps[0].angle, 3.14159f);

    // Verify descriptor has size 32 bytes
    EXPECT_EQ(descriptors[0].size(), 32);

    // Verify it is not completely empty (at least some bits set due to the generated texture)
    bool non_zero = false;
    for (uint8_t byte : descriptors[0]) {
        if (byte > 0) {
            non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(non_zero);
}

#include <gtest/gtest.h>
#include <visioncore/algorithms/feature_matching.hpp>
#include <vector>

using namespace visioncore;

TEST(FeatureMatchingTest, HammingDistanceCalculation) {
    // Descriptor with all 0s
    std::vector<uint8_t> d1(32, 0x00);
    // Descriptor with all 1s (32 * 8 = 256 bits)
    std::vector<uint8_t> d2(32, 0xFF);
    // Descriptor with alternating bits (01010101 = 0x55, popcount is 4 per byte -> 32 * 4 = 128)
    std::vector<uint8_t> d3(32, 0x55);

    EXPECT_EQ(compute_hamming_distance(d1, d2), 256);
    EXPECT_EQ(compute_hamming_distance(d1, d3), 128);
    EXPECT_EQ(compute_hamming_distance(d2, d3), 128);
    EXPECT_EQ(compute_hamming_distance(d1, d1), 0);
}

TEST(FeatureMatchingTest, BruteForceMatching) {
    std::vector<std::vector<uint8_t>> query = {
        std::vector<uint8_t>(32, 0xAA),
        std::vector<uint8_t>(32, 0x55)
    };

    std::vector<std::vector<uint8_t>> train = {
        std::vector<uint8_t>(32, 0x11), // noisy
        std::vector<uint8_t>(32, 0x55), // exact match for query[1]
        std::vector<uint8_t>(32, 0xAA)  // exact match for query[0]
    };

    std::vector<Match> matches;
    match_brute_force(query, train, matches);

    EXPECT_EQ(matches.size(), 2);

    // query[0] (0xAA) matches train[2] (0xAA)
    EXPECT_EQ(matches[0].query_idx, 0);
    EXPECT_EQ(matches[0].train_idx, 2);
    EXPECT_DOUBLE_EQ(matches[0].distance, 0.0);

    // query[1] (0x55) matches train[1] (0x55)
    EXPECT_EQ(matches[1].query_idx, 1);
    EXPECT_EQ(matches[1].train_idx, 1);
    EXPECT_DOUBLE_EQ(matches[1].distance, 0.0);
}

TEST(FeatureMatchingTest, RatioTest) {
    std::vector<std::vector<uint8_t>> query = {
        std::vector<uint8_t>(32, 0xFF)
    };

    // Case 1: Ambiguous train set (both train[0] and train[1] are very close to query)
    std::vector<uint8_t> t0(32, 0xFF);
    t0[0] = 0xFE; // Hamming distance = 1
    std::vector<uint8_t> t1(32, 0xFF);
    t1[0] = 0xFC; // Hamming distance = 2

    std::vector<std::vector<uint8_t>> train_ambiguous = { t0, t1 };
    std::vector<Match> matches1;
    match_ratio_test(query, train_ambiguous, matches1, 0.7f);

    // Should be rejected since 1/2 = 0.5 < 0.7 (wait, 0.5 IS less than 0.7, so it should be accepted!)
    // Let's test a ratio of 0.4. 1/2 = 0.5 > 0.4, so it should be rejected.
    match_ratio_test(query, train_ambiguous, matches1, 0.4f);
    EXPECT_TRUE(matches1.empty());

    // Case 2: Distinct match (nearest distance = 1, second nearest = 10)
    std::vector<uint8_t> t2(32, 0x00); // Hamming distance = 256
    std::vector<std::vector<uint8_t>> train_distinct = { t0, t2 };
    std::vector<Match> matches2;
    match_ratio_test(query, train_distinct, matches2, 0.4f);

    // Should be accepted since 1 / 256 = 0.0039 < 0.4
    EXPECT_EQ(matches2.size(), 1);
    EXPECT_EQ(matches2[0].train_idx, 0);
}

#pragma once

#include <vector>
#include <bit>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Representation of a match between two keypoints.
 */
struct Match {
    int query_idx;  ///< Index of keypoint in query set
    int train_idx;  ///< Index of keypoint in train set
    float distance; ///< Distance metric (e.g., Hamming distance)

    Match() : query_idx(-1), train_idx(-1), distance(std::numeric_limits<float>::max()) {}
    Match(int q_idx, int t_idx, float dist) : query_idx(q_idx), train_idx(t_idx), distance(dist) {}
};

/**
 * @brief Calculates the Hamming distance between two 32-byte binary descriptors.
 * Leverages C++20 std::popcount for hardware-accelerated computation.
 */
inline unsigned int compute_hamming_distance(const std::vector<uint8_t>& desc1, const std::vector<uint8_t>& desc2) {
    if (desc1.size() != 32 || desc2.size() != 32) {
        throw std::invalid_argument("ORB descriptors must be exactly 32 bytes.");
    }
    
    unsigned int distance = 0;
    for (size_t i = 0; i < 32; ++i) {
        distance += std::popcount(static_cast<unsigned char>(desc1[i] ^ desc2[i]));
    }
    return distance;
}

/**
 * @brief Performs brute-force matching between two sets of binary descriptors.
 * Finds the nearest neighbor in the train set for each descriptor in the query set.
 */
inline void match_brute_force(const std::vector<std::vector<uint8_t>>& query_descs,
                             const std::vector<std::vector<uint8_t>>& train_descs,
                             std::vector<Match>& matches) {
    matches.clear();
    matches.reserve(query_descs.size());

    for (size_t q = 0; q < query_descs.size(); ++q) {
        unsigned int min_dist = std::numeric_limits<unsigned int>::max();
        int best_t = -1;

        for (size_t t = 0; t < train_descs.size(); ++t) {
            unsigned int dist = compute_hamming_distance(query_descs[q], train_descs[t]);
            if (dist < min_dist) {
                min_dist = dist;
                best_t = static_cast<int>(t);
            }
        }

        if (best_t != -1) {
            matches.emplace_back(static_cast<int>(q), best_t, static_cast<float>(min_dist));
        }
    }
}

/**
 * @brief Performs matching with Lowe's Ratio Test to reject ambiguous matches.
 * Matches are accepted if the ratio of the nearest neighbor distance to the second-nearest 
 * neighbor distance is less than the given threshold.
 */
inline void match_ratio_test(const std::vector<std::vector<uint8_t>>& query_descs,
                            const std::vector<std::vector<uint8_t>>& train_descs,
                            std::vector<Match>& matches,
                            float ratio_threshold = 0.75f) {
    matches.clear();
    matches.reserve(query_descs.size());

    if (train_descs.size() < 2) {
        // Cannot apply ratio test if there are fewer than 2 candidates in the train set
        return;
    }

    for (size_t q = 0; q < query_descs.size(); ++q) {
        unsigned int dist1 = std::numeric_limits<unsigned int>::max();
        unsigned int dist2 = std::numeric_limits<unsigned int>::max();
        int best_t = -1;

        for (size_t t = 0; t < train_descs.size(); ++t) {
            unsigned int dist = compute_hamming_distance(query_descs[q], train_descs[t]);
            if (dist < dist1) {
                dist2 = dist1;
                dist1 = dist;
                best_t = static_cast<int>(t);
            } else if (dist < dist2) {
                dist2 = dist;
            }
        }

        if (best_t != -1 && dist2 > 0) {
            float ratio = static_cast<float>(dist1) / static_cast<float>(dist2);
            if (ratio < ratio_threshold) {
                matches.emplace_back(static_cast<int>(q), best_t, static_cast<float>(dist1));
            }
        }
    }
}

} // namespace visioncore

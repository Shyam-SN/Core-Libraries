#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Representation of a binary comparison pair for BRIEF.
 */
struct BriefPair {
    int x1, y1;
    int x2, y2;
};

/**
 * @brief Generates 256 reproducible binary comparison coordinate pairs.
 * Uses a linear congruential generator (LCG) with a fixed seed to ensure determinism.
 */
inline std::vector<BriefPair> generate_brief_pairs(int count = 256, int radius = 15) {
    std::vector<BriefPair> pairs;
    pairs.reserve(count);

    // Fixed-seed LCG for reproducible coordinates
    uint32_t seed = 0x12345678;
    auto rand_int = [&seed](int min_val, int max_val) -> int {
        seed = seed * 1664525 + 1013904223;
        int range = max_val - min_val + 1;
        return min_val + static_cast<int>(seed % range);
    };

    while (pairs.size() < static_cast<size_t>(count)) {
        int x1 = rand_int(-radius, radius);
        int y1 = rand_int(-radius, radius);
        int x2 = rand_int(-radius, radius);
        int y2 = rand_int(-radius, radius);

        // Ensure both comparison points are inside the circular patch
        if (x1 * x1 + y1 * y1 <= radius * radius && x2 * x2 + y2 * y2 <= radius * radius) {
            pairs.push_back({x1, y1, x2, y2});
        }
    }
    return pairs;
}

/**
 * @brief Computes the orientation angle of a keypoint using intensity centroids.
 * 
 * @param src Grayscale input image.
 * @param kpx Keypoint X coordinate.
 * @param kpy Keypoint Y coordinate.
 * @param radius Patch radius to calculate moments over (usually 15).
 * @return float Orientation angle in radians.
 */
inline float compute_keypoint_orientation(const ImageView<const uint8_t, 1>& src, float kpx, float kpy, int radius = 15) {
    double m01 = 0.0;
    double m10 = 0.0;

    int ix = static_cast<int>(std::round(kpx));
    int iy = static_cast<int>(std::round(kpy));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= radius * radius) {
                int px = ix + dx;
                int py = iy + dy;

                if (px >= 0 && px < static_cast<int>(src.width()) &&
                    py >= 0 && py < static_cast<int>(src.height())) {
                    uint8_t val = src(static_cast<size_t>(px), static_cast<size_t>(py));
                    m10 += dx * val;
                    m01 += dy * val;
                }
            }
        }
    }
    return static_cast<float>(std::atan2(m01, m10));
}

/**
 * @brief Computes 256-bit (32-byte) ORB descriptors for the given keypoints.
 * Mutates the keypoints array to compute orientation and filter out border candidates.
 * 
 * @param src Grayscale input image view.
 * @param keypoints Vector of input keypoints (will be mutated and filtered).
 * @param descriptors Output vector of 32-byte descriptors.
 */
inline void compute_orb_descriptors(const ImageView<const uint8_t, 1>& src,
                                    std::vector<Keypoint>& keypoints,
                                    std::vector<std::vector<uint8_t>>& descriptors) {
    const int radius = 15;
    const auto brief_pairs = generate_brief_pairs(256, radius);

    std::vector<Keypoint> filtered_kps;
    filtered_kps.reserve(keypoints.size());
    descriptors.clear();

    for (auto& kp : keypoints) {
        int ix = static_cast<int>(std::round(kp.x));
        int iy = static_cast<int>(std::round(kp.y));

        // Filter keypoints too close to the borders (need radius of 15)
        if (ix >= radius && ix < static_cast<int>(src.width()) - radius &&
            iy >= radius && iy < static_cast<int>(src.height()) - radius) {

            // 1. Calculate orientation if not already defined
            if (kp.angle < 0.0f) {
                kp.angle = compute_keypoint_orientation(src, kp.x, kp.y, radius);
            }

            filtered_kps.push_back(kp);

            // 2. Compute 256-bit descriptor
            std::vector<uint8_t> desc(32, 0);
            float cos_t = std::cos(kp.angle);
            float sin_t = std::sin(kp.angle);

            for (size_t i = 0; i < 256; ++i) {
                const auto& pair = brief_pairs[i];

                // Rotate test coordinate 1
                int rx1 = static_cast<int>(std::round(pair.x1 * cos_t - pair.y1 * sin_t));
                int ry1 = static_cast<int>(std::round(pair.x1 * sin_t + pair.y1 * cos_t));

                // Rotate test coordinate 2
                int rx2 = static_cast<int>(std::round(pair.x2 * cos_t - pair.y2 * sin_t));
                int ry2 = static_cast<int>(std::round(pair.x2 * sin_t + pair.y2 * cos_t));

                uint8_t val1 = src(static_cast<size_t>(ix + rx1), static_cast<size_t>(iy + ry1));
                uint8_t val2 = src(static_cast<size_t>(ix + rx2), static_cast<size_t>(iy + ry2));

                if (val1 < val2) {
                    size_t byte_idx = i / 8;
                    size_t bit_idx = i % 8;
                    desc[byte_idx] |= (1 << bit_idx);
                }
            }
            descriptors.push_back(desc);
        }
    }
    keypoints = std::move(filtered_kps);
}

} // namespace visioncore

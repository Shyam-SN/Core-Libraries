#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Performs basic global binary thresholding.
 * Maps pixels above or equal to thresh to max_val, and others to 0.
 */
inline void threshold_binary(const ImageView<const uint8_t, 1>& src,
                             const ImageView<uint8_t, 1>& dst,
                             uint8_t thresh,
                             uint8_t max_val = 255) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }
    const size_t w = src.width();
    const size_t h = src.height();

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            dst(x, y) = src(x, y) >= thresh ? max_val : 0;
        }
    }
}

/**
 * @brief Computes Otsu's optimal global threshold.
 * Maximizes inter-class variance between background and foreground classes.
 */
inline uint8_t compute_otsu_threshold(const ImageView<const uint8_t, 1>& src) {
    const size_t w = src.width();
    const size_t h = src.height();
    const double total_pixels = static_cast<double>(w * h);

    if (total_pixels == 0.0) return 0;

    // 1. Compute histogram
    std::vector<size_t> hist(256, 0);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            hist[src(x, y)]++;
        }
    }

    // 2. Sum of all pixel values (for calculating mean)
    double sum = 0.0;
    for (int t = 0; t < 256; ++t) {
        sum += t * static_cast<double>(hist[t]);
    }

    double sumB = 0.0;
    double wB = 0.0;
    double wF = 0.0;

    double varMax = -1.0;
    double threshold_sum = 0.0;
    int threshold_count = 0;

    for (int t = 0; t < 256; ++t) {
        wB += static_cast<double>(hist[t]);
        if (wB == 0) continue;

        wF = total_pixels - wB;
        if (wF == 0) break;

        sumB += t * static_cast<double>(hist[t]);

        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        // Calculate Between-Class Variance
        double varBetween = wB * wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax + 1e-6) {
            varMax = varBetween;
            threshold_sum = t;
            threshold_count = 1;
        } else if (std::abs(varBetween - varMax) < 1e-6) {
            threshold_sum += t;
            threshold_count++;
        }
    }

    if (threshold_count > 0) {
        return static_cast<uint8_t>(std::round(threshold_sum / threshold_count));
    }
    return 0;
}

/**
 * @brief Automatically thresholds an image using Otsu's optimal calculated threshold.
 */
inline void threshold_otsu(const ImageView<const uint8_t, 1>& src,
                           const ImageView<uint8_t, 1>& dst,
                           uint8_t max_val = 255) {
    uint8_t thresh = compute_otsu_threshold(src);
    threshold_binary(src, dst, thresh, max_val);
}

/**
 * @brief Method choice for localized thresholding neighborhood computation.
 */
enum class AdaptiveMethod {
    MEAN,       ///< Box filter (uniform neighborhood average)
    GAUSSIAN    ///< Gaussian blur (Gaussian weighted average)
};

/**
 * @brief Performs localized adaptive thresholding.
 * Threshold is computed dynamically for each pixel: T(x, y) = local_mean(x, y) - C
 */
inline void threshold_adaptive(const ImageView<const uint8_t, 1>& src,
                              const ImageView<uint8_t, 1>& dst,
                              size_t block_size,
                              double C,
                              AdaptiveMethod method = AdaptiveMethod::MEAN,
                              uint8_t max_val = 255) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }
    if (block_size % 2 == 0) {
        throw std::invalid_argument("Block size must be odd.");
    }

    const size_t w = src.width();
    const size_t h = src.height();

    // 1. Generate local mean reference image using optimized filtering modules
    Image<uint8_t, 1> local_mean(w, h);
    if (method == AdaptiveMethod::MEAN) {
        box_filter(src, local_mean.view(), block_size, BorderType::REPLICATE);
    } else {
        gaussian_blur(src, local_mean.view(), block_size, 0.0, BorderType::REPLICATE);
    }

    // 2. Apply thresholding based on localized T(x, y) = local_mean - C
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            double local_t = static_cast<double>(local_mean(x, y)) - C;
            dst(x, y) = static_cast<double>(src(x, y)) >= local_t ? max_val : 0;
        }
    }
}

} // namespace visioncore

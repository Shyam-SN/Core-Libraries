#pragma once

#include <visioncore/core/image.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace visioncore {

/**
 * @brief Computes the 256-bin intensity histogram of a single-channel 8-bit image view.
 */
inline std::vector<size_t> compute_histogram(const ImageView<const uint8_t, 1>& src) {
    std::vector<size_t> hist(256, 0);
    const size_t w = src.width();
    const size_t h = src.height();

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            hist[src(x, y)]++;
        }
    }
    return hist;
}

/**
 * @brief Performs global Histogram Equalization to enhance the contrast of a single-channel 8-bit image.
 */
inline void equalize_histogram(const ImageView<const uint8_t, 1>& src, const ImageView<uint8_t, 1>& dst) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }

    const size_t w = src.width();
    const size_t h = src.height();
    const double total_pixels = static_cast<double>(w * h);

    if (total_pixels == 0.0) return;

    // 1. Compute histogram
    std::vector<size_t> hist = compute_histogram(src);

    // 2. Compute Cumulative Distribution Function (CDF)
    std::vector<double> cdf(256, 0.0);
    double cumulative = 0.0;
    for (size_t i = 0; i < 256; ++i) {
        cumulative += static_cast<double>(hist[i]);
        cdf[i] = cumulative / total_pixels;
    }

    // 3. Find first non-zero CDF value
    double cdf_min = 0.0;
    for (size_t i = 0; i < 256; ++i) {
        if (cdf[i] > 0.0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // 4. Generate lookup mapping table
    uint8_t mapping[256];
    if (1.0 - cdf_min < 1e-6) {
        // If image is flat (all pixels identical), map them to their original value
        for (size_t i = 0; i < 256; ++i) mapping[i] = static_cast<uint8_t>(i);
    } else {
        for (size_t i = 0; i < 256; ++i) {
            double equalized = (cdf[i] - cdf_min) / (1.0 - cdf_min) * 255.0;
            mapping[i] = static_cast<uint8_t>(std::clamp(std::round(equalized), 0.0, 255.0));
        }
    }

    // 5. Apply map to destination image
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            dst(x, y) = mapping[src(x, y)];
        }
    }
}

} // namespace visioncore

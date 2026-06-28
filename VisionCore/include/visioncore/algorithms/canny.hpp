#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <visioncore/algorithms/convolution.hpp>
#include <vector>
#include <cmath>
#include <stack>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Performs Canny edge detection on a single-channel 8-bit image view.
 * 
 * @param src Input grayscale image view.
 * @param dst Output binary edge map. Must have the same dimensions as src.
 * @param low_thresh Lower threshold for edge hysteresis.
 * @param high_thresh Upper threshold for edge hysteresis.
 * @param blur_ksize Size of the Gaussian blur kernel to remove noise (must be odd).
 * @param blur_sigma Standard deviation of the Gaussian blur kernel.
 */
inline void canny(const ImageView<const uint8_t, 1>& src,
                  const ImageView<uint8_t, 1>& dst,
                  double low_thresh,
                  double high_thresh,
                  size_t blur_ksize = 3,
                  double blur_sigma = 0.0) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }

    const size_t w = src.width();
    const size_t h = src.height();

    // 1. Noise Reduction (Gaussian blur)
    Image<uint8_t, 1> blurred(w, h);
    gaussian_blur(src, blurred.view(), blur_ksize, blur_sigma, BorderType::REPLICATE);

    // 2. Gradients (Sobel)
    Image<float, 1> dx(w, h);
    Image<float, 1> dy(w, h);
    sobel_x(blurred.const_view(), dx.view(), BorderType::REPLICATE);
    sobel_y(blurred.const_view(), dy.view(), BorderType::REPLICATE);

    // Calculate magnitude
    Image<float, 1> magnitude(w, h);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            magnitude(x, y) = std::hypot(dx(x, y), dy(x, y));
        }
    }

    // 3. Non-Maximum Suppression & Double Thresholding
    // Initialize temporary edge map: 0 = non-edge, 127 = weak, 255 = strong
    Image<uint8_t, 1> suppressed(w, h, 0);

    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            float mag = magnitude(x, y);
            if (mag < 1e-5f) {
                continue;
            }

            float gx = dx(x, y);
            float gy = dy(x, y);

            float mag_n1 = 0.0f;
            float mag_n2 = 0.0f;

            float abs_gx = std::abs(gx);
            float abs_gy = std::abs(gy);

            // Angle classification into 4 sectors (0, 45, 90, 135 degrees)
            if (abs_gx >= abs_gy) {
                // Horizontal-ish
                float ratio = gy / gx;
                if (ratio >= -0.4142f && ratio < 0.4142f) {
                    // Sector 0 (Horizontal)
                    mag_n1 = magnitude(x - 1, y);
                    mag_n2 = magnitude(x + 1, y);
                } else if (ratio >= 0.4142f) {
                    // Sector 45 (Diagonal Up-Right)
                    mag_n1 = magnitude(x - 1, y - 1);
                    mag_n2 = magnitude(x + 1, y + 1);
                } else {
                    // Sector 135 (Diagonal Up-Left)
                    mag_n1 = magnitude(x - 1, y + 1);
                    mag_n2 = magnitude(x + 1, y - 1);
                }
            } else {
                // Vertical-ish
                float ratio = gx / gy;
                if (ratio >= -0.4142f && ratio < 0.4142f) {
                    // Sector 90 (Vertical)
                    mag_n1 = magnitude(x, y - 1);
                    mag_n2 = magnitude(x, y + 1);
                } else if (ratio >= 0.4142f) {
                    // Sector 45 (Diagonal Up-Right)
                    mag_n1 = magnitude(x - 1, y - 1);
                    mag_n2 = magnitude(x + 1, y + 1);
                } else {
                    // Sector 135 (Diagonal Up-Left)
                    mag_n1 = magnitude(x - 1, y + 1);
                    mag_n2 = magnitude(x + 1, y - 1);
                }
            }

            // Suppress non-maxima
            if (mag >= mag_n1 && mag >= mag_n2) {
                // 4. Double Thresholding
                if (mag >= high_thresh) {
                    suppressed(x, y) = 255; // Strong edge
                } else if (mag >= low_thresh) {
                    suppressed(x, y) = 127; // Weak edge
                }
            }
        }
    }

    // 5. Hysteresis Edge Tracking
    // Initialize destination as 0
    std::fill(dst.data(), dst.data() + w * h, 0);

    std::stack<std::pair<size_t, size_t>> edge_stack;

    // Push all strong edges onto the stack
    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            if (suppressed(x, y) == 255) {
                dst(x, y) = 255;
                edge_stack.push({x, y});
            }
        }
    }

    // DFS to trace connected weak edges
    while (!edge_stack.empty()) {
        auto [cx, cy] = edge_stack.top();
        edge_stack.pop();

        for (int dy_offset = -1; dy_offset <= 1; ++dy_offset) {
            for (int dx_offset = -1; dx_offset <= 1; ++dx_offset) {
                if (dx_offset == 0 && dy_offset == 0) continue;

                size_t nx = cx + dx_offset;
                size_t ny = cy + dy_offset;

                if (nx > 0 && nx < w - 1 && ny > 0 && ny < h - 1) {
                    if (suppressed(nx, ny) == 127 && dst(nx, ny) == 0) {
                        dst(nx, ny) = 255;
                        edge_stack.push({nx, ny});
                    }
                }
            }
        }
    }
}

} // namespace visioncore

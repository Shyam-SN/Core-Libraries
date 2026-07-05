#pragma once

#include <visioncore/core/image.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace visioncore {

/**
 * @brief Border padding strategies for handling image edge cases.
 */
enum class BorderType {
    ZERO,       ///< Pad with zeros (black border)
    REPLICATE,  ///< Replicate edge pixels (aaaaa|abcdefgh|hhhhh)
    REFLECT     ///< Reflect across the border (cb|abcdefgh|gf)
};

/**
 * @brief Helper function to retrieve a pixel value at any coordinate, applying the specified border padding strategy.
 */
template <PixelType T, size_t Channels>
[[nodiscard]] inline T get_pixel_border(const ImageView<T, Channels>& view, int64_t x, int64_t y, size_t c, BorderType border) noexcept {
    const int64_t w = static_cast<int64_t>(view.width());
    const int64_t h = static_cast<int64_t>(view.height());

    if (x >= 0 && x < w && y >= 0 && y < h) {
        return view(static_cast<size_t>(x), static_cast<size_t>(y), c);
    }

    if (border == BorderType::ZERO) {
        return T(0);
    } else if (border == BorderType::REPLICATE) {
        const int64_t clamped_x = std::clamp(x, int64_t(0), w - 1);
        const int64_t clamped_y = std::clamp(y, int64_t(0), h - 1);
        return view(static_cast<size_t>(clamped_x), static_cast<size_t>(clamped_y), c);
    } else { // BorderType::REFLECT
        auto reflect = [](int64_t val, int64_t max_val) noexcept {
            if (val < 0) {
                val = -val;
            }
            if (val >= max_val) {
                val = 2 * (max_val - 1) - val;
            }
            return std::clamp(val, int64_t(0), max_val - 1);
        };
        const int64_t rx = reflect(x, w);
        const int64_t ry = reflect(y, h);
        return view(static_cast<size_t>(rx), static_cast<size_t>(ry), c);
    }
}

/**
 * @brief Performs a 2D spatial convolution on an input image view using a 2D kernel.
 * 
 * @tparam T Input pixel element type.
 * @tparam K Kernel element type (typically float or double).
 * @tparam Channels Number of channels in the image.
 * 
 * @param src Input image view.
 * @param dst Output image view. Must have the same dimensions and channels as src.
 * @param kernel 2D kernel represented as a flat row-major vector.
 * @param k_width Kernel width (must be odd).
 * @param k_height Kernel height (must be odd).
 * @param border Border handling strategy.
 */
template <PixelType T_In, PixelType T_Out, typename K, size_t Channels>
void convolve2d(const ImageView<T_In, Channels>& src,
                const ImageView<T_Out, Channels>& dst,
                const std::vector<K>& kernel,
                size_t k_width,
                size_t k_height,
                BorderType border = BorderType::REPLICATE) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }
    if (k_width % 2 == 0 || k_height % 2 == 0) {
        throw std::invalid_argument("Kernel dimensions must be odd.");
    }
    if (kernel.size() != k_width * k_height) {
        throw std::invalid_argument("Kernel size does not match specified dimensions.");
    }

    const int64_t w = static_cast<int64_t>(src.width());
    const int64_t h = static_cast<int64_t>(src.height());
    const int64_t kw_half = static_cast<int64_t>(k_width / 2);
    const int64_t kh_half = static_cast<int64_t>(k_height / 2);

    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                double sum = 0.0;
                for (int64_t ky = 0; ky < static_cast<int64_t>(k_height); ++ky) {
                    for (int64_t kx = 0; kx < static_cast<int64_t>(k_width); ++kx) {
                        const int64_t px = x + kx - kw_half;
                        const int64_t py = y + ky - kh_half;
                        const K k_val = kernel[ky * k_width + kx];
                        sum += get_pixel_border(src, px, py, c, border) * static_cast<double>(k_val);
                    }
                }
                if constexpr (std::is_same_v<std::remove_cv_t<T_Out>, uint8_t>) {
                    dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(std::clamp(std::round(sum), 0.0, 255.0));
                } else {
                    dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(sum);
                }
            }
        }
    }
}

/**
 * @brief Performs 2D separable convolution using horizontal and vertical 1D kernels.
 * Reduces algorithmic complexity from O(W * H * Kw * Kh) to O(W * H * (Kw + Kh)).
 */
template <PixelType T_In, PixelType T_Out, typename K, size_t Channels>
void convolve_separable(const ImageView<T_In, Channels>& src,
                        const ImageView<T_Out, Channels>& dst,
                        const std::vector<K>& row_kernel,
                        const std::vector<K>& col_kernel,
                        BorderType border = BorderType::REPLICATE) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }
    if (row_kernel.size() % 2 == 0 || col_kernel.size() % 2 == 0) {
        throw std::invalid_argument("Kernel dimensions must be odd.");
    }

    const int64_t w = static_cast<int64_t>(src.width());
    const int64_t h = static_cast<int64_t>(src.height());
    const int64_t kw_half = static_cast<int64_t>(row_kernel.size() / 2);
    const int64_t kh_half = static_cast<int64_t>(col_kernel.size() / 2);

    // Intermediate buffer to hold horizontal pass results.
    // Use double precision to prevent intermediate rounding/clipping issues.
    Image<double, Channels> temp(src.width(), src.height());
    ImageView<double, Channels> temp_view = temp.view();

    // 1. Horizontal Pass (convolve along rows)
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                double sum = 0.0;
                for (int64_t kx = 0; kx < static_cast<int64_t>(row_kernel.size()); ++kx) {
                    const int64_t px = x + kx - kw_half;
                    sum += get_pixel_border(src, px, y, c, border) * static_cast<double>(row_kernel[kx]);
                }
                temp_view(static_cast<size_t>(x), static_cast<size_t>(y), c) = sum;
            }
        }
    }

    // 2. Vertical Pass (convolve temp along columns and store in dst)
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                double sum = 0.0;
                for (int64_t ky = 0; ky < static_cast<int64_t>(col_kernel.size()); ++ky) {
                    const int64_t py = y + ky - kh_half;
                    sum += get_pixel_border(temp_view, x, py, c, border) * static_cast<double>(col_kernel[ky]);
                }
                if constexpr (std::is_same_v<std::remove_cv_t<T_Out>, uint8_t>) {
                    dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(std::clamp(std::round(sum), 0.0, 255.0));
                } else {
                    dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(sum);
                }
            }
        }
    }
}

} // namespace visioncore

#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/convolution.hpp> // for BorderType and get_pixel_border
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Structuring Element (kernel footprint) for morphological operations.
 */
struct StructuringElement {
    size_t width;
    size_t height;
    std::vector<uint8_t> data; // 1 for active footprint pixels, 0 for background

    /**
     * @brief Creates a rectangular structuring element filled with ones.
     */
    static StructuringElement rectangle(size_t w, size_t h) {
        if (w % 2 == 0 || h % 2 == 0) {
            throw std::invalid_argument("Structuring element dimensions must be odd.");
        }
        return { w, h, std::vector<uint8_t>(w * h, 1) };
    }

    /**
     * @brief Creates a cross-shaped structuring element.
     */
    static StructuringElement cross(size_t w, size_t h) {
        if (w % 2 == 0 || h % 2 == 0) {
            throw std::invalid_argument("Structuring element dimensions must be odd.");
        }
        std::vector<uint8_t> d(w * h, 0);
        size_t cy = h / 2;
        size_t cx = w / 2;
        
        for (size_t x = 0; x < w; ++x) {
            d[cy * w + x] = 1;
        }
        for (size_t y = 0; y < h; ++y) {
            d[y * w + cx] = 1;
        }
        return { w, h, d };
    }
};

/**
 * @brief Performs morphological dilation (neighborhood maximum).
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void dilate(const ImageView<T_In, Channels>& src,
            const ImageView<T_Out, Channels>& dst,
            const StructuringElement& se,
            BorderType border = BorderType::REPLICATE) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }

    const int64_t w = static_cast<int64_t>(src.width());
    const int64_t h = static_cast<int64_t>(src.height());
    const int64_t kw_half = static_cast<int64_t>(se.width / 2);
    const int64_t kh_half = static_cast<int64_t>(se.height / 2);

    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                std::remove_const_t<T_In> max_val = std::numeric_limits<std::remove_const_t<T_In>>::lowest();
                for (int64_t ky = 0; ky < static_cast<int64_t>(se.height); ++ky) {
                    for (int64_t kx = 0; kx < static_cast<int64_t>(se.width); ++kx) {
                        if (se.data[ky * se.width + kx] == 1) {
                            const int64_t px = x + kx - kw_half;
                            const int64_t py = y + ky - kh_half;
                            std::remove_const_t<T_In> pixel_val = get_pixel_border(src, px, py, c, border);
                            max_val = std::max(max_val, pixel_val);
                        }
                    }
                }
                dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(max_val);
            }
        }
    }
}

/**
 * @brief Performs morphological erosion (neighborhood minimum).
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void erode(const ImageView<T_In, Channels>& src,
           const ImageView<T_Out, Channels>& dst,
           const StructuringElement& se,
           BorderType border = BorderType::REPLICATE) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        throw std::invalid_argument("Source and destination dimensions do not match.");
    }

    const int64_t w = static_cast<int64_t>(src.width());
    const int64_t h = static_cast<int64_t>(src.height());
    const int64_t kw_half = static_cast<int64_t>(se.width / 2);
    const int64_t kh_half = static_cast<int64_t>(se.height / 2);

    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                std::remove_const_t<T_In> min_val = std::numeric_limits<std::remove_const_t<T_In>>::max();
                for (int64_t ky = 0; ky < static_cast<int64_t>(se.height); ++ky) {
                    for (int64_t kx = 0; kx < static_cast<int64_t>(se.width); ++kx) {
                        if (se.data[ky * se.width + kx] == 1) {
                            const int64_t px = x + kx - kw_half;
                            const int64_t py = y + ky - kh_half;
                            std::remove_const_t<T_In> pixel_val = get_pixel_border(src, px, py, c, border);
                            min_val = std::min(min_val, pixel_val);
                        }
                    }
                }
                dst(static_cast<size_t>(x), static_cast<size_t>(y), c) = static_cast<T_Out>(min_val);
            }
        }
    }
}

/**
 * @brief Performs morphological opening (erosion followed by dilation).
 * Removes small noise and fine detail structures smaller than the structuring element.
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void opening(const ImageView<T_In, Channels>& src,
             const ImageView<T_Out, Channels>& dst,
             const StructuringElement& se,
             BorderType border = BorderType::REPLICATE) {
    Image<std::remove_const_t<T_In>, Channels> temp(src.width(), src.height());
    erode(src, temp.view(), se, border);
    dilate(temp.const_view(), dst, se, border);
}

/**
 * @brief Performs morphological closing (dilation followed by erosion).
 * Closes small holes and gaps in foreground objects.
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void closing(const ImageView<T_In, Channels>& src,
             const ImageView<T_Out, Channels>& dst,
             const StructuringElement& se,
             BorderType border = BorderType::REPLICATE) {
    Image<std::remove_const_t<T_In>, Channels> temp(src.width(), src.height());
    dilate(src, temp.view(), se, border);
    erode(temp.const_view(), dst, se, border);
}

} // namespace visioncore

#pragma once

#include <visioncore/core/concepts.hpp>
#include <visioncore/core/allocator.hpp>
#include <vector>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace visioncore {

// Forward declaration
template <PixelType T, size_t Channels>
class ImageView;

/**
 * @brief Owning, contiguous, aligned multi-channel 2D image class.
 * Uses interleaved memory layout (e.g., RGBRGB... or GrayGray...).
 */
template <PixelType T, size_t Channels = 1>
class Image {
public:
    using value_type = T;
    static constexpr size_t channels = Channels;

    // Constructors
    Image() noexcept = default;
    
    Image(size_t width, size_t height)
        : width_(width), height_(height) {
        if (width_ > 0 && height_ > 0) {
            data_.resize(width_ * height_ * Channels);
        }
    }

    Image(size_t width, size_t height, T default_value)
        : width_(width), height_(height) {
        if (width_ > 0 && height_ > 0) {
            data_.assign(width_ * height_ * Channels, default_value);
        }
    }

    // Rule of 5: Default behaviors are safe as we use std::vector
    ~Image() = default;
    Image(const Image&) = default;
    Image& operator=(const Image&) = default;
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;

    // Factory methods
    static Image from_span(size_t width, size_t height, std::span<const T> source_data) {
        if (source_data.size() < width * height * Channels) {
            throw std::invalid_argument("Source span is too small for the specified image dimensions.");
        }
        Image img(width, height);
        std::copy(source_data.begin(), source_data.begin() + (width * height * Channels), img.data_.begin());
        return img;
    }

    // Accessors
    [[nodiscard]] size_t width() const noexcept { return width_; }
    [[nodiscard]] size_t height() const noexcept { return height_; }
    [[nodiscard]] size_t stride_elements() const noexcept { return width_ * Channels; }
    [[nodiscard]] size_t stride_bytes() const noexcept { return stride_elements() * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    // Flat span access
    [[nodiscard]] std::span<T> span() noexcept { return data_; }
    [[nodiscard]] std::span<const T> span() const noexcept { return data_; }

    // Pixel element access
    [[nodiscard]] T& operator()(size_t x, size_t y, size_t c = 0) noexcept {
        return data_[(y * width_ + x) * Channels + c];
    }

    [[nodiscard]] const T& operator()(size_t x, size_t y, size_t c = 0) const noexcept {
        return data_[(y * width_ + x) * Channels + c];
    }

    [[nodiscard]] T& at(size_t x, size_t y, size_t c = 0) {
        if (x >= width_ || y >= height_ || c >= Channels) {
            throw std::out_of_range("Pixel coordinates out of bounds");
        }
        return operator()(x, y, c);
    }

    [[nodiscard]] const T& at(size_t x, size_t y, size_t c = 0) const {
        if (x >= width_ || y >= height_ || c >= Channels) {
            throw std::out_of_range("Pixel coordinates out of bounds");
        }
        return operator()(x, y, c);
    }

    // Create a non-owning ImageView of the entire image
    [[nodiscard]] ImageView<T, Channels> view() noexcept;
    [[nodiscard]] ImageView<const T, Channels> const_view() const noexcept;

    // Implicit conversions to ImageView
    operator ImageView<T, Channels>() noexcept;
    operator ImageView<const T, Channels>() const noexcept;

private:
    size_t width_ = 0;
    size_t height_ = 0;
    std::vector<T, AlignedAllocator<T, 32>> data_;
};

/**
 * @brief Non-owning view of a 2D image. Supports custom strides for zero-copy ROIs.
 */
template <PixelType T, size_t Channels = 1>
class ImageView {
public:
    using value_type = T;
    static constexpr size_t channels = Channels;

    // Constructors
    ImageView() noexcept = default;
    
    ImageView(T* data, size_t width, size_t height, size_t stride_elements) noexcept
        : data_(data), width_(width), height_(height), stride_elements_(stride_elements) {}

    // Copy constructor for const promotion
    template <typename U>
    requires std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>
    ImageView(const ImageView<U, Channels>& other) noexcept
        : data_(other.data()), width_(other.width()), height_(other.height()), stride_elements_(other.stride_elements()) {}

    // Accessors
    [[nodiscard]] size_t width() const noexcept { return width_; }
    [[nodiscard]] size_t height() const noexcept { return height_; }
    [[nodiscard]] size_t stride_elements() const noexcept { return stride_elements_; }
    [[nodiscard]] size_t stride_bytes() const noexcept { return stride_elements_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

    [[nodiscard]] T* data() const noexcept { return data_; }

    // Pixel element access
    [[nodiscard]] T& operator()(size_t x, size_t y, size_t c = 0) const noexcept {
        return data_[y * stride_elements_ + x * Channels + c];
    }

    [[nodiscard]] T& at(size_t x, size_t y, size_t c = 0) const {
        if (x >= width_ || y >= height_ || c >= Channels) {
            throw std::out_of_range("ImageView pixel coordinates out of bounds");
        }
        return operator()(x, y, c);
    }

    // Get a sub-view (ROI)
    [[nodiscard]] ImageView<T, Channels> sub_view(size_t x, size_t y, size_t roi_width, size_t roi_height) const {
        if (x + roi_width > width_ || y + roi_height > height_) {
            throw std::out_of_range("Requested sub-view region exceeds original image bounds");
        }
        T* sub_data = data_ + y * stride_elements_ + x * Channels;
        return ImageView<T, Channels>(sub_data, roi_width, roi_height, stride_elements_);
    }

private:
    T* data_ = nullptr;
    size_t width_ = 0;
    size_t height_ = 0;
    size_t stride_elements_ = 0;
};

// Implementations of conversion functions
template <PixelType T, size_t Channels>
ImageView<T, Channels> Image<T, Channels>::view() noexcept {
    return ImageView<T, Channels>(data(), width_, height_, stride_elements());
}

template <PixelType T, size_t Channels>
ImageView<const T, Channels> Image<T, Channels>::const_view() const noexcept {
    return ImageView<const T, Channels>(data(), width_, height_, stride_elements());
}

template <PixelType T, size_t Channels>
Image<T, Channels>::operator ImageView<T, Channels>() noexcept {
    return view();
}

template <PixelType T, size_t Channels>
Image<T, Channels>::operator ImageView<const T, Channels>() const noexcept {
    return const_view();
}

} // namespace visioncore

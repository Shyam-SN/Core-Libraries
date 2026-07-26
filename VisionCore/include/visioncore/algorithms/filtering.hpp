#pragma once

#include <visioncore/algorithms/convolution.hpp>
#include <vector>
#include <cmath>
#include <numbers>

namespace visioncore {

/**
 * @brief Applies a uniform Box (mean) filter to the image.
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void box_filter(const ImageView<T_In, Channels>& src,
                const ImageView<T_Out, Channels>& dst,
                size_t k_size,
                BorderType border = BorderType::REPLICATE) {
    if (k_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd.");
    }
    
    std::vector<double> row_kernel(k_size, 1.0 / static_cast<double>(k_size));
    std::vector<double> col_kernel(k_size, 1.0 / static_cast<double>(k_size));

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Generates a 1D Gaussian kernel.
 */
[[nodiscard]] inline std::vector<double> generate_gaussian_kernel1d(size_t k_size, double sigma) {
    if (k_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd.");
    }
    
    std::vector<double> kernel(k_size);
    const double half_k = static_cast<double>(k_size / 2);
    double sum = 0.0;

    // Use default sigma if not specified
    if (sigma <= 0.0) {
        sigma = 0.3 * (static_cast<double>(k_size - 1) * 0.5 - 1.0) + 0.8;
    }

    const double two_sigma_sq = 2.0 * sigma * sigma;
    const double constant = 1.0 / (std::sqrt(2.0 * std::numbers::pi) * sigma);

    for (size_t i = 0; i < k_size; ++i) {
        const double x = static_cast<double>(i) - half_k;
        kernel[i] = constant * std::exp(-(x * x) / two_sigma_sq);
        sum += kernel[i];
    }

    // Normalize
    for (double& val : kernel) {
        val /= sum;
    }

    return kernel;
}

/**
 * @brief Applies a Gaussian Blur filter to the image.
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void gaussian_blur(const ImageView<T_In, Channels>& src,
                   const ImageView<T_Out, Channels>& dst,
                   size_t k_size,
                   double sigma = 0.0,
                   BorderType border = BorderType::REPLICATE) {
    auto kernel = generate_gaussian_kernel1d(k_size, sigma);
    convolve_separable(src, dst, kernel, kernel, border);
}

/**
 * @brief Computes the Sobel X gradient (horizontal derivative).
 * Output elements are typically signed (e.g. float or int16_t) to preserve negative gradients.
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void sobel_x(const ImageView<T, Channels>& src,
             const ImageView<T_Out, Channels>& dst,
             BorderType border = BorderType::REPLICATE) {
    // Sobel X kernel can be separated into row_kernel = [-1, 0, 1] and col_kernel = [1, 2, 1]
    const std::vector<double> row_kernel = {-1.0, 0.0, 1.0};
    const std::vector<double> col_kernel = {1.0, 2.0, 1.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the Sobel Y gradient (vertical derivative).
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void sobel_y(const ImageView<T, Channels>& src,
             const ImageView<T_Out, Channels>& dst,
             BorderType border = BorderType::REPLICATE) {
    // Sobel Y kernel can be separated into row_kernel = [1, 2, 1] and col_kernel = [-1, 0, 1]
    const std::vector<double> row_kernel = {1.0, 2.0, 1.0};
    const std::vector<double> col_kernel = {-1.0, 0.0, 1.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the gradient magnitude from Sobel X and Y components.
 * magnitude = sqrt(dx^2 + dy^2)
 */
template <PixelType T_In, PixelType T_Out, size_t Channels>
void sobel_magnitude(const ImageView<T_In, Channels>& dx,
                     const ImageView<T_In, Channels>& dy,
                     const ImageView<T_Out, Channels>& magnitude) {
    if (dx.width() != dy.width() || dx.height() != dy.height() || dx.width() != magnitude.width() || dx.height() != magnitude.height()) {
        throw std::invalid_argument("All input and output dimensions must match.");
    }

    const size_t w = dx.width();
    const size_t h = dx.height();

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < Channels; ++c) {
                double val_x = static_cast<double>(dx(x, y, c));
                double val_y = static_cast<double>(dy(x, y, c));
                double mag = std::hypot(val_x, val_y);
                magnitude(x, y, c) = static_cast<T_Out>(std::clamp(mag, 0.0, 255.0));
            }
        }
    }
}

} // namespace visioncore

#pragma once

#include <visioncore/algorithms/convolution.hpp>
#include <vector>

namespace visioncore {

/**
 * @brief Computes the Prewitt X gradient (horizontal derivative).
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void prewitt_x(const ImageView<T, Channels>& src,
               const ImageView<T_Out, Channels>& dst,
               BorderType border = BorderType::REPLICATE) {
    // Prewitt X kernel is separable: row = [-1, 0, 1], col = [1, 1, 1]^T
    const std::vector<double> row_kernel = {-1.0, 0.0, 1.0};
    const std::vector<double> col_kernel = {1.0, 1.0, 1.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the Prewitt Y gradient (vertical derivative).
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void prewitt_y(const ImageView<T, Channels>& src,
               const ImageView<T_Out, Channels>& dst,
               BorderType border = BorderType::REPLICATE) {
    // Prewitt Y kernel is separable: row = [1, 1, 1], col = [-1, 0, 1]^T
    const std::vector<double> row_kernel = {1.0, 1.0, 1.0};
    const std::vector<double> col_kernel = {-1.0, 0.0, 1.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the Scharr X gradient (horizontal derivative, higher angular accuracy than Sobel).
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void scharr_x(const ImageView<T, Channels>& src,
              const ImageView<T_Out, Channels>& dst,
              BorderType border = BorderType::REPLICATE) {
    // Scharr X kernel is separable: row = [-1, 0, 1], col = [3, 10, 3]^T
    const std::vector<double> row_kernel = {-1.0, 0.0, 1.0};
    const std::vector<double> col_kernel = {3.0, 10.0, 3.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the Scharr Y gradient (vertical derivative).
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void scharr_y(const ImageView<T, Channels>& src,
              const ImageView<T_Out, Channels>& dst,
              BorderType border = BorderType::REPLICATE) {
    // Scharr Y kernel is separable: row = [3, 10, 3], col = [-1, 0, 1]^T
    const std::vector<double> row_kernel = {3.0, 10.0, 3.0};
    const std::vector<double> col_kernel = {-1.0, 0.0, 1.0};

    convolve_separable(src, dst, row_kernel, col_kernel, border);
}

/**
 * @brief Computes the Laplacian of an image (second derivative).
 * Uses the 3x3 isotropic discrete Laplacian filter kernel.
 */
template <PixelType T, PixelType T_Out, size_t Channels>
void laplacian(const ImageView<T, Channels>& src,
               const ImageView<T_Out, Channels>& dst,
               BorderType border = BorderType::REPLICATE) {
    // 3x3 Laplacian kernel with negative center (weight = -8)
    const std::vector<double> kernel = {
        1.0,  1.0, 1.0,
        1.0, -8.0, 1.0,
        1.0,  1.0, 1.0
    };

    convolve2d(src, dst, kernel, 3, 3, border);
}

} // namespace visioncore

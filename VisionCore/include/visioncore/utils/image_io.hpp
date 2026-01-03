#pragma once

#include <visioncore/core/image.hpp>
#include <string>
#include <optional>
#include <cstdint>

namespace visioncore::utils {

/**
 * @brief Loads an image from file as a single-channel Grayscale image.
 * @param file_path Path to the image file.
 * @return Optional containing the loaded Image, or std::nullopt if loading failed.
 */
[[nodiscard]] std::optional<Image<uint8_t, 1>> load_grayscale(const std::string& file_path);

/**
 * @brief Loads an image from file as a 3-channel RGB image (converting from OpenCV's BGR).
 * @param file_path Path to the image file.
 * @return Optional containing the loaded Image, or std::nullopt if loading failed.
 */
[[nodiscard]] std::optional<Image<uint8_t, 3>> load_rgb(const std::string& file_path);

/**
 * @brief Saves a single-channel Grayscale image to file.
 */
bool save_image(const std::string& file_path, const ImageView<const uint8_t, 1>& image);

/**
 * @brief Saves a 3-channel RGB image to file (converting to BGR for OpenCV).
 */
bool save_image(const std::string& file_path, const ImageView<const uint8_t, 3>& image);

} // namespace visioncore::utils

#pragma once

#include <type_traits>
#include <concepts>
#include <cstdint>

namespace visioncore {

/**
 * @brief Concept enforcing that a type is valid for pixel representations.
 * Crucial for avoiding instantiations with non-numeric structures (e.g. pointers, arbitrary classes).
 */
template <typename T>
concept PixelType = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

} // namespace visioncore

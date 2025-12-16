#pragma once

#include <type_traits>
#include <concepts>

namespace geometrycore {

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

} // namespace geometrycore

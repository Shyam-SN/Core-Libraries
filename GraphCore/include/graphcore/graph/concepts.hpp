#pragma once

#include <type_traits>
#include <concepts>

namespace graphcore {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

} // namespace graphcore

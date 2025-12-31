#pragma once

#include <deeplearningcore/core/concepts.hpp>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace deeplearningcore {

template <Numeric T>
class Storage {
public:
    explicit Storage(size_t size)
        : size_(size) {
        if (size > 0) {
            data_ = std::vector<T>(size, T{0});
        }
    }

    Storage(const T* data, size_t size)
        : size_(size) {
        if (size > 0) {
            data_ = std::vector<T>(data, data + size);
        }
    }

    Storage(std::initializer_list<T> list)
        : data_(list), size_(list.size()) {}

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] T& operator[](size_t idx) { return data_[idx]; }
    [[nodiscard]] const T& operator[](size_t idx) const { return data_[idx]; }

    void fill(T val) {
        std::fill(data_.begin(), data_.end(), val);
    }

private:
    std::vector<T> data_;
    size_t size_{0};
};

} // namespace deeplearningcore

#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>

namespace deeplearningcore {

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<size_t> dims)
        : dims_(dims) {
        compute_strides();
    }

    explicit Shape(std::vector<size_t> dims)
        : dims_(std::move(dims)) {
        compute_strides();
    }

    [[nodiscard]] size_t ndim() const noexcept { return dims_.size(); }
    [[nodiscard]] size_t size() const noexcept { return dims_.size(); }

    [[nodiscard]] const std::vector<size_t>& dims() const noexcept { return dims_; }
    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return strides_; }

    [[nodiscard]] size_t operator[](size_t i) const {
        if (i >= dims_.size()) {
            throw std::out_of_range("Shape dimension index out of range");
        }
        return dims_[i];
    }

    [[nodiscard]] size_t stride(size_t i) const {
        if (i >= strides_.size()) {
            throw std::out_of_range("Shape stride index out of range");
        }
        return strides_[i];
    }

    [[nodiscard]] size_t numel() const noexcept {
        if (dims_.empty()) return 0;
        size_t n = 1;
        for (auto d : dims_) n *= d;
        return n;
    }

    [[nodiscard]] size_t get_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("Index dimension mismatch");
        }
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= dims_[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

    [[nodiscard]] bool operator==(const Shape& other) const noexcept {
        return dims_ == other.dims_;
    }

    [[nodiscard]] bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            ss << dims_[i] << (i + 1 < dims_.size() ? ", " : "");
        }
        ss << ")";
        return ss.str();
    }

    static bool can_broadcast(const Shape& s1, const Shape& s2) {
        size_t n1 = s1.ndim();
        size_t n2 = s2.ndim();
        size_t max_n = std::max(n1, n2);
        for (size_t i = 0; i < max_n; ++i) {
            size_t d1 = (i < n1) ? s1[n1 - 1 - i] : 1;
            size_t d2 = (i < n2) ? s2[n2 - 1 - i] : 1;
            if (d1 != d2 && d1 != 1 && d2 != 1) return false;
        }
        return true;
    }

    static Shape broadcast(const Shape& s1, const Shape& s2) {
        size_t n1 = s1.ndim();
        size_t n2 = s2.ndim();
        size_t max_n = std::max(n1, n2);
        std::vector<size_t> res_dims(max_n);

        for (size_t i = 0; i < max_n; ++i) {
            size_t d1 = (i < n1) ? s1[n1 - 1 - i] : 1;
            size_t d2 = (i < n2) ? s2[n2 - 1 - i] : 1;
            if (d1 != d2 && d1 != 1 && d2 != 1) {
                throw std::invalid_argument("Cannot broadcast shapes " + s1.to_string() + " and " + s2.to_string());
            }
            res_dims[max_n - 1 - i] = std::max(d1, d2);
        }
        return Shape(res_dims);
    }

private:
    void compute_strides() {
        strides_.resize(dims_.size());
        if (dims_.empty()) return;
        size_t stride = 1;
        for (int i = static_cast<int>(dims_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= dims_[i];
        }
    }

    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
};

} // namespace deeplearningcore

#pragma once

#include <vector>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace pointcloudcore {

class Tensor {
public:
    std::vector<double> data;
    std::vector<size_t> shape;

    Tensor() = default;
    Tensor(std::vector<size_t> shape, double val = 0.0) : shape(shape) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        data.resize(total, val);
    }

    [[nodiscard]] size_t size() const noexcept { return data.size(); }
    [[nodiscard]] size_t ndim() const noexcept { return shape.size(); }

    [[nodiscard]] static Tensor zeros(std::vector<size_t> shape) {
        return Tensor(shape, 0.0);
    }

    [[nodiscard]] static Tensor ones(std::vector<size_t> shape) {
        return Tensor(shape, 1.0);
    }

    [[nodiscard]] double at(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) throw std::invalid_argument("Index dimension mismatch");
        size_t offset = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            offset += indices[i] * stride;
            stride *= shape[i];
        }
        return data[offset];
    }

    void set(const std::vector<size_t>& indices, double val) {
        if (indices.size() != shape.size()) throw std::invalid_argument("Index dimension mismatch");
        size_t offset = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            offset += indices[i] * stride;
            stride *= shape[i];
        }
        data[offset] = val;
    }
};

} // namespace pointcloudcore

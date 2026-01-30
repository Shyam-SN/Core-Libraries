#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class MaxPool2D : public Module {
public:
    explicit MaxPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0);

    Tensor forward(const Tensor& input) override;

private:
    size_t kernel_h_, kernel_w_;
    size_t stride_h_, stride_w_;
    size_t pad_h_, pad_w_;
};

class AvgPool2D : public Module {
public:
    explicit AvgPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0);

    Tensor forward(const Tensor& input) override;

private:
    size_t kernel_h_, kernel_w_;
    size_t stride_h_, stride_w_;
    size_t pad_h_, pad_w_;
};

} // namespace deeplearningcore

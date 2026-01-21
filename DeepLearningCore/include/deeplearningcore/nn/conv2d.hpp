#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class Conv2D : public Module {
public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, bool use_bias = true);

    Conv2D(size_t in_channels, size_t out_channels, std::pair<size_t, size_t> kernel_size,
           std::pair<size_t, size_t> stride = {1, 1}, std::pair<size_t, size_t> padding = {0, 0}, bool use_bias = true);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> weight() const { return weight_; }
    [[nodiscard]] std::shared_ptr<Tensor> bias() const { return bias_; }
    [[nodiscard]] size_t in_channels() const noexcept { return in_channels_; }
    [[nodiscard]] size_t out_channels() const noexcept { return out_channels_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_h_, kernel_w_;
    size_t stride_h_, stride_w_;
    size_t pad_h_, pad_w_;
    bool use_bias_;

    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
};

} // namespace deeplearningcore

#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> weight() const { return weight_; }
    [[nodiscard]] std::shared_ptr<Tensor> bias() const { return bias_; }
    [[nodiscard]] size_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] size_t out_features() const noexcept { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
};

} // namespace deeplearningcore

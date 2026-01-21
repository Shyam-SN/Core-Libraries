#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class BatchNorm1D : public Module {
public:
    explicit BatchNorm1D(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> gamma() const { return gamma_; }
    [[nodiscard]] std::shared_ptr<Tensor> beta() const { return beta_; }
    [[nodiscard]] std::shared_ptr<Tensor> running_mean() const { return running_mean_; }
    [[nodiscard]] std::shared_ptr<Tensor> running_var() const { return running_var_; }
    [[nodiscard]] size_t num_features() const noexcept { return num_features_; }

private:
    size_t num_features_;
    float eps_;
    float momentum_;

    std::shared_ptr<Tensor> gamma_;
    std::shared_ptr<Tensor> beta_;
    std::shared_ptr<Tensor> running_mean_;
    std::shared_ptr<Tensor> running_var_;
};

class BatchNorm2D : public Module {
public:
    explicit BatchNorm2D(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> gamma() const { return gamma_; }
    [[nodiscard]] std::shared_ptr<Tensor> beta() const { return beta_; }
    [[nodiscard]] size_t num_features() const noexcept { return num_features_; }

private:
    size_t num_features_;
    float eps_;
    float momentum_;

    std::shared_ptr<Tensor> gamma_;
    std::shared_ptr<Tensor> beta_;
    std::shared_ptr<Tensor> running_mean_;
    std::shared_ptr<Tensor> running_var_;
};

} // namespace deeplearningcore

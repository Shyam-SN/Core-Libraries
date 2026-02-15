#pragma once

#include <deeplearningcore/nn/module.hpp>
#include <deeplearningcore/nn/linear.hpp>

namespace deeplearningcore {

class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t normalized_shape, float eps = 1e-5f);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> gamma() const { return gamma_; }
    [[nodiscard]] std::shared_ptr<Tensor> beta() const { return beta_; }
    [[nodiscard]] size_t normalized_shape() const noexcept { return normalized_shape_; }

private:
    size_t normalized_shape_;
    float eps_;
    std::shared_ptr<Tensor> gamma_;
    std::shared_ptr<Tensor> beta_;
};

class PositionalEncoding : public Module {
public:
    PositionalEncoding(size_t d_model, size_t max_len = 5000);

    Tensor forward(const Tensor& input) override;

private:
    Tensor pe_;
};

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);

    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask = nullptr);
    Tensor forward(const Tensor& input) override { return forward(input, input, input); }

    [[nodiscard]] size_t d_model() const noexcept { return d_model_; }
    [[nodiscard]] size_t num_heads() const noexcept { return num_heads_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;

    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;
};

} // namespace deeplearningcore

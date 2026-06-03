#include <deeplearningcore/nn/transformer.hpp>
#include <cmath>

namespace deeplearningcore {

LayerNorm::LayerNorm(size_t normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {

    gamma_ = std::make_shared<Tensor>(Tensor::ones(Shape{1, normalized_shape}, true));
    beta_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, normalized_shape}, true));

    register_parameter("gamma", gamma_);
    register_parameter("beta", beta_);
}

Tensor LayerNorm::forward(const Tensor& input) {
    size_t last_dim = input.shape()[input.ndim() - 1];
    size_t outer_numel = input.numel() / last_dim;

    Tensor out(input.shape(), input.requires_grad() && NoGradScope::is_grad_enabled());

    for (size_t o = 0; o < outer_numel; ++o) {
        size_t offset = o * last_dim;
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            mean += input[offset + i];
        }
        mean /= static_cast<float>(last_dim);

        float var = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float diff = input[offset + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(last_dim);

        float inv_std = 1.0f / std::sqrt(var + eps_);

        for (size_t i = 0; i < last_dim; ++i) {
            float x_hat = (input[offset + i] - mean) * inv_std;
            out[offset + i] = (*gamma_)[i] * x_hat + (*beta_)[i];
        }
    }

    return out;
}

PositionalEncoding::PositionalEncoding(size_t d_model, size_t max_len)
    : pe_(Shape{max_len, d_model}, false) {

    for (size_t pos = 0; pos < max_len; ++pos) {
        for (size_t i = 0; i < d_model; i += 2) {
            float div_term = std::exp(static_cast<float>(i) * (-std::log(10000.0f) / static_cast<float>(d_model)));
            pe_.set_at({pos, i}, std::sin(static_cast<float>(pos) * div_term));
            if (i + 1 < d_model) {
                pe_.set_at({pos, i + 1}, std::cos(static_cast<float>(pos) * div_term));
            }
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& input) {
    size_t seq_len = input.shape()[0];
    size_t d_model = input.shape()[1];

    Tensor pe_sub(Shape{seq_len, d_model}, false);
    for (size_t i = 0; i < seq_len * d_model; ++i) {
        pe_sub[i] = pe_[i];
    }
    return input.add(pe_sub);
}

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads), head_dim_(d_model / num_heads) {

    q_proj_ = std::make_shared<Linear>(d_model, d_model);
    k_proj_ = std::make_shared<Linear>(d_model, d_model);
    v_proj_ = std::make_shared<Linear>(d_model, d_model);
    out_proj_ = std::make_shared<Linear>(d_model, d_model);

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    (void)mask;
    Tensor Q = (*q_proj_)(query);
    Tensor K = (*k_proj_)(key);
    Tensor V = (*v_proj_)(value);

    // Scaled dot-product attention scores = Q * K^T / sqrt(head_dim)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    Tensor scores = Q.matmul(K.transpose()).mul(scale);

    Tensor attn_weights = scores.softmax(-1);
    Tensor context = attn_weights.matmul(V);

    return (*out_proj_)(context);
}

} // namespace deeplearningcore

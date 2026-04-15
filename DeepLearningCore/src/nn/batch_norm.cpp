#include <deeplearningcore/nn/batch_norm.hpp>
#include <cmath>

namespace deeplearningcore {

BatchNorm1D::BatchNorm1D(size_t num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {

    gamma_ = std::make_shared<Tensor>(Tensor::ones(Shape{1, num_features}, true));
    beta_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, num_features}, true));
    running_mean_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, num_features}, false));
    running_var_ = std::make_shared<Tensor>(Tensor::ones(Shape{1, num_features}, false));

    register_parameter("gamma", gamma_);
    register_parameter("beta", beta_);
}

Tensor BatchNorm1D::forward(const Tensor& input) {
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];

    Tensor out(input.shape(), input.requires_grad() && NoGradScope::is_grad_enabled());

    if (is_training()) {
        for (size_t c = 0; c < C; ++c) {
            float mean_val = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                mean_val += input[n * C + c];
            }
            mean_val /= static_cast<float>(N);

            float var_val = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                float diff = input[n * C + c] - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<float>(N);

            // Update running stats
            (*running_mean_)[c] = (1.0f - momentum_) * (*running_mean_)[c] + momentum_ * mean_val;
            (*running_var_)[c] = (1.0f - momentum_) * (*running_var_)[c] + momentum_ * var_val;

            float inv_std = 1.0f / std::sqrt(var_val + eps_);
            float g = (*gamma_)[c];
            float b = (*beta_)[c];

            for (size_t n = 0; n < N; ++n) {
                float x_hat = (input[n * C + c] - mean_val) * inv_std;
                out[n * C + c] = g * x_hat + b;
            }
        }
    } else {
        for (size_t c = 0; c < C; ++c) {
            float mean_val = (*running_mean_)[c];
            float var_val = (*running_var_)[c];
            float inv_std = 1.0f / std::sqrt(var_val + eps_);
            float g = (*gamma_)[c];
            float b = (*beta_)[c];

            for (size_t n = 0; n < N; ++n) {
                float x_hat = (input[n * C + c] - mean_val) * inv_std;
                out[n * C + c] = g * x_hat + b;
            }
        }
    }

    return out;
}

BatchNorm2D::BatchNorm2D(size_t num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {

    gamma_ = std::make_shared<Tensor>(Tensor::ones(Shape{1, num_features, 1, 1}, true));
    beta_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, num_features, 1, 1}, true));
    running_mean_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, num_features, 1, 1}, false));
    running_var_ = std::make_shared<Tensor>(Tensor::ones(Shape{1, num_features, 1, 1}, false));

    register_parameter("gamma", gamma_);
    register_parameter("beta", beta_);
}

Tensor BatchNorm2D::forward(const Tensor& input) {
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];
    size_t spatial_size = H * W;

    Tensor out(input.shape(), input.requires_grad() && NoGradScope::is_grad_enabled());

    for (size_t c = 0; c < C; ++c) {
        float mean_val = (*running_mean_)[c];
        float var_val = (*running_var_)[c];

        if (is_training()) {
            mean_val = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    mean_val += input[((n * C + c) * H * W) + hw];
                }
            }
            mean_val /= static_cast<float>(N * spatial_size);

            var_val = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    float diff = input[((n * C + c) * H * W) + hw] - mean_val;
                    var_val += diff * diff;
                }
            }
            var_val /= static_cast<float>(N * spatial_size);

            (*running_mean_)[c] = (1.0f - momentum_) * (*running_mean_)[c] + momentum_ * mean_val;
            (*running_var_)[c] = (1.0f - momentum_) * (*running_var_)[c] + momentum_ * var_val;
        }

        float inv_std = 1.0f / std::sqrt(var_val + eps_);
        float g = (*gamma_)[c];
        float b = (*beta_)[c];

        for (size_t n = 0; n < N; ++n) {
            for (size_t hw = 0; hw < spatial_size; ++hw) {
                size_t idx = ((n * C + c) * H * W) + hw;
                float x_hat = (input[idx] - mean_val) * inv_std;
                out[idx] = g * x_hat + b;
            }
        }
    }

    return out;
}

} // namespace deeplearningcore

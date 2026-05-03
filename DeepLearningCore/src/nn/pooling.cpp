#include <deeplearningcore/nn/pooling.hpp>
#include <algorithm>

namespace deeplearningcore {

MaxPool2D::MaxPool2D(size_t kernel_size, size_t stride, size_t padding)
    : kernel_h_(kernel_size), kernel_w_(kernel_size),
      stride_h_(stride == 0 ? kernel_size : stride), stride_w_(stride == 0 ? kernel_size : stride),
      pad_h_(padding), pad_w_(padding) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    size_t H_out = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    size_t W_out = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    Tensor out(Shape{N, C, H_out, W_out}, input.requires_grad() && NoGradScope::is_grad_enabled());

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float max_val = -1e9f;
                    for (size_t kh = 0; kh < kernel_h_; ++kh) {
                        for (size_t kw = 0; kw < kernel_w_; ++kw) {
                            int hi = static_cast<int>(ho * stride_h_ + kh) - static_cast<int>(pad_h_);
                            int wi = static_cast<int>(wo * stride_w_ + kw) - static_cast<int>(pad_w_);
                            if (hi >= 0 && hi < static_cast<int>(H) && wi >= 0 && wi < static_cast<int>(W)) {
                                size_t idx = ((n * C + c) * H + hi) * W + wi;
                                max_val = std::max(max_val, input[idx]);
                            }
                        }
                    }
                    size_t out_idx = ((n * C + c) * H_out + ho) * W_out + wo;
                    out[out_idx] = max_val;
                }
            }
        }
    }

    return out;
}

AvgPool2D::AvgPool2D(size_t kernel_size, size_t stride, size_t padding)
    : kernel_h_(kernel_size), kernel_w_(kernel_size),
      stride_h_(stride == 0 ? kernel_size : stride), stride_w_(stride == 0 ? kernel_size : stride),
      pad_h_(padding), pad_w_(padding) {}

Tensor AvgPool2D::forward(const Tensor& input) {
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    size_t H_out = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    size_t W_out = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    Tensor out(Shape{N, C, H_out, W_out}, input.requires_grad() && NoGradScope::is_grad_enabled());

    float k_size = static_cast<float>(kernel_h_ * kernel_w_);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float sum_val = 0.0f;
                    for (size_t kh = 0; kh < kernel_h_; ++kh) {
                        for (size_t kw = 0; kw < kernel_w_; ++kw) {
                            int hi = static_cast<int>(ho * stride_h_ + kh) - static_cast<int>(pad_h_);
                            int wi = static_cast<int>(wo * stride_w_ + kw) - static_cast<int>(pad_w_);
                            if (hi >= 0 && hi < static_cast<int>(H) && wi >= 0 && wi < static_cast<int>(W)) {
                                size_t idx = ((n * C + c) * H + hi) * W + wi;
                                sum_val += input[idx];
                            }
                        }
                    }
                    size_t out_idx = ((n * C + c) * H_out + ho) * W_out + wo;
                    out[out_idx] = sum_val / k_size;
                }
            }
        }
    }

    return out;
}

} // namespace deeplearningcore

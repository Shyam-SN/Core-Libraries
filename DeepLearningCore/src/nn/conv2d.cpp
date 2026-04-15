#include <deeplearningcore/nn/conv2d.hpp>
#include <cmath>

namespace deeplearningcore {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding, bool use_bias)
    : Conv2D(in_channels, out_channels, {kernel_size, kernel_size}, {stride, stride}, {padding, padding}, use_bias) {}

Conv2D::Conv2D(size_t in_channels, size_t out_channels, std::pair<size_t, size_t> kernel_size,
               std::pair<size_t, size_t> stride, std::pair<size_t, size_t> padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_size.first), kernel_w_(kernel_size.second),
      stride_h_(stride.first), stride_w_(stride.second),
      pad_h_(padding.first), pad_w_(padding.second),
      use_bias_(use_bias) {

    float k = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_h_ * kernel_w_));
    weight_ = std::make_shared<Tensor>(Tensor::rand(Shape{out_channels, in_channels, kernel_h_, kernel_w_}, -k, k, true));
    register_parameter("weight", weight_);

    if (use_bias_) {
        bias_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, out_channels, 1, 1}, true));
        register_parameter("bias", bias_);
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2D requires 4D input [N, C_in, H, W]");
    }
    size_t N = input.shape()[0];
    size_t C_in = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    size_t H_out = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    size_t W_out = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    bool req_grad = (input.requires_grad() || weight_->requires_grad()) && NoGradScope::is_grad_enabled();
    Tensor out(Shape{N, out_channels_, H_out, W_out}, req_grad);

    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float sum_val = (use_bias_ && bias_) ? (*bias_)[oc] : 0.0f;

                    for (size_t ic = 0; ic < C_in; ++ic) {
                        for (size_t kh = 0; kh < kernel_h_; ++kh) {
                            for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                int hi = static_cast<int>(ho * stride_h_ + kh) - static_cast<int>(pad_h_);
                                int wi = static_cast<int>(wo * stride_w_ + kw) - static_cast<int>(pad_w_);

                                if (hi >= 0 && hi < static_cast<int>(H) && wi >= 0 && wi < static_cast<int>(W)) {
                                    size_t in_idx = ((n * C_in + ic) * H + hi) * W + wi;
                                    size_t w_idx = ((oc * C_in + ic) * kernel_h_ + kh) * kernel_w_ + kw;
                                    sum_val += input[in_idx] * (*weight_)[w_idx];
                                }
                            }
                        }
                    }

                    size_t out_idx = ((n * out_channels_ + oc) * H_out + ho) * W_out + wo;
                    out[out_idx] = sum_val;
                }
            }
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (input.requires_grad()) node->add_parent(input.autograd_node());
        if (weight_->requires_grad()) node->add_parent(weight_->autograd_node());

        auto in_ptr = std::make_shared<Tensor>(input);
        auto w_ptr = weight_;
        auto b_ptr = bias_;

        size_t kh = kernel_h_, kw = kernel_w_;
        size_t sh = stride_h_, sw = stride_w_;
        size_t ph = pad_h_, pw = pad_w_;

        node->set_backward_fn([in_ptr, w_ptr, N, C_in, H, W, H_out, W_out, kh, kw, sh, sw, ph, pw](const Tensor& grad) {
            if (in_ptr->requires_grad()) {
                Tensor d_in(in_ptr->shape(), 0.0f, false);
                for (size_t n = 0; n < N; ++n) {
                    for (size_t oc = 0; oc < w_ptr->shape()[0]; ++oc) {
                        for (size_t ho = 0; ho < H_out; ++ho) {
                            for (size_t wo = 0; wo < W_out; ++wo) {
                                float g = grad[((n * w_ptr->shape()[0] + oc) * H_out + ho) * W_out + wo];
                                for (size_t ic = 0; ic < C_in; ++ic) {
                                    for (size_t k_h = 0; k_h < kh; ++k_h) {
                                        for (size_t k_w = 0; k_w < kw; ++k_w) {
                                            int hi = static_cast<int>(ho * sh + k_h) - static_cast<int>(ph);
                                            int wi = static_cast<int>(wo * sw + k_w) - static_cast<int>(pw);
                                            if (hi >= 0 && hi < static_cast<int>(H) && wi >= 0 && wi < static_cast<int>(W)) {
                                                size_t in_idx = ((n * C_in + ic) * H + hi) * W + wi;
                                                size_t w_idx = ((oc * C_in + ic) * kh + k_h) * kw + k_w;
                                                d_in[in_idx] += g * (*w_ptr)[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                in_ptr->backward(d_in);
            }

            if (w_ptr->requires_grad()) {
                Tensor d_w(w_ptr->shape(), 0.0f, false);
                for (size_t n = 0; n < N; ++n) {
                    for (size_t oc = 0; oc < w_ptr->shape()[0]; ++oc) {
                        for (size_t ho = 0; ho < H_out; ++ho) {
                            for (size_t wo = 0; wo < W_out; ++wo) {
                                float g = grad[((n * w_ptr->shape()[0] + oc) * H_out + ho) * W_out + wo];
                                for (size_t ic = 0; ic < C_in; ++ic) {
                                    for (size_t k_h = 0; k_h < kh; ++k_h) {
                                        for (size_t k_w = 0; k_w < kw; ++k_w) {
                                            int hi = static_cast<int>(ho * sh + k_h) - static_cast<int>(ph);
                                            int wi = static_cast<int>(wo * sw + k_w) - static_cast<int>(pw);
                                            if (hi >= 0 && hi < static_cast<int>(H) && wi >= 0 && wi < static_cast<int>(W)) {
                                                size_t in_idx = ((n * C_in + ic) * H + hi) * W + wi;
                                                size_t w_idx = ((oc * C_in + ic) * kh + k_h) * kw + k_w;
                                                d_w[w_idx] += g * (*in_ptr)[in_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                w_ptr->backward(d_w);
            }
        });
        out.set_autograd_node(node);
    }

    return out;
}

} // namespace deeplearningcore

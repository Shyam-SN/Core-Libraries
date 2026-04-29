#include <deeplearningcore/nn/linear.hpp>
#include <cmath>

namespace deeplearningcore {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    
    // He/Kaiming Uniform initialization
    float stddev = 1.0f / std::sqrt(static_cast<float>(in_features));
    weight_ = std::make_shared<Tensor>(Tensor::rand(Shape{in_features, out_features}, -stddev, stddev, true));
    register_parameter("weight", weight_);

    if (use_bias_) {
        bias_ = std::make_shared<Tensor>(Tensor::zeros(Shape{1, out_features}, true));
        register_parameter("bias", bias_);
    }
}

Tensor Linear::forward(const Tensor& input) {
    Tensor out = input.matmul(*weight_);
    if (use_bias_ && bias_) {
        out = out.add(*bias_);
    }
    return out;
}

} // namespace deeplearningcore

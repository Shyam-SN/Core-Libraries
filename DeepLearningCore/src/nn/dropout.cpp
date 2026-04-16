#include <deeplearningcore/nn/dropout.hpp>
#include <random>

namespace deeplearningcore {

Dropout::Dropout(float p) : p_(p) {}

Tensor Dropout::forward(const Tensor& input) {
    if (!is_training() || p_ <= 0.0f) {
        return input;
    }

    Tensor out(input.shape(), input.requires_grad() && NoGradScope::is_grad_enabled());
    static std::mt19937 gen(42);
    std::bernoulli_distribution dis(1.0f - p_);
    float scale = 1.0f / (1.0f - p_);

    for (size_t i = 0; i < input.numel(); ++i) {
        bool keep = dis(gen);
        out[i] = keep ? input[i] * scale : 0.0f;
    }

    return out;
}

} // namespace deeplearningcore

#include <deeplearningcore/losses/losses.hpp>
#include <cmath>
#include <algorithm>

namespace deeplearningcore {

Tensor MSELoss::forward(const Tensor& input, const Tensor& target) {
    Tensor diff = input.sub(target);
    Tensor sq_diff = diff.mul(diff);
    return sq_diff.mean();
}

Tensor CrossEntropyLoss::forward(const Tensor& input, const Tensor& target) {
    Tensor log_s = input.log_softmax(-1);
    Tensor nll = log_s.mul(target).mul(-1.0f);
    return nll.sum().div(static_cast<float>(input.shape()[0]));
}

Tensor BCELoss::forward(const Tensor& input, const Tensor& target) {
    Tensor eps_tensor = Tensor::scalar(1e-7f);
    Tensor one_minus_input = Tensor::scalar(1.0f).sub(input).add(eps_tensor);
    Tensor input_clamp = input.add(eps_tensor);

    Tensor term1 = target.mul(input_clamp.log_softmax(-1));
    Tensor term2 = Tensor::scalar(1.0f).sub(target).mul(one_minus_input.log_softmax(-1));

    Tensor loss = term1.add(term2).mul(-1.0f);
    return loss.mean();
}

Tensor L1Loss::forward(const Tensor& input, const Tensor& target) {
    Tensor diff = input.sub(target);
    Tensor abs_diff(diff.shape(), diff.requires_grad() && NoGradScope::is_grad_enabled());
    for (size_t i = 0; i < diff.numel(); ++i) {
        abs_diff[i] = std::abs(diff[i]);
    }
    return abs_diff.mean();
}

} // namespace deeplearningcore

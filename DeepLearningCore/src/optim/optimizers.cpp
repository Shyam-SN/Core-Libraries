#include <deeplearningcore/optim/optimizers.hpp>
#include <cmath>

namespace deeplearningcore {

SGD::SGD(std::vector<std::shared_ptr<Tensor>> params, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(params), lr), momentum_(momentum), weight_decay_(weight_decay) {
    velocities_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        velocities_[i] = std::vector<float>(params_[i]->numel(), 0.0f);
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p->has_grad()) continue;

        auto g = p->grad();
        size_t n = p->numel();

        for (size_t j = 0; j < n; ++j) {
            float grad_val = (*g)[j];
            if (weight_decay_ != 0.0f) {
                grad_val += weight_decay_ * (*p)[j];
            }
            if (momentum_ != 0.0f) {
                velocities_[i][j] = momentum_ * velocities_[i][j] + grad_val;
                grad_val = velocities_[i][j];
            }
            (*p)[j] -= lr_ * grad_val;
        }
    }
}

Adam::Adam(std::vector<std::shared_ptr<Tensor>> params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        m_[i] = std::vector<float>(params_[i]->numel(), 0.0f);
        v_[i] = std::vector<float>(params_[i]->numel(), 0.0f);
    }
}

void Adam::step() {
    t_++;
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p->has_grad()) continue;

        auto g = p->grad();
        size_t n = p->numel();

        for (size_t j = 0; j < n; ++j) {
            float grad_val = (*g)[j];
            if (weight_decay_ != 0.0f) {
                grad_val += weight_decay_ * (*p)[j];
            }

            m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad_val;
            v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad_val * grad_val;

            float m_hat = m_[i][j] / bias_correction1;
            float v_hat = v_[i][j] / bias_correction2;

            (*p)[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

RMSProp::RMSProp(std::vector<std::shared_ptr<Tensor>> params, float lr, float alpha, float eps, float weight_decay)
    : Optimizer(std::move(params), lr), alpha_(alpha), eps_(eps), weight_decay_(weight_decay) {
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        v_[i] = std::vector<float>(params_[i]->numel(), 0.0f);
    }
}

void RMSProp::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p->has_grad()) continue;

        auto g = p->grad();
        size_t n = p->numel();

        for (size_t j = 0; j < n; ++j) {
            float grad_val = (*g)[j];
            if (weight_decay_ != 0.0f) {
                grad_val += weight_decay_ * (*p)[j];
            }

            v_[i][j] = alpha_ * v_[i][j] + (1.0f - alpha_) * grad_val * grad_val;
            (*p)[j] -= lr_ * grad_val / (std::sqrt(v_[i][j]) + eps_);
        }
    }
}

} // namespace deeplearningcore

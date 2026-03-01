#pragma once

#include <deeplearningcore/core/tensor.hpp>
#include <vector>
#include <memory>

namespace deeplearningcore {

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<Tensor>> params, float lr)
        : params_(std::move(params)), lr_(lr) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() {
        for (auto& p : params_) {
            p->zero_grad();
        }
    }

    [[nodiscard]] float lr() const noexcept { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
    float lr_;
};

} // namespace deeplearningcore

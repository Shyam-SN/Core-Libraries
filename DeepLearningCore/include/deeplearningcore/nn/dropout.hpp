#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5f);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] float p() const noexcept { return p_; }

private:
    float p_;
};

} // namespace deeplearningcore

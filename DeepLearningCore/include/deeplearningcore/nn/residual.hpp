#pragma once

#include <deeplearningcore/nn/module.hpp>

namespace deeplearningcore {

class ResidualBlock : public Module {
public:
    explicit ResidualBlock(std::shared_ptr<Module> fn);

    Tensor forward(const Tensor& input) override;

private:
    std::shared_ptr<Module> fn_;
};

} // namespace deeplearningcore

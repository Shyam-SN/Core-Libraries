#include <deeplearningcore/nn/residual.hpp>

namespace deeplearningcore {

ResidualBlock::ResidualBlock(std::shared_ptr<Module> fn)
    : fn_(std::move(fn)) {
    register_module("fn", fn_);
}

Tensor ResidualBlock::forward(const Tensor& input) {
    Tensor fx = (*fn_)(input);
    return fx.add(input);
}

} // namespace deeplearningcore

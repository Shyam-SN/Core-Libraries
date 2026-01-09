#pragma once

#include <deeplearningcore/core/tensor.hpp>

namespace deeplearningcore {

class Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor forward(const Tensor& input, const Tensor& target) = 0;

    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }
};

class MSELoss : public Loss {
public:
    Tensor forward(const Tensor& input, const Tensor& target) override;
};

class CrossEntropyLoss : public Loss {
public:
    Tensor forward(const Tensor& input, const Tensor& target) override;
};

class BCELoss : public Loss {
public:
    Tensor forward(const Tensor& input, const Tensor& target) override;
};

class L1Loss : public Loss {
public:
    Tensor forward(const Tensor& input, const Tensor& target) override;
};

} // namespace deeplearningcore

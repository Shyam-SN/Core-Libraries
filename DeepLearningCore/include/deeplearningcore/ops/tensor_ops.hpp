#pragma once

#include <deeplearningcore/core/tensor.hpp>

namespace deeplearningcore {

namespace ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);
Tensor reshape(const Tensor& a, const Shape& new_shape);

Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor softmax(const Tensor& a, int dim = -1);
Tensor log_softmax(const Tensor& a, int dim = -1);

Tensor sum(const Tensor& a, int dim = -1, bool keepdim = false);
Tensor mean(const Tensor& a, int dim = -1, bool keepdim = false);

} // namespace ops

} // namespace deeplearningcore

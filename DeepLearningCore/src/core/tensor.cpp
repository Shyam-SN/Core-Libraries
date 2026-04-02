#include <deeplearningcore/core/tensor.hpp>
#include <deeplearningcore/ops/tensor_ops.hpp>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace deeplearningcore {

TensorImpl::TensorImpl()
    : shape({0}), storage(std::make_shared<Storage<float>>(0)), requires_grad(false) {}

TensorImpl::TensorImpl(Shape s, bool req_grad)
    : shape(std::move(s)), storage(std::make_shared<Storage<float>>(shape.numel())), requires_grad(req_grad) {}

TensorImpl::TensorImpl(Shape s, std::vector<float> data, bool req_grad)
    : shape(std::move(s)), storage(std::make_shared<Storage<float>>(data.data(), data.size())), requires_grad(req_grad) {
    if (data.size() != shape.numel()) {
        throw std::invalid_argument("Data size does not match shape numel");
    }
}

TensorImpl::TensorImpl(Shape s, float fill_value, bool req_grad)
    : shape(std::move(s)), storage(std::make_shared<Storage<float>>(shape.numel())), requires_grad(req_grad) {
    storage->fill(fill_value);
}

Tensor::Tensor()
    : impl_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
    : impl_(std::move(impl)) {}

Tensor::Tensor(Shape shape, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), requires_grad)) {}

Tensor::Tensor(Shape shape, std::vector<float> data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), std::move(data), requires_grad)) {}

Tensor::Tensor(Shape shape, float fill_value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), fill_value, requires_grad)) {}

Tensor Tensor::zeros(Shape shape, bool requires_grad) {
    return Tensor(std::move(shape), 0.0f, requires_grad);
}

Tensor Tensor::ones(Shape shape, bool requires_grad) {
    return Tensor(std::move(shape), 1.0f, requires_grad);
}

Tensor Tensor::randn(Shape shape, float mean, float stddev, bool requires_grad) {
    Tensor t(std::move(shape), requires_grad);
    static std::mt19937 gen(42);
    std::normal_distribution<float> dis(mean, stddev);
    for (size_t i = 0; i < t.numel(); ++i) {
        t[i] = dis(gen);
    }
    return t;
}

Tensor Tensor::rand(Shape shape, float min_val, float max_val, bool requires_grad) {
    Tensor t(std::move(shape), requires_grad);
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < t.numel(); ++i) {
        t[i] = dis(gen);
    }
    return t;
}

Tensor Tensor::scalar(float val, bool requires_grad) {
    return Tensor(Shape{1}, std::vector<float>{val}, requires_grad);
}

float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called on tensors with 1 element");
    }
    return (*impl_->storage)[0];
}

float Tensor::at(const std::vector<size_t>& indices) const {
    size_t flat_idx = impl_->shape.get_flat_index(indices);
    return (*impl_->storage)[flat_idx];
}

void Tensor::set_at(const std::vector<size_t>& indices, float val) {
    size_t flat_idx = impl_->shape.get_flat_index(indices);
    (*impl_->storage)[flat_idx] = val;
}

void Tensor::zero_grad() {
    if (impl_->grad) {
        impl_->grad->impl_->storage->fill(0.0f);
    }
}

void Tensor::backward(const Tensor& grad_tensor) {
    if (!impl_->requires_grad) return;

    if (!impl_->grad) {
        impl_->grad = std::make_shared<Tensor>(grad_tensor);
    } else {
        // Accumulate gradient
        for (size_t i = 0; i < numel(); ++i) {
            (*impl_->grad)[i] += grad_tensor[i];
        }
    }

    if (impl_->autograd_node) {
        AutogradNode::run_backward(impl_->autograd_node, grad_tensor);
    }
}

void Tensor::backward() {
    if (numel() != 1) {
        throw std::runtime_error("implicit backward() can only be called on 1-element scalar tensors");
    }
    backward(Tensor::ones(shape()));
}

Tensor Tensor::add(const Tensor& rhs) const {
    return ops::add(*this, rhs);
}

Tensor Tensor::sub(const Tensor& rhs) const {
    return ops::sub(*this, rhs);
}

Tensor Tensor::mul(const Tensor& rhs) const {
    return ops::mul(*this, rhs);
}

Tensor Tensor::div(const Tensor& rhs) const {
    return ops::div(*this, rhs);
}

Tensor Tensor::add(float scalar_val) const {
    return add(Tensor::scalar(scalar_val));
}

Tensor Tensor::sub(float scalar_val) const {
    return sub(Tensor::scalar(scalar_val));
}

Tensor Tensor::mul(float scalar_val) const {
    return mul(Tensor::scalar(scalar_val));
}

Tensor Tensor::div(float scalar_val) const {
    return div(Tensor::scalar(scalar_val));
}

Tensor Tensor::matmul(const Tensor& rhs) const {
    return ops::matmul(*this, rhs);
}

Tensor Tensor::transpose() const {
    return ops::transpose(*this);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    return ops::reshape(*this, new_shape);
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    return ops::sum(*this, dim, keepdim);
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    return ops::mean(*this, dim, keepdim);
}

Tensor Tensor::relu() const {
    return ops::relu(*this);
}

Tensor Tensor::sigmoid() const {
    return ops::sigmoid(*this);
}

Tensor Tensor::tanh_op() const {
    return ops::tanh(*this);
}

Tensor Tensor::gelu() const {
    return ops::gelu(*this);
}

Tensor Tensor::softmax(int dim) const {
    return ops::softmax(*this, dim);
}

Tensor Tensor::log_softmax(int dim) const {
    return ops::log_softmax(*this, dim);
}

Tensor Tensor::operator-() const {
    return mul(-1.0f);
}

void Tensor::print() const {
    std::cout << "Tensor shape=" << impl_->shape.to_string() << " data=[";
    size_t n = std::min<size_t>(numel(), 10);
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(4) << (*impl_->storage)[i] << (i + 1 < n ? ", " : "");
    }
    if (numel() > 10) std::cout << "...";
    std::cout << "]\n";
}

} // namespace deeplearningcore

#pragma once

#include <deeplearningcore/core/shape.hpp>
#include <deeplearningcore/core/storage.hpp>
#include <deeplearningcore/core/autograd.hpp>
#include <memory>
#include <vector>
#include <initializer_list>
#include <iostream>

namespace deeplearningcore {

struct TensorImpl {
    Shape shape;
    std::shared_ptr<Storage<float>> storage;
    bool requires_grad{false};
    std::shared_ptr<Tensor> grad{nullptr};
    std::shared_ptr<AutogradNode> autograd_node{nullptr};

    TensorImpl();
    explicit TensorImpl(Shape shape, bool requires_grad = false);
    TensorImpl(Shape shape, std::vector<float> data, bool requires_grad = false);
    TensorImpl(Shape shape, float fill_value, bool requires_grad = false);
};

class Tensor {
public:
    Tensor();
    explicit Tensor(std::shared_ptr<TensorImpl> impl);
    explicit Tensor(Shape shape, bool requires_grad = false);
    Tensor(Shape shape, std::vector<float> data, bool requires_grad = false);
    Tensor(Shape shape, float fill_value, bool requires_grad = false);

    // Factory methods
    static Tensor zeros(Shape shape, bool requires_grad = false);
    static Tensor ones(Shape shape, bool requires_grad = false);
    static Tensor randn(Shape shape, float mean = 0.0f, float stddev = 1.0f, bool requires_grad = false);
    static Tensor rand(Shape shape, float min_val = 0.0f, float max_val = 1.0f, bool requires_grad = false);
    static Tensor scalar(float val, bool requires_grad = false);

    // Properties
    [[nodiscard]] const Shape& shape() const noexcept { return impl_->shape; }
    [[nodiscard]] size_t ndim() const noexcept { return impl_->shape.ndim(); }
    [[nodiscard]] size_t numel() const noexcept { return impl_->shape.numel(); }
    [[nodiscard]] bool requires_grad() const noexcept { return impl_->requires_grad; }
    void set_requires_grad(bool req) { impl_->requires_grad = req; }

    [[nodiscard]] float* data() noexcept { return impl_->storage->data(); }
    [[nodiscard]] const float* data() const noexcept { return impl_->storage->data(); }

    [[nodiscard]] float item() const;
    [[nodiscard]] float at(const std::vector<size_t>& indices) const;
    void set_at(const std::vector<size_t>& indices, float val);

    [[nodiscard]] float& operator[](size_t idx) { return (*impl_->storage)[idx]; }
    [[nodiscard]] const float& operator[](size_t idx) const { return (*impl_->storage)[idx]; }

    // Gradient management
    [[nodiscard]] bool has_grad() const noexcept { return impl_->grad != nullptr; }
    [[nodiscard]] std::shared_ptr<Tensor> grad() const { return impl_->grad; }
    void set_grad(std::shared_ptr<Tensor> g) { impl_->grad = std::move(g); }
    void zero_grad();

    // Autograd node
    [[nodiscard]] std::shared_ptr<AutogradNode> autograd_node() const { return impl_->autograd_node; }
    void set_autograd_node(std::shared_ptr<AutogradNode> node) { impl_->autograd_node = std::move(node); }

    // Backpropagation
    void backward(const Tensor& grad_tensor);
    void backward();

    // Operations (with autograd support)
    Tensor add(const Tensor& rhs) const;
    Tensor sub(const Tensor& rhs) const;
    Tensor mul(const Tensor& rhs) const;
    Tensor div(const Tensor& rhs) const;

    Tensor add(float scalar) const;
    Tensor sub(float scalar) const;
    Tensor mul(float scalar) const;
    Tensor div(float scalar) const;

    Tensor matmul(const Tensor& rhs) const;
    Tensor transpose() const;
    Tensor reshape(const Shape& new_shape) const;

    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;

    // Activations
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh_op() const;
    Tensor gelu() const;
    Tensor softmax(int dim = -1) const;
    Tensor log_softmax(int dim = -1) const;

    // Operator overloads
    Tensor operator+(const Tensor& rhs) const { return add(rhs); }
    Tensor operator-(const Tensor& rhs) const { return sub(rhs); }
    Tensor operator*(const Tensor& rhs) const { return mul(rhs); }
    Tensor operator/(const Tensor& rhs) const { return div(rhs); }

    Tensor operator+(float scalar) const { return add(scalar); }
    Tensor operator-(float scalar) const { return sub(scalar); }
    Tensor operator*(float scalar) const { return mul(scalar); }
    Tensor operator/(float scalar) const { return div(scalar); }

    Tensor operator-() const;

    void print() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace deeplearningcore

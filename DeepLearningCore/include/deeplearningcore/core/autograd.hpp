#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

namespace deeplearningcore {

class Tensor;

class AutogradNode : public std::enable_shared_from_this<AutogradNode> {
public:
    using BackwardFn = std::function<void(const Tensor& grad)>;

    AutogradNode() = default;
    explicit AutogradNode(BackwardFn fn, std::vector<std::shared_ptr<AutogradNode>> parents = {})
        : backward_fn_(std::move(fn)), parents_(std::move(parents)) {}

    void set_backward_fn(BackwardFn fn) { backward_fn_ = std::move(fn); }
    void add_parent(std::shared_ptr<AutogradNode> parent) {
        if (parent) parents_.push_back(std::move(parent));
    }

    const std::vector<std::shared_ptr<AutogradNode>>& parents() const { return parents_; }
    void backward(const Tensor& grad);

    static void run_backward(std::shared_ptr<AutogradNode> root, const Tensor& root_grad);

private:
    BackwardFn backward_fn_;
    std::vector<std::shared_ptr<AutogradNode>> parents_;
};

class NoGradScope {
public:
    NoGradScope();
    ~NoGradScope();

    static bool is_grad_enabled();

private:
    bool prev_state_;
};

} // namespace deeplearningcore

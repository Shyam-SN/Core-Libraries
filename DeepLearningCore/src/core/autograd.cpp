#include <deeplearningcore/core/autograd.hpp>
#include <deeplearningcore/core/tensor.hpp>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace deeplearningcore {

static thread_local bool g_grad_enabled = true;

NoGradScope::NoGradScope() : prev_state_(g_grad_enabled) {
    g_grad_enabled = false;
}

NoGradScope::~NoGradScope() {
    g_grad_enabled = prev_state_;
}

bool NoGradScope::is_grad_enabled() {
    return g_grad_enabled;
}

void AutogradNode::backward(const Tensor& grad) {
    if (backward_fn_) {
        backward_fn_(grad);
    }
}

void AutogradNode::run_backward(std::shared_ptr<AutogradNode> root, const Tensor& root_grad) {
    if (!root) return;

    // Topological sort of computational graph DAG
    std::vector<std::shared_ptr<AutogradNode>> topo;
    std::unordered_set<AutogradNode*> visited;

    std::function<void(std::shared_ptr<AutogradNode>)> build_topo = [&](std::shared_ptr<AutogradNode> node) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        for (auto& parent : node->parents()) {
            build_topo(parent);
        }
        topo.push_back(node);
    };

    build_topo(root);
    std::reverse(topo.begin(), topo.end());

    // Execute backward pass in topological order
    root->backward(root_grad);
}

} // namespace deeplearningcore

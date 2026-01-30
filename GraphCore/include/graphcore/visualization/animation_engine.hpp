#pragma once

#include <vector>
#include <string>
#include <unordered_set>

namespace graphcore {

struct AnimationStep {
    size_t step_index{0};
    std::string description;
    std::unordered_set<size_t> highlighted_nodes;
    std::unordered_set<size_t> active_nodes;
    std::vector<std::pair<size_t, size_t>> highlighted_edges;
    std::vector<std::string> queue_or_stack_state;
};

class AnimationEngine {
public:
    void add_step(std::string desc,
                  std::unordered_set<size_t> high_nodes = {},
                  std::unordered_set<size_t> act_nodes = {},
                  std::vector<std::pair<size_t, size_t>> high_edges = {},
                  std::vector<std::string> struct_state = {}) {
        steps_.push_back({
            steps_.size(),
            std::move(desc),
            std::move(high_nodes),
            std::move(act_nodes),
            std::move(high_edges),
            std::move(struct_state)
        });
    }

    void clear() { steps_.clear(); current_step_ = 0; }
    [[nodiscard]] size_t total_steps() const noexcept { return steps_.size(); }
    [[nodiscard]] size_t current_step_index() const noexcept { return current_step_; }

    [[nodiscard]] const AnimationStep* get_current_step() const {
        if (steps_.empty()) return nullptr;
        return &steps_[current_step_];
    }

    bool step_next() {
        if (current_step_ + 1 < steps_.size()) {
            current_step_++;
            return true;
        }
        return false;
    }

    bool step_prev() {
        if (current_step_ > 0) {
            current_step_--;
            return true;
        }
        return false;
    }

    const std::vector<AnimationStep>& steps() const noexcept { return steps_; }

private:
    std::vector<AnimationStep> steps_;
    size_t current_step_{0};
};

} // namespace graphcore

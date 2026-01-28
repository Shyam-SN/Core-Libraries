#pragma once

#include <deeplearningcore/core/tensor.hpp>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace deeplearningcore {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }

    void train(bool mode = true) {
        is_training_ = mode;
        for (auto& [name, submod] : submodules_) {
            if (submod) submod->train(mode);
        }
    }

    void eval() {
        train(false);
    }

    [[nodiscard]] bool is_training() const noexcept { return is_training_; }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& [name, param] : parameters_) {
            if (param && param->requires_grad()) {
                params.push_back(param);
            }
        }
        for (auto& [name, submod] : submodules_) {
            if (submod) {
                auto sub_params = submod->parameters();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
        }
        return params;
    }

    void zero_grad() {
        for (auto& param : parameters()) {
            param->zero_grad();
        }
    }

protected:
    void register_parameter(const std::string& name, std::shared_ptr<Tensor> param) {
        parameters_[name] = param;
    }

    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_[name] = module;
    }

private:
    bool is_training_{true};
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace deeplearningcore

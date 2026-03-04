#pragma once

#include <deeplearningcore/optim/optimizer.hpp>
#include <vector>

namespace deeplearningcore {

class SGD : public Optimizer {
public:
    SGD(std::vector<std::shared_ptr<Tensor>> params, float lr, float momentum = 0.0f, float weight_decay = 0.0f);

    void step() override;

private:
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocities_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<Tensor>> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t t_{0};
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
};

class RMSProp : public Optimizer {
public:
    RMSProp(std::vector<std::shared_ptr<Tensor>> params, float lr = 0.01f, float alpha = 0.99f, float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

private:
    float alpha_;
    float eps_;
    float weight_decay_;
    std::vector<std::vector<float>> v_;
};

} // namespace deeplearningcore

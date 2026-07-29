#include <gtest/gtest.h>
#include <deeplearningcore/optim/optimizers.hpp>

using namespace deeplearningcore;

TEST(OptimizerTest, SGDSimpleConvergence) {
    // Minimize (w - 3.0)^2 => dw = 2*(w - 3)
    auto w = std::make_shared<Tensor>(Tensor::scalar(0.0f, true));
    SGD opt({w}, 0.1f);

    for (int step = 0; step < 50; ++step) {
        opt.zero_grad();
        Tensor diff = w->sub(3.0f);
        Tensor loss = diff.mul(diff);
        loss.backward();
        opt.step();
    }

    EXPECT_NEAR(w->item(), 3.0f, 1e-2f);
}

TEST(OptimizerTest, AdamSimpleConvergence) {
    auto w = std::make_shared<Tensor>(Tensor::scalar(-2.0f, true));
    Adam opt({w}, 0.1f);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        Tensor diff = w->sub(5.0f);
        Tensor loss = diff.mul(diff);
        loss.backward();
        opt.step();
    }

    EXPECT_NEAR(w->item(), 5.0f, 1e-2f);
}

TEST(OptimizerTest, RMSPropSimpleConvergence) {
    auto w = std::make_shared<Tensor>(Tensor::scalar(10.0f, true));
    RMSProp opt({w}, 0.1f);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        Tensor diff = w->sub(1.0f);
        Tensor loss = diff.mul(diff);
        loss.backward();
        opt.step();
    }

    EXPECT_NEAR(w->item(), 1.0f, 1e-1f);
}

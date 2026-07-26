#include <gtest/gtest.h>
#include <deeplearningcore/losses/losses.hpp>

using namespace deeplearningcore;

TEST(LossTest, MSELossEvaluation) {
    MSELoss loss_fn;
    Tensor input(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor target(Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});

    Tensor loss = loss_fn(input, target);
    // diff = [0, 1, 2, 3], sq_diff = [0, 1, 4, 9], sum = 14, mean = 3.5
    EXPECT_FLOAT_EQ(loss.item(), 3.5f);

    loss.backward();
    EXPECT_TRUE(input.has_grad());
}

TEST(LossTest, CrossEntropyLossEvaluation) {
    CrossEntropyLoss loss_fn;
    Tensor input(Shape{2, 3}, {1.0f, 2.0f, 0.0f, 0.5f, 2.5f, 0.0f}, true);
    Tensor target(Shape{2, 3}, {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    Tensor loss = loss_fn(input, target);
    EXPECT_GT(loss.item(), 0.0f);

    loss.backward();
    EXPECT_TRUE(input.has_grad());
}

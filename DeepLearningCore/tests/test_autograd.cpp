#include <gtest/gtest.h>
#include <deeplearningcore/core/tensor.hpp>

using namespace deeplearningcore;

TEST(AutogradTest, ScalarAdditionGradient) {
    Tensor a = Tensor::scalar(3.0f, true);
    Tensor b = Tensor::scalar(4.0f, true);

    Tensor c = a.add(b); // c = a + b = 7
    c.backward();

    EXPECT_FLOAT_EQ(a.grad()->item(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->item(), 1.0f);
}

TEST(AutogradTest, PolynomialGradient) {
    // y = 3*x^2 + 2*x + 1  => dy/dx = 6*x + 2
    // at x = 2: y = 3*4 + 4 + 1 = 17, dy/dx = 12 + 2 = 14
    Tensor x = Tensor::scalar(2.0f, true);
    Tensor x_sq = x.mul(x);
    Tensor term1 = x_sq.mul(3.0f);
    Tensor term2 = x.mul(2.0f);
    Tensor y = term1.add(term2).add(1.0f);

    y.backward();

    EXPECT_FLOAT_EQ(y.item(), 17.0f);
    EXPECT_FLOAT_EQ(x.grad()->item(), 14.0f);
}

TEST(AutogradTest, MatrixMultiplicationGradient) {
    Tensor W(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor X(Shape{2, 2}, {0.5f, 1.0f, 1.5f, 2.0f}, true);

    Tensor Y = X.matmul(W);
    Tensor loss = Y.sum();
    loss.backward();

    EXPECT_TRUE(W.has_grad());
    EXPECT_TRUE(X.has_grad());
    EXPECT_FLOAT_EQ(W.grad()->at({0, 0}), 2.0f); // sum of column 0 of X: 0.5 + 1.5 = 2.0
    EXPECT_FLOAT_EQ(W.grad()->at({1, 0}), 3.0f); // sum of column 1 of X: 1.0 + 2.0 = 3.0
}

#include <gtest/gtest.h>
#include <deeplearningcore/nn/linear.hpp>
#include <deeplearningcore/nn/conv2d.hpp>
#include <deeplearningcore/nn/batch_norm.hpp>
#include <deeplearningcore/nn/dropout.hpp>

using namespace deeplearningcore;

TEST(LayerTest, LinearLayer) {
    Linear fc(4, 2);
    Tensor x = Tensor::ones(Shape{3, 4});
    Tensor y = fc(x);

    EXPECT_EQ(y.shape(), (Shape{3, 2}));
    EXPECT_EQ(fc.parameters().size(), 2);
}

TEST(LayerTest, Conv2DLayer) {
    Conv2D conv(3, 16, 3, 1, 1); // in=3, out=16, k=3, s=1, p=1
    Tensor x = Tensor::ones(Shape{2, 3, 32, 32});
    Tensor y = conv(x);

    EXPECT_EQ(y.shape(), (Shape{2, 16, 32, 32}));
}

TEST(LayerTest, BatchNorm1D) {
    BatchNorm1D bn(4);
    Tensor x = Tensor::rand(Shape{8, 4}, -1.0f, 1.0f);
    Tensor y = bn(x);

    EXPECT_EQ(y.shape(), (Shape{8, 4}));
}

TEST(LayerTest, DropoutLayer) {
    Dropout drop(0.5f);
    Tensor x = Tensor::ones(Shape{10, 10});

    drop.train(true);
    Tensor y_train = drop(x);
    EXPECT_EQ(y_train.shape(), (Shape{10, 10}));

    drop.eval();
    Tensor y_eval = drop(x);
    EXPECT_FLOAT_EQ(y_eval.at({0, 0}), 1.0f);
}

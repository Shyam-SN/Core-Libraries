#include <gtest/gtest.h>
#include <deeplearningcore/core/tensor.hpp>

using namespace deeplearningcore;

TEST(TensorTest, ShapeAndStrides) {
    Shape shape{2, 3, 4};
    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape.numel(), 24);
    EXPECT_EQ(shape.stride(0), 12);
    EXPECT_EQ(shape.stride(1), 4);
    EXPECT_EQ(shape.stride(2), 1);
}

TEST(TensorTest, CreationAndIndexing) {
    Tensor t = Tensor::ones(Shape{2, 2});
    EXPECT_EQ(t.numel(), 4);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 1.0f);

    t.set_at({0, 1}, 5.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 5.0f);
}

TEST(TensorTest, ArithmeticOperations) {
    Tensor a = Tensor::ones(Shape{2, 2});
    Tensor b = Tensor::ones(Shape{2, 2}).mul(2.0f);

    Tensor c = a.add(b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 3.0f);

    Tensor d = b.sub(a);
    EXPECT_FLOAT_EQ(d.at({0, 0}), 1.0f);

    Tensor e = a.mul(b);
    EXPECT_FLOAT_EQ(e.at({0, 0}), 2.0f);

    Tensor f = b.div(2.0f);
    EXPECT_FLOAT_EQ(f.at({0, 0}), 1.0f);
}

TEST(TensorTest, MatrixMultiplication) {
    Tensor a(Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b(Shape{3, 2}, {1, 2, 3, 4, 5, 6});

    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), (Shape{2, 2}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 22.0f); // 1*1 + 2*3 + 3*5 = 1+6+15 = 22
    EXPECT_FLOAT_EQ(c.at({0, 1}), 28.0f); // 1*2 + 2*4 + 3*6 = 2+8+18 = 28
    EXPECT_FLOAT_EQ(c.at({1, 0}), 49.0f); // 4*1 + 5*3 + 6*5 = 4+15+30 = 49
    EXPECT_FLOAT_EQ(c.at({1, 1}), 64.0f); // 4*2 + 5*4 + 6*6 = 8+20+36 = 64
}

TEST(TensorTest, TransposeAndReshape) {
    Tensor a(Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor at = a.transpose();

    EXPECT_EQ(at.shape(), (Shape{3, 2}));
    EXPECT_FLOAT_EQ(at.at({0, 1}), 4.0f);

    Tensor ar = a.reshape(Shape{6, 1});
    EXPECT_EQ(ar.shape(), (Shape{6, 1}));
    EXPECT_FLOAT_EQ(ar.at({5, 0}), 6.0f);
}

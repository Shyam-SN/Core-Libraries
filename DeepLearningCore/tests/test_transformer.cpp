#include <gtest/gtest.h>
#include <deeplearningcore/nn/transformer.hpp>

using namespace deeplearningcore;

TEST(TransformerTest, LayerNormLayer) {
    LayerNorm ln(16);
    Tensor x = Tensor::rand(Shape{4, 16}, -2.0f, 2.0f, true);
    Tensor y = ln(x);

    EXPECT_EQ(y.shape(), (Shape{4, 16}));
}

TEST(TransformerTest, PositionalEncodingLayer) {
    PositionalEncoding pe(32, 100);
    Tensor x = Tensor::zeros(Shape{10, 32});
    Tensor y = pe(x);

    EXPECT_EQ(y.shape(), (Shape{10, 32}));
}

TEST(TransformerTest, MultiHeadAttentionLayer) {
    MultiHeadAttention mha(32, 4); // d_model=32, num_heads=4
    Tensor x = Tensor::rand(Shape{10, 32}, -1.0f, 1.0f, true);
    Tensor out = mha(x);

    EXPECT_EQ(out.shape(), (Shape{10, 32}));
}

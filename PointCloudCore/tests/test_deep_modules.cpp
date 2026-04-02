#include <gtest/gtest.h>
#include <pointcloudcore/nn/pointnet_plusplus.hpp>
#include <pointcloudcore/nn/point_transformer.hpp>
#include <pointcloudcore/nn/pointnext.hpp>

using namespace pointcloudcore;

TEST(DeepModulesTest, PointNetPlusPlusSAForward) {
    PointCloud cloud;
    for (size_t i = 0; i < 50; ++i) {
        cloud.points.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    PointNetSetAbstraction sa(16, 0.2, 8, 3, 64);
    Tensor feats({1, 50, 3}, 1.0);
    Tensor out_feats = sa.forward(cloud, feats);

    EXPECT_EQ(out_feats.shape[0], 1);
    EXPECT_EQ(out_feats.shape[1], 16);
    EXPECT_EQ(out_feats.shape[2], 64);
}

TEST(DeepModulesTest, PointTransformerBlockForward) {
    PointCloud cloud;
    for (size_t i = 0; i < 20; ++i) {
        cloud.points.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    PointTransformerBlock pt_block(32, 64, 8);
    Tensor feats({1, 20, 32}, 1.0);
    Tensor out_feats = pt_block.forward(cloud, feats);

    EXPECT_EQ(out_feats.shape[0], 1);
    EXPECT_EQ(out_feats.shape[1], 20);
    EXPECT_EQ(out_feats.shape[2], 64);
}

TEST(DeepModulesTest, PointNeXtBlockForward) {
    PointCloud cloud;
    for (size_t i = 0; i < 20; ++i) {
        cloud.points.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    PointNeXtBlock pnext_block(32, 64, 4);
    Tensor feats({1, 20, 32}, 1.0);
    Tensor out_feats = pnext_block.forward(cloud, feats);

    EXPECT_EQ(out_feats.shape[0], 1);
    EXPECT_EQ(out_feats.shape[1], 20);
    EXPECT_EQ(out_feats.shape[2], 64);
}

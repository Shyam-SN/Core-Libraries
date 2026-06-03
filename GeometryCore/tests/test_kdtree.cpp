#include <gtest/gtest.h>
#include <geometrycore/pointcloud/kdtree.hpp>

using namespace geometrycore;

TEST(KDTreeTest, NearestNeighborSearch) {
    std::vector<Vector3> pts = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {5.0, 5.0, 5.0}
    };

    KDTree kdtree(pts);

    double dist_sq = 0.0;
    size_t idx = kdtree.nearest_neighbor({1.1, 1.0, 0.9}, dist_sq);

    EXPECT_EQ(idx, 1);
    EXPECT_NEAR(dist_sq, 0.02, 1e-6);
}

TEST(KDTreeTest, RadiusSearch) {
    std::vector<Vector3> pts = {
        {0.0, 0.0, 0.0},
        {0.1, 0.0, 0.0},
        {0.2, 0.0, 0.0},
        {5.0, 5.0, 5.0}
    };

    KDTree kdtree(pts);
    auto indices = kdtree.radius_search({0.0, 0.0, 0.0}, 0.3);

    EXPECT_EQ(indices.size(), 3);
}

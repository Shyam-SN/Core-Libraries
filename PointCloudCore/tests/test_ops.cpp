#include <gtest/gtest.h>
#include <pointcloudcore/ops/pca_normals.hpp>
#include <pointcloudcore/ops/sampling.hpp>
#include <pointcloudcore/ops/neighborhood.hpp>

using namespace pointcloudcore;

TEST(OpsTest, PCANormalsEstimation) {
    PointCloud cloud;
    // Planar point cloud along z = 1.0
    for (double x = -1.0; x <= 1.0; x += 0.2) {
        for (double y = -1.0; y <= 1.0; y += 0.2) {
            cloud.points.push_back({x, y, 1.0});
        }
    }

    PCANormals::estimate_normals(cloud, 10);
    EXPECT_EQ(cloud.normals.size(), cloud.points.size());

    // Normals on planar z=1 should point along z axis
    for (const auto& n : cloud.normals) {
        EXPECT_NEAR(n.norm(), 1.0, 1e-4);
        EXPECT_NEAR(std::abs(n.z), 1.0, 1e-1);
    }
}

TEST(OpsTest, FarthestPointSampling) {
    PointCloud cloud;
    for (size_t i = 0; i < 100; ++i) {
        cloud.points.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    PointCloud sampled = Sampling::farthest_point_sampling(cloud, 10);
    EXPECT_EQ(sampled.size(), 10);
}

TEST(OpsTest, KNNAndRadiusSearch) {
    PointCloud cloud;
    cloud.points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};

    auto knn = Neighborhood::knn_search(cloud, cloud, 2);
    EXPECT_EQ(knn[0].size(), 2);
    EXPECT_EQ(knn[0][0], 0);
    EXPECT_EQ(knn[0][1], 1);

    auto rad = Neighborhood::radius_search(cloud, cloud, 2.5);
    EXPECT_EQ(rad[0].size(), 3);
}

#include <gtest/gtest.h>
#include <pointcloudcore/ops/fpfh_descriptor.hpp>
#include <pointcloudcore/ops/pca_normals.hpp>

using namespace pointcloudcore;

TEST(DescriptorTest, FPFHComputation) {
    PointCloud cloud;
    for (double x = -0.5; x <= 0.5; x += 0.2) {
        for (double y = -0.5; y <= 0.5; y += 0.2) {
            cloud.points.push_back({x, y, 0.5});
        }
    }

    PCANormals::estimate_normals(cloud, 10);
    auto fpfh = FPFHDescriptor::compute_fpfh(cloud, 0.5);

    EXPECT_EQ(fpfh.size(), cloud.size());
    EXPECT_EQ(fpfh[0].size(), 33);
}

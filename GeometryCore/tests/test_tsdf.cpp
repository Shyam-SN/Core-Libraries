#include <gtest/gtest.h>
#include <geometrycore/spatial/tsdf_volume.hpp>

using namespace geometrycore;

TEST(TSDFTest, IntegrationAndIndexing) {
    TSDFVolume volume(Vector3(-1.0, -1.0, -1.0), Vector3(2.0, 2.0, 2.0), 0.1, 0.2);

    EXPECT_GT(volume.dim_x(), 10);
    EXPECT_GT(volume.dim_y(), 10);
    EXPECT_GT(volume.dim_z(), 10);

    volume.set_voxel(0, 0, 0, -0.5f, 1.0f);
    TSDFVoxel voxel = volume.get_voxel(0, 0, 0);

    EXPECT_FLOAT_EQ(voxel.tsdf, -0.5f);
    EXPECT_FLOAT_EQ(voxel.weight, 1.0f);
}

#include <gtest/gtest.h>
#include <geometrycore/spatial/tsdf_volume.hpp>
#include <geometrycore/mesh/marching_cubes.hpp>

using namespace geometrycore;

TEST(MarchingCubesTest, IsoSurfaceMeshExtraction) {
    TSDFVolume volume(Vector3(-1.0, -1.0, -1.0), Vector3(2.0, 2.0, 2.0), 0.2, 0.4);

    // Create a sphere distance field in TSDF volume
    for (size_t iz = 0; iz < volume.dim_z(); ++iz) {
        for (size_t iy = 0; iy < volume.dim_y(); ++iy) {
            for (size_t ix = 0; ix < volume.dim_x(); ++ix) {
                Vector3 p = volume.voxel_to_world(ix, iy, iz);
                double dist = p.norm() - 0.5; // Sphere of radius 0.5
                float tsdf = static_cast<float>(std::max(-1.0, std::min(1.0, dist / 0.4)));
                volume.set_voxel(ix, iy, iz, tsdf, 1.0f);
            }
        }
    }

    Mesh mesh = MarchingCubes::extract_mesh(volume, 0.0f);
    EXPECT_GT(mesh.num_vertices(), 0);
}

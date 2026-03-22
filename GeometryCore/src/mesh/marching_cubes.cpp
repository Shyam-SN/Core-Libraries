#include <geometrycore/mesh/marching_cubes.hpp>
#include <array>

namespace geometrycore {

// Marching Cubes edge table and triangle table
static const int edgeTable[256] = {
0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

static Vector3 vertex_interp(float iso_level, const Vector3& p1, const Vector3& p2, float val1, float val2) {
    if (std::abs(iso_level - val1) < 1e-5f) return p1;
    if (std::abs(iso_level - val2) < 1e-5f) return p2;
    if (std::abs(val1 - val2) < 1e-5f) return p1;

    double mu = (iso_level - val1) / (val2 - val1);
    return p1 + (p2 - p1) * mu;
}

Mesh MarchingCubes::extract_mesh(const TSDFVolume& volume, float iso_level) {
    Mesh mesh;
    size_t nx = volume.dim_x();
    size_t ny = volume.dim_y();
    size_t nz = volume.dim_z();

    if (nx < 2 || ny < 2 || nz < 2) return mesh;

    for (size_t iz = 0; iz < nz - 1; ++iz) {
        for (size_t iy = 0; iy < ny - 1; ++iy) {
            for (size_t ix = 0; ix < nx - 1; ++ix) {
                // Get 8 corners of voxel cube
                std::array<Vector3, 8> corner_pos{
                    volume.voxel_to_world(ix, iy, iz),
                    volume.voxel_to_world(ix + 1, iy, iz),
                    volume.voxel_to_world(ix + 1, iy + 1, iz),
                    volume.voxel_to_world(ix, iy + 1, iz),
                    volume.voxel_to_world(ix, iy, iz + 1),
                    volume.voxel_to_world(ix + 1, iy, iz + 1),
                    volume.voxel_to_world(ix + 1, iy + 1, iz + 1),
                    volume.voxel_to_world(ix, iy + 1, iz + 1)
                };

                std::array<TSDFVoxel, 8> corner_voxels{
                    volume.get_voxel(ix, iy, iz),
                    volume.get_voxel(ix + 1, iy, iz),
                    volume.get_voxel(ix + 1, iy + 1, iz),
                    volume.get_voxel(ix, iy + 1, iz),
                    volume.get_voxel(ix, iy, iz + 1),
                    volume.get_voxel(ix + 1, iy, iz + 1),
                    volume.get_voxel(ix + 1, iy + 1, iz + 1),
                    volume.get_voxel(ix, iy + 1, iz + 1)
                };

                // Determine cube index
                int cube_idx = 0;
                for (int i = 0; i < 8; ++i) {
                    if (corner_voxels[i].weight <= 0.0f) continue;
                    if (corner_voxels[i].tsdf < iso_level) cube_idx |= (1 << i);
                }

                int edges = edgeTable[cube_idx];
                if (edges == 0) continue;

                // Interpolate 12 edge vertices
                std::array<Vector3, 12> vert_list;
                if (edges & 1) vert_list[0] = vertex_interp(iso_level, corner_pos[0], corner_pos[1], corner_voxels[0].tsdf, corner_voxels[1].tsdf);
                if (edges & 2) vert_list[1] = vertex_interp(iso_level, corner_pos[1], corner_pos[2], corner_voxels[1].tsdf, corner_voxels[2].tsdf);
                if (edges & 4) vert_list[2] = vertex_interp(iso_level, corner_pos[2], corner_pos[3], corner_voxels[2].tsdf, corner_voxels[3].tsdf);
                if (edges & 8) vert_list[3] = vertex_interp(iso_level, corner_pos[3], corner_pos[0], corner_voxels[3].tsdf, corner_voxels[0].tsdf);
                if (edges & 16) vert_list[4] = vertex_interp(iso_level, corner_pos[4], corner_pos[5], corner_voxels[4].tsdf, corner_voxels[5].tsdf);
                if (edges & 32) vert_list[5] = vertex_interp(iso_level, corner_pos[5], corner_pos[6], corner_voxels[5].tsdf, corner_voxels[6].tsdf);
                if (edges & 64) vert_list[6] = vertex_interp(iso_level, corner_pos[6], corner_pos[7], corner_voxels[6].tsdf, corner_voxels[7].tsdf);
                if (edges & 128) vert_list[7] = vertex_interp(iso_level, corner_pos[7], corner_pos[4], corner_voxels[7].tsdf, corner_voxels[4].tsdf);
                if (edges & 256) vert_list[8] = vertex_interp(iso_level, corner_pos[0], corner_pos[4], corner_voxels[0].tsdf, corner_voxels[4].tsdf);
                if (edges & 512) vert_list[9] = vertex_interp(iso_level, corner_pos[1], corner_pos[5], corner_voxels[1].tsdf, corner_voxels[5].tsdf);
                if (edges & 1024) vert_list[10] = vertex_interp(iso_level, corner_pos[2], corner_pos[6], corner_voxels[2].tsdf, corner_voxels[6].tsdf);
                if (edges & 2048) vert_list[11] = vertex_interp(iso_level, corner_pos[3], corner_pos[7], corner_voxels[3].tsdf, corner_voxels[7].tsdf);

                // Add vertices to mesh
                for (int e = 0; e < 12; ++e) {
                    if (edges & (1 << e)) {
                        mesh.vertices.push_back(vert_list[e]);
                    }
                }
            }
        }
    }

    mesh.compute_vertex_normals();
    return mesh;
}

} // namespace geometrycore

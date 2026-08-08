#include <geometrycore/geometrycore.hpp>
#include <iostream>
#include <iomanip>

using namespace geometrycore;

int main() {
    std::cout << "=========================================================\n";
    std::cout << " GeometryCore: 3D Vision & Geometry Processing Demo       \n";
    std::cout << "=========================================================\n\n";

    Timer timer;

    // 1. Point Cloud & ICP Registration Demo
    std::cout << "=== 1. Point Cloud & ICP Rigid Registration ===\n";
    PointCloud target;
    for (double x = -0.5; x <= 0.5; x += 0.05) {
        for (double y = -0.5; y <= 0.5; y += 0.05) {
            target.points.push_back({x, y, 1.0 + x * x + y * y});
        }
    }
    std::cout << "[INFO] Target Point Cloud created with " << target.size() << " points.\n";

    Matrix3 R_gt = Matrix3::rot_z(0.15) * Matrix3::rot_x(0.1);
    Vector3 t_gt(0.1, -0.05, 0.08);

    PointCloud source = target;
    source.transform(R_gt.inverse(), -(R_gt.inverse() * t_gt));
    std::cout << "[INFO] Source Point Cloud created with perturbation.\n";

    Timer icp_timer;
    ICPResult icp_res = ICP::align(source, target, 50, 0.5, 1e-6);
    std::cout << "[BENCHMARK] ICP Registration completed in " << icp_timer.elapsed_ms() << " ms.\n";
    std::cout << "[INFO] ICP Converged: " << (icp_res.converged ? "YES" : "NO") << "\n";
    std::cout << "[INFO] Iterations: " << icp_res.iterations << " - Inlier RMSE: " << icp_res.inlier_rmse << "\n";
    std::cout << "[INFO] Estimated Translation t: [" << icp_res.t.x << ", " << icp_res.t.y << ", " << icp_res.t.z << "]\n\n";

    // 2. TSDF Volume Integration & Marching Cubes Demo
    std::cout << "=== 2. TSDF Volume Integration & Marching Cubes Mesh Generation ===\n";
    TSDFVolume volume(Vector3(-1.0, -1.0, -1.0), Vector3(2.0, 2.0, 2.0), 0.08, 0.2);

    for (size_t iz = 0; iz < volume.dim_z(); ++iz) {
        for (size_t iy = 0; iy < volume.dim_y(); ++iy) {
            for (size_t ix = 0; ix < volume.dim_x(); ++ix) {
                Vector3 p = volume.voxel_to_world(ix, iy, iz);
                double dist = p.norm() - 0.6; // Sphere of radius 0.6
                float tsdf = static_cast<float>(std::max(-1.0, std::min(1.0, dist / 0.2)));
                volume.set_voxel(ix, iy, iz, tsdf, 1.0f);
            }
        }
    }

    Timer mc_timer;
    Mesh mesh = MarchingCubes::extract_mesh(volume, 0.0f);
    std::cout << "[BENCHMARK] Marching Cubes Mesh Extraction completed in " << mc_timer.elapsed_ms() << " ms.\n";
    std::cout << "[INFO] Generated Mesh Vertices: " << mesh.num_vertices() << "\n";

    std::cout << "\nDemo executed successfully in " << timer.elapsed_ms() << " ms.\n";
    std::cout << "=========================================================\n";

    return 0;
}

#include <pointcloudcore/pointcloudcore.hpp>
#include <iostream>
#include <iomanip>

using namespace pointcloudcore;

int main() {
    std::cout << "=========================================================\n";
    std::cout << " PointCloudCore: 3D Perception & Geometric Learning Demo  \n";
    std::cout << "=========================================================\n\n";

    Timer timer;

    // 1. Point Cloud Operations & Normal Estimation
    std::cout << "=== 1. Classical Point Cloud Perception & Descriptors ===\n";
    PointCloud target;
    for (double x = -0.5; x <= 0.5; x += 0.05) {
        for (double y = -0.5; y <= 0.5; y += 0.05) {
            target.points.push_back({x, y, 0.5 + 0.1 * (x * x + y * y)});
        }
    }
    std::cout << "[INFO] Target Point Cloud created with " << target.size() << " points.\n";

    Timer pca_timer;
    PCANormals::estimate_normals(target, 15);
    std::cout << "[BENCHMARK] PCA Surface Normals estimated in " << pca_timer.elapsed_ms() << " ms.\n";

    Timer fpfh_timer;
    auto fpfh_tgt = FPFHDescriptor::compute_fpfh(target, 0.3);
    std::cout << "[BENCHMARK] 33D FPFH Descriptors computed in " << fpfh_timer.elapsed_ms() << " ms.\n";

    // 2. Farthest Point Sampling & Subsampling
    std::cout << "\n=== 2. Farthest Point Sampling (FPS) ===\n";
    Timer fps_timer;
    PointCloud sampled = Sampling::farthest_point_sampling(target, 64);
    std::cout << "[BENCHMARK] FPS sampled " << sampled.size() << " points in " << fps_timer.elapsed_ms() << " ms.\n";

    // 3. PointNet++ Set Abstraction & Point Transformer Modules
    std::cout << "\n=== 3. 3D Geometric Deep Learning Modules ===\n";
    PointNetSetAbstraction sa(32, 0.2, 8, 3, 64);
    Tensor input_feats({1, target.size(), 3}, 1.0);

    Timer sa_timer;
    Tensor sa_out = sa.forward(target, input_feats);
    std::cout << "[BENCHMARK] PointNet++ Set Abstraction forward pass completed in " << sa_timer.elapsed_ms() << " ms.\n";
    std::cout << "[INFO] Output Tensor Shape: [" << sa_out.shape[0] << ", " << sa_out.shape[1] << ", " << sa_out.shape[2] << "]\n";

    PointTransformerBlock pt_block(64, 128, 8);
    Timer pt_timer;
    Tensor pt_out = pt_block.forward(sampled, sa_out);
    std::cout << "[BENCHMARK] Point Transformer Vector Attention block completed in " << pt_timer.elapsed_ms() << " ms.\n";
    std::cout << "[INFO] Output Tensor Shape: [" << pt_out.shape[0] << ", " << pt_out.shape[1] << ", " << pt_out.shape[2] << "]\n";

    // 4. Chamfer Distance Evaluation
    std::cout << "\n=== 4. 3D Shape Distance Evaluation ===\n";
    double cd = Metrics::chamfer_distance(target, sampled);
    std::cout << "[INFO] Chamfer Distance CD(Target, Sampled): " << cd << "\n";

    std::cout << "\nDemo executed successfully in " << timer.elapsed_ms() << " ms.\n";
    std::cout << "=========================================================\n";

    return 0;
}

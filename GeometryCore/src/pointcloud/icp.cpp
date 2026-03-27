#include <geometrycore/pointcloud/icp.hpp>
#include <geometrycore/pointcloud/kdtree.hpp>
#include <iostream>

namespace geometrycore {

ICPResult ICP::align(const PointCloud& source, const PointCloud& target,
                     size_t max_iterations, double max_correspondence_distance,
                     double tolerance) {
    ICPResult res;
    if (source.empty() || target.empty()) return res;

    KDTree target_kdtree(target.points);

    PointCloud src_transformed = source;
    Matrix3 R_total = Matrix3::identity();
    Vector3 t_total = Vector3::zero();

    double prev_rmse = 1e9;
    double max_dist_sq = max_correspondence_distance * max_correspondence_distance;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // 1. Find correspondences
        std::vector<Vector3> src_corr;
        std::vector<Vector3> tgt_corr;
        double sum_sq_err = 0.0;

        for (const auto& p : src_transformed.points) {
            double dist_sq = 0.0;
            size_t tgt_idx = target_kdtree.nearest_neighbor(p, dist_sq);
            if (dist_sq <= max_dist_sq) {
                src_corr.push_back(p);
                tgt_corr.push_back(target.points[tgt_idx]);
                sum_sq_err += dist_sq;
            }
        }

        size_t n_corr = src_corr.size();
        if (n_corr < 3) {
            res.converged = false;
            break;
        }

        double rmse = std::sqrt(sum_sq_err / static_cast<double>(n_corr));

        // 2. Compute centroids
        Vector3 mu_src = Vector3::zero();
        Vector3 mu_tgt = Vector3::zero();
        for (size_t i = 0; i < n_corr; ++i) {
            mu_src += src_corr[i];
            mu_tgt += tgt_corr[i];
        }
        mu_src /= static_cast<double>(n_corr);
        mu_tgt /= static_cast<double>(n_corr);

        // 3. Compute cross-covariance matrix H
        Matrix3 H = Matrix3::zero();
        for (size_t i = 0; i < n_corr; ++i) {
            Vector3 dp = src_corr[i] - mu_src;
            Vector3 dq = tgt_corr[i] - mu_tgt;

            H.set(0,0, H.at(0,0) + dp.x * dq.x); H.set(0,1, H.at(0,1) + dp.x * dq.y); H.set(0,2, H.at(0,2) + dp.x * dq.z);
            H.set(1,0, H.at(1,0) + dp.y * dq.x); H.set(1,1, H.at(1,1) + dp.y * dq.y); H.set(1,2, H.at(1,2) + dp.y * dq.z);
            H.set(2,0, H.at(2,0) + dp.z * dq.x); H.set(2,1, H.at(2,1) + dp.z * dq.y); H.set(2,2, H.at(2,2) + dp.z * dq.z);
        }

        // 4. Kabsch-Umeyama SVD H = U * S * V^T => R_step = V * U^T
        Matrix3 U, V;
        Vector3 S;
        H.svd(U, S, V);

        Matrix3 R_step = V * U.transpose();
        if (R_step.determinant() < 0.0) {
            Matrix3 V_fix = V;
            V_fix.set(0, 2, -V.at(0, 2));
            V_fix.set(1, 2, -V.at(1, 2));
            V_fix.set(2, 2, -V.at(2, 2));
            R_step = V_fix * U.transpose();
        }

        Vector3 t_step = mu_tgt - R_step * mu_src;

        // 5. Update transformed source cloud & total transformation
        src_transformed.transform(R_step, t_step);
        R_total = R_step * R_total;
        t_total = R_step * t_total + t_step;

        res.iterations = iter + 1;
        res.inlier_rmse = rmse;
        res.fitness = static_cast<double>(n_corr) / static_cast<double>(source.size());

        if (std::abs(prev_rmse - rmse) < tolerance) {
            res.converged = true;
            break;
        }
        prev_rmse = rmse;
    }

    res.R = R_total;
    res.t = t_total;
    res.T = Matrix4::transformation(R_total, t_total);

    return res;
}

} // namespace geometrycore

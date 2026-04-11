#include <geometrycore/spatial/tsdf_volume.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace geometrycore {

TSDFVolume::TSDFVolume(Vector3 origin, Vector3 size, double voxel_size, double trunc_dist)
    : origin_(origin), size_(size), voxel_size_(voxel_size), trunc_dist_(trunc_dist) {

    dim_x_ = static_cast<size_t>(std::ceil(size_.x / voxel_size_));
    dim_y_ = static_cast<size_t>(std::ceil(size_.y / voxel_size_));
    dim_z_ = static_cast<size_t>(std::ceil(size_.z / voxel_size_));

    voxels_.resize(dim_x_ * dim_y_ * dim_z_);
}

TSDFVoxel TSDFVolume::get_voxel(size_t ix, size_t iy, size_t iz) const {
    if (ix >= dim_x_ || iy >= dim_y_ || iz >= dim_z_) {
        throw std::out_of_range("TSDF voxel index out of bounds");
    }
    return voxels_[(iz * dim_y_ + iy) * dim_x_ + ix];
}

void TSDFVolume::set_voxel(size_t ix, size_t iy, size_t iz, float tsdf, float weight) {
    if (ix >= dim_x_ || iy >= dim_y_ || iz >= dim_z_) {
        throw std::out_of_range("TSDF voxel index out of bounds");
    }
    size_t idx = (iz * dim_y_ + iy) * dim_x_ + ix;
    voxels_[idx].tsdf = tsdf;
    voxels_[idx].weight = weight;
}

Vector3 TSDFVolume::voxel_to_world(size_t ix, size_t iy, size_t iz) const {
    return {
        origin_.x + (static_cast<double>(ix) + 0.5) * voxel_size_,
        origin_.y + (static_cast<double>(iy) + 0.5) * voxel_size_,
        origin_.z + (static_cast<double>(iz) + 0.5) * voxel_size_
    };
}

void TSDFVolume::integrate_depth(const std::vector<float>& depth_map, const CameraModel& cam, const Matrix4& T_cw) {
    // Transform all voxel world coordinates into camera frame and project into depth map
    for (size_t iz = 0; iz < dim_z_; ++iz) {
        for (size_t iy = 0; iy < dim_y_; ++iy) {
            for (size_t ix = 0; ix < dim_x_; ++ix) {
                Vector3 p_world = voxel_to_world(ix, iy, iz);
                Vector3 p_cam = T_cw.transform_point(p_world);

                if (p_cam.z <= 0.0) continue;

                auto [u, v] = cam.project_pixel(p_cam);
                int iu = static_cast<int>(std::round(u));
                int iv = static_cast<int>(std::round(v));

                if (iu >= 0 && iu < static_cast<int>(cam.width) && iv >= 0 && iv < static_cast<int>(cam.height)) {
                    float depth_val = depth_map[iv * cam.width + iu];
                    if (depth_val <= 0.0f) continue;

                    double sdf = static_cast<double>(depth_val) - p_cam.z;
                    if (sdf >= -trunc_dist_) {
                        double tsdf = std::min(1.0, sdf / trunc_dist_);
                        size_t idx = (iz * dim_y_ + iy) * dim_x_ + ix;

                        float old_tsdf = voxels_[idx].tsdf;
                        float old_w = voxels_[idx].weight;
                        float new_w = old_w + 1.0f;

                        voxels_[idx].tsdf = static_cast<float>((old_tsdf * old_w + tsdf) / new_w);
                        voxels_[idx].weight = new_w;
                    }
                }
            }
        }
    }
}

} // namespace geometrycore

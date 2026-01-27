#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <geometrycore/camera/camera_model.hpp>
#include <vector>

namespace geometrycore {

struct TSDFVoxel {
    float tsdf{1.0f};
    float weight{0.0f};
    float color[3]{0.0f, 0.0f, 0.0f};
};

class TSDFVolume {
public:
    TSDFVolume(Vector3 origin, Vector3 size, double voxel_size = 0.02, double trunc_dist = 0.08);

    [[nodiscard]] size_t dim_x() const noexcept { return dim_x_; }
    [[nodiscard]] size_t dim_y() const noexcept { return dim_y_; }
    [[nodiscard]] size_t dim_z() const noexcept { return dim_z_; }
    [[nodiscard]] double voxel_size() const noexcept { return voxel_size_; }
    [[nodiscard]] double trunc_dist() const noexcept { return trunc_dist_; }
    [[nodiscard]] Vector3 origin() const noexcept { return origin_; }

    [[nodiscard]] const std::vector<TSDFVoxel>& voxels() const noexcept { return voxels_; }
    [[nodiscard]] TSDFVoxel get_voxel(size_t ix, size_t iy, size_t iz) const;
    void set_voxel(size_t ix, size_t iy, size_t iz, float tsdf, float weight);

    Vector3 voxel_to_world(size_t ix, size_t iy, size_t iz) const;

    void integrate_depth(const std::vector<float>& depth_map, const CameraModel& cam, const Matrix4& T_cw);

private:
    Vector3 origin_;
    Vector3 size_;
    double voxel_size_;
    double trunc_dist_;

    size_t dim_x_{0};
    size_t dim_y_{0};
    size_t dim_z_{0};
    std::vector<TSDFVoxel> voxels_;
};

} // namespace geometrycore

#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <vector>
#include <utility>

namespace geometrycore {

class PointCloud {
public:
    std::vector<Vector3> points;
    std::vector<Vector3> normals;
    std::vector<Vector3> colors;

    PointCloud() = default;
    explicit PointCloud(std::vector<Vector3> pts) : points(std::move(pts)) {}

    [[nodiscard]] size_t size() const noexcept { return points.size(); }
    [[nodiscard]] bool empty() const noexcept { return points.empty(); }

    [[nodiscard]] Vector3 compute_mean() const {
        if (points.empty()) return Vector3::zero();
        Vector3 sum = Vector3::zero();
        for (const auto& p : points) sum += p;
        return sum / static_cast<double>(points.size());
    }

    [[nodiscard]] std::pair<Vector3, Vector3> compute_bounding_box() const {
        if (points.empty()) return {Vector3::zero(), Vector3::zero()};
        Vector3 min_pt = points[0];
        Vector3 max_pt = points[0];
        for (const auto& p : points) {
            min_pt.x = std::min(min_pt.x, p.x);
            min_pt.y = std::min(min_pt.y, p.y);
            min_pt.z = std::min(min_pt.z, p.z);

            max_pt.x = std::max(max_pt.x, p.x);
            max_pt.y = std::max(max_pt.y, p.y);
            max_pt.z = std::max(max_pt.z, p.z);
        }
        return {min_pt, max_pt};
    }

    void transform(const Matrix3& R, const Vector3& t) {
        for (auto& p : points) {
            p = R * p + t;
        }
        for (auto& n : normals) {
            n = R * n;
        }
    }
};

} // namespace geometrycore

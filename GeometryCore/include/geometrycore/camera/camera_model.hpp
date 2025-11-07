#pragma once

#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <utility>

namespace geometrycore {

class CameraModel {
public:
    double fx{500.0};
    double fy{500.0};
    double cx{320.0};
    double cy{240.0};

    double k1{0.0};
    double k2{0.0};
    double p1{0.0};
    double p2{0.0};

    size_t width{640};
    size_t height{480};

    CameraModel() = default;
    CameraModel(double fx, double fy, double cx, double cy, size_t w = 640, size_t h = 480)
        : fx(fx), fy(fy), cx(cx), cy(cy), width(w), height(h) {}

    [[nodiscard]] Matrix3 K() const {
        return {fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0};
    }

    [[nodiscard]] std::pair<double, double> project_normalized(const Vector3& p_cam) const {
        if (p_cam.z <= 1e-6) return {0.0, 0.0};
        double x = p_cam.x / p_cam.z;
        double y = p_cam.y / p_cam.z;

        // Apply radial & tangential distortion
        double r2 = x * x + y * y;
        double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        double x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        return {x_dist, y_dist};
    }

    [[nodiscard]] std::pair<double, double> project_pixel(const Vector3& p_cam) const {
        auto [xn, yn] = project_normalized(p_cam);
        return {fx * xn + cx, fy * yn + cy};
    }

    [[nodiscard]] Vector3 unproject_pixel(double u, double v, double depth) const {
        double xn = (u - cx) / fx;
        double yn = (v - cy) / fy;
        return {xn * depth, yn * depth, depth};
    }
};

} // namespace geometrycore

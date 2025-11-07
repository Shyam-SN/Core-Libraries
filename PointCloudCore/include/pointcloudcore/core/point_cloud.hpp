#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <utility>
#include <stdexcept>

namespace pointcloudcore {

struct Vector3d {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vector3d() = default;
    Vector3d(double x, double y, double z) : x(x), y(y), z(z) {}

    [[nodiscard]] double norm() const { return std::sqrt(x * x + y * y + z * z); }
    [[nodiscard]] double squared_norm() const { return x * x + y * y + z * z; }

    [[nodiscard]] Vector3d normalized() const {
        double n = norm();
        if (n < 1e-12) return {0.0, 0.0, 0.0};
        return {x / n, y / n, z / n};
    }

    void normalize() {
        double n = norm();
        if (n >= 1e-12) {
            x /= n; y /= n; z /= n;
        }
    }

    [[nodiscard]] double dot(const Vector3d& other) const { return x * other.x + y * other.y + z * other.z; }
    [[nodiscard]] Vector3d cross(const Vector3d& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    Vector3d operator+(const Vector3d& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vector3d operator-(const Vector3d& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vector3d operator*(double s) const { return {x * s, y * s, z * s}; }
    Vector3d operator/(double s) const { return {x / s, y / s, z / s}; }
    Vector3d operator-() const { return {-x, -y, -z}; }

    Vector3d& operator+=(const Vector3d& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vector3d& operator-=(const Vector3d& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vector3d& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    Vector3d& operator/=(double s) { x /= s; y /= s; z /= s; return *this; }

    double operator[](size_t i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        throw std::out_of_range("Vector3d index out of range");
    }

    double& operator[](size_t i) {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        throw std::out_of_range("Vector3d index out of range");
    }
};

class PointCloud {
public:
    std::vector<Vector3d> points;
    std::vector<Vector3d> normals;
    std::vector<Vector3d> colors;
    std::vector<std::vector<double>> features;

    PointCloud() = default;
    explicit PointCloud(std::vector<Vector3d> pts) : points(std::move(pts)) {}

    [[nodiscard]] size_t size() const noexcept { return points.size(); }
    [[nodiscard]] bool empty() const noexcept { return points.empty(); }

    [[nodiscard]] Vector3d compute_centroid() const {
        if (points.empty()) return {0.0, 0.0, 0.0};
        Vector3d sum{0.0, 0.0, 0.0};
        for (const auto& p : points) sum += p;
        return sum / static_cast<double>(points.size());
    }

    [[nodiscard]] std::pair<Vector3d, Vector3d> compute_bounding_box() const {
        if (points.empty()) return {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        Vector3d min_p = points[0];
        Vector3d max_p = points[0];
        for (const auto& p : points) {
            min_p.x = std::min(min_p.x, p.x);
            min_p.y = std::min(min_p.y, p.y);
            min_p.z = std::min(min_p.z, p.z);

            max_p.x = std::max(max_p.x, p.x);
            max_p.y = std::max(max_p.y, p.y);
            max_p.z = std::max(max_p.z, p.z);
        }
        return {min_p, max_p};
    }
};

} // namespace pointcloudcore

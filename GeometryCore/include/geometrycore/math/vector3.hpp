#pragma once

#include <geometrycore/math/concepts.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace geometrycore {

class Vector3 {
public:
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vector3() = default;
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

    [[nodiscard]] static Vector3 zero() { return {0.0, 0.0, 0.0}; }
    [[nodiscard]] static Vector3 ones() { return {1.0, 1.0, 1.0}; }
    [[nodiscard]] static Vector3 unit_x() { return {1.0, 0.0, 0.0}; }
    [[nodiscard]] static Vector3 unit_y() { return {0.0, 1.0, 0.0}; }
    [[nodiscard]] static Vector3 unit_z() { return {0.0, 0.0, 1.0}; }

    [[nodiscard]] double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    [[nodiscard]] double squared_norm() const {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] Vector3 normalized() const {
        double n = norm();
        if (n < 1e-12) return zero();
        return {x / n, y / n, z / n};
    }

    void normalize() {
        double n = norm();
        if (n >= 1e-12) {
            x /= n; y /= n; z /= n;
        }
    }

    [[nodiscard]] double dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    [[nodiscard]] Vector3 cross(const Vector3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    [[nodiscard]] double distance(const Vector3& other) const {
        return (*this - other).norm();
    }

    Vector3 operator+(const Vector3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vector3 operator-(const Vector3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vector3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vector3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vector3 operator-() const { return {-x, -y, -z}; }

    Vector3& operator+=(const Vector3& other) { x += other.x; y += other.y; z += other.z; return *this; }
    Vector3& operator-=(const Vector3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    Vector3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    Vector3& operator/=(double s) { x /= s; y /= s; z /= s; return *this; }

    bool operator==(const Vector3& other) const {
        return std::abs(x - other.x) < 1e-9 && std::abs(y - other.y) < 1e-9 && std::abs(z - other.z) < 1e-9;
    }

    double operator[](size_t i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        throw std::out_of_range("Vector3 index out of range");
    }

    double& operator[](size_t i) {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        throw std::out_of_range("Vector3 index out of range");
    }
};

inline Vector3 operator*(double s, const Vector3& v) {
    return v * s;
}

} // namespace geometrycore

#pragma once

namespace visioncore {

/**
 * @brief Representation of a 2D geometric point with double precision.
 */
struct Point2D {
    double x; ///< X coordinate
    double y; ///< Y coordinate

    Point2D() : x(0.0), y(0.0) {}
    Point2D(double xx, double yy) : x(xx), y(yy) {}
};

/**
 * @brief Representation of a 3D geometric point with double precision.
 */
struct Point3D {
    double x; ///< X coordinate
    double y; ///< Y coordinate
    double z; ///< Z coordinate

    Point3D() : x(0.0), y(0.0), z(0.0) {}
    Point3D(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
};


} // namespace visioncore

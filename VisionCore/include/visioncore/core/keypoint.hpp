#pragma once

namespace visioncore {

/**
 * @brief Representation of a detected keypoint / interest point.
 */
struct Keypoint {
    float x;        ///< X-coordinate of the keypoint
    float y;        ///< Y-coordinate of the keypoint
    float response; ///< Response strength / corner score
    float size;     ///< Diameter of the meaningful keypoint neighborhood
    float angle;    ///< Orientation angle in degrees (default to -1 if unoriented)

    Keypoint() : x(0.0f), y(0.0f), response(0.0f), size(0.0f), angle(-1.0f) {}
    Keypoint(float xx, float yy, float resp = 0.0f, float sz = 0.0f, float ang = -1.0f)
        : x(xx), y(yy), response(resp), size(sz), angle(ang) {}
};

} // namespace visioncore

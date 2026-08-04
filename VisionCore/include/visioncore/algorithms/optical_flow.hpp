#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/gradients.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Performs bilinear interpolation on a single-channel image view at coordinate (x, y).
 */
template <typename T>
inline float interpolate_bilinear(const ImageView<const T, 1>& img, float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::clamp(x0, 0, static_cast<int>(img.width()) - 1);
    x1 = std::clamp(x1, 0, static_cast<int>(img.width()) - 1);
    y0 = std::clamp(y0, 0, static_cast<int>(img.height()) - 1);
    y1 = std::clamp(y1, 0, static_cast<int>(img.height()) - 1);

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    float v00 = static_cast<float>(img(x0, y0));
    float v10 = static_cast<float>(img(x1, y0));
    float v01 = static_cast<float>(img(x0, y1));
    float v11 = static_cast<float>(img(x1, y1));

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v10 +
           (1.0f - dx) * dy * v01 +
           dx * dy * v11;
}

/**
 * @brief Calculates Lucas-Kanade optical flow for a set of sparse keypoints from prev_img to next_img.
 * Uses iterative Gauss-Newton subpixel tracking.
 * 
 * @param prev_img Grayscale image view of the first frame.
 * @param next_img Grayscale image view of the second frame.
 * @param prev_pts Vector of input keypoints from the first frame.
 * @param next_pts Vector of tracked output keypoints in the second frame.
 * @param status Status vector (1 for successfully tracked, 0 for lost).
 * @param win_size Width/height of the local tracking window (must be odd).
 * @param max_iter Maximum iterations for Gauss-Newton refinement.
 * @param min_determinant Minimum determinant threshold for structure tensor stability.
 */
inline void calc_optical_flow_lk(const ImageView<const uint8_t, 1>& prev_img,
                                 const ImageView<const uint8_t, 1>& next_img,
                                 const std::vector<Keypoint>& prev_pts,
                                 std::vector<Keypoint>& next_pts,
                                 std::vector<uint8_t>& status,
                                 size_t win_size = 15,
                                 size_t max_iter = 20,
                                 float min_determinant = 1e-4f) {
    if (prev_img.width() != next_img.width() || prev_img.height() != next_img.height()) {
        throw std::invalid_argument("Frame dimensions must match.");
    }
    if (win_size % 2 == 0) {
        throw std::invalid_argument("Window size must be odd.");
    }

    const int half_w = static_cast<int>(win_size / 2);
    const size_t num_pts = prev_pts.size();

    next_pts.resize(num_pts);
    status.assign(num_pts, 0);

    // Compute spatial gradients of the first frame
    const size_t w = prev_img.width();
    const size_t h = prev_img.height();
    Image<float, 1> dx(w, h);
    Image<float, 1> dy(w, h);
    sobel_x(prev_img, dx.view(), BorderType::REPLICATE);
    sobel_y(prev_img, dy.view(), BorderType::REPLICATE);

    for (size_t i = 0; i < num_pts; ++i) {
        const auto& pt = prev_pts[i];
        float px = pt.x;
        float py = pt.y;

        // Verify if keypoint is within tracking boundaries
        if (px < half_w || px >= w - half_w || py < half_w || py >= h - half_w) {
            status[i] = 0;
            continue;
        }

        // Initialize tracked coordinate
        float ux = px;
        float uy = py;

        bool success = false;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Build the 2x2 structure tensor components
            // G = [ sum(Ix^2)   sum(Ix*Iy) ]
            //     [ sum(Ix*Iy)  sum(Iy^2)  ]
            // b = [ sum(Ix * It) ]
            //     [ sum(Iy * It) ]
            double sIxx = 0.0;
            double sIyy = 0.0;
            double sIxy = 0.0;
            double sIxt = 0.0;
            double sIyt = 0.0;

            for (int dy_offset = -half_w; dy_offset <= half_w; ++dy_offset) {
                for (int dx_offset = -half_w; dx_offset <= half_w; ++dx_offset) {
                    float x_pos = px + dx_offset;
                    float y_pos = py + dy_offset;

                    // Spatial gradients from previous frame at current window position
                    float ix_val = interpolate_bilinear(dx.const_view(), x_pos, y_pos);
                    float iy_val = interpolate_bilinear(dy.const_view(), x_pos, y_pos);

                    // Interpolated pixel values
                    float prev_val = interpolate_bilinear(prev_img, x_pos, y_pos);
                    // Sample next frame at current tracked location
                    float next_val = interpolate_bilinear(next_img, ux + dx_offset, uy + dy_offset);

                    // Temporal gradient It = next - prev
                    float it_val = next_val - prev_val;

                    sIxx += ix_val * ix_val;
                    sIyy += iy_val * iy_val;
                    sIxy += ix_val * iy_val;
                    sIxt += ix_val * it_val;
                    sIyt += iy_val * it_val;
                }
            }

            // Determinant check
            double det = sIxx * sIyy - sIxy * sIxy;
            if (std::abs(det) < min_determinant) {
                break; // Singular matrix, stop tracking this point
            }

            // Cramer's rule to solve the linear system:
            // [ sIxx  sIxy ] [ delta_u ] = [ -sIxt ]
            // [ sIxy  sIyy ] [ delta_v ]   [ -sIyt ]
            double delta_u = (-sIxt * sIyy - sIxy * -sIyt) / det;
            double delta_v = (sIxx * -sIyt - -sIxt * sIxy) / det;

            ux += static_cast<float>(delta_u);
            uy += static_cast<float>(delta_v);

            // Converged if update is extremely small
            if (delta_u * delta_u + delta_v * delta_v < 1e-4) {
                success = true;
                break;
            }
        }

        // Final boundary checks for tracked coordinates
        if (success && ux >= half_w && ux < w - half_w && uy >= half_w && uy < h - half_w) {
            next_pts[i] = Keypoint(ux, uy, pt.response, pt.size, pt.angle);
            status[i] = 1;
        } else {
            status[i] = 0;
        }
    }
}

} // namespace visioncore

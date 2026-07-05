#pragma once

#include <visioncore/core/image.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <visioncore/algorithms/convolution.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace visioncore {

/**
 * @brief Performs Harris Corner Detection on a grayscale image view.
 */
inline void detect_harris(const ImageView<const uint8_t, 1>& src,
                          std::vector<Keypoint>& keypoints,
                          double k = 0.04,
                          double threshold = 1e5,
                          size_t block_size = 3,
                          BorderType border = BorderType::REPLICATE) {
    if (block_size % 2 == 0) {
        throw std::invalid_argument("Block size must be odd.");
    }

    const size_t w = src.width();
    const size_t h = src.height();

    // 1. Calculate Sobel X and Sobel Y
    Image<float, 1> dx(w, h);
    Image<float, 1> dy(w, h);
    sobel_x(src, dx.view(), border);
    sobel_y(src, dy.view(), border);

    // 2. Compute structure tensor products: Ixx, Iyy, Ixy
    Image<float, 1> Ixx(w, h);
    Image<float, 1> Iyy(w, h);
    Image<float, 1> Ixy(w, h);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float gx = dx(x, y);
            float gy = dy(x, y);
            Ixx(x, y) = gx * gx;
            Iyy(x, y) = gy * gy;
            Ixy(x, y) = gx * gy;
        }
    }

    // 3. Smooth structure tensor elements over block window
    Image<float, 1> sIxx(w, h);
    Image<float, 1> sIyy(w, h);
    Image<float, 1> sIxy(w, h);
    box_filter(Ixx.const_view(), sIxx.view(), block_size, border);
    box_filter(Iyy.const_view(), sIyy.view(), block_size, border);
    box_filter(Ixy.const_view(), sIxy.view(), block_size, border);

    // 4. Calculate Harris response R = det(M) - k * trace(M)^2
    Image<float, 1> response(w, h, 0.0f);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float a = sIxx(x, y);
            float b = sIxy(x, y);
            float c = sIyy(x, y);

            float det = a * c - b * b;
            float trace = a + c;
            response(x, y) = det - static_cast<float>(k) * trace * trace;
        }
    }

    // 5. Non-Maximum Suppression (8-neighborhood)
    keypoints.clear();
    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            float val = response(x, y);
            if (val >= threshold) {
                // Check if it's a local maximum in a 3x3 window
                bool is_max = true;
                for (int dy_offset = -1; dy_offset <= 1; ++dy_offset) {
                    for (int dx_offset = -1; dx_offset <= 1; ++dx_offset) {
                        if (dx_offset == 0 && dy_offset == 0) continue;
                        if (response(x + dx_offset, y + dy_offset) > val) {
                            is_max = false;
                            break;
                        }
                    }
                    if (!is_max) break;
                }

                if (is_max) {
                    keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y), val, static_cast<float>(block_size));
                }
            }
        }
    }
}

/**
 * @brief Performs FAST-9 corner detection on a grayscale image view.
 */
inline void detect_fast(const ImageView<const uint8_t, 1>& src,
                         std::vector<Keypoint>& keypoints,
                         int threshold = 20,
                         bool nonmax_suppression = true) {
    const int w = static_cast<int>(src.width());
    const int h = static_cast<int>(src.height());

    // Bresenham circle offset coordinates (16 pixels)
    const int offsets[16][2] = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1},
        {3, 0},  {3, 1},  {2, 2},  {1, 3},
        {0, 3},  {-1, 3}, {-2, 2}, {-3, 1},
        {-3, 0}, {-3, -1},{-2, -2},{-1, -3}
    };

    // Buffer to hold corner scores for non-maximum suppression
    Image<int, 1> scores(src.width(), src.height(), 0);
    std::vector<std::pair<int, int>> candidates;

    for (int y = 3; y < h - 3; ++y) {
        for (int x = 3; x < w - 3; ++x) {
            int ip = src(x, y);
            int t_low = ip - threshold;
            int t_high = ip + threshold;

            // 1. High-speed check (pixels 1, 9, 5, 13)
            int p1 = src(x + offsets[0][0], y + offsets[0][1]);
            int p9 = src(x + offsets[8][0], y + offsets[8][1]);
            
            // Check top/bottom first
            int count_brighter = 0;
            int count_darker = 0;

            if (p1 > t_high) count_brighter++;
            else if (p1 < t_low) count_darker++;

            if (p9 > t_high) count_brighter++;
            else if (p9 < t_low) count_darker++;

            if (count_brighter == 0 && count_darker == 0) continue;

            // Check right/left
            int p5 = src(x + offsets[4][0], y + offsets[4][1]);
            int p13 = src(x + offsets[12][0], y + offsets[12][1]);

            if (p5 > t_high) count_brighter++;
            else if (p5 < t_low) count_darker++;

            if (p13 > t_high) count_brighter++;
            else if (p13 < t_low) count_darker++;

            // Must have at least 3 pixels brighter or darker out of the 4 cardinials
            if (count_brighter < 3 && count_darker < 3) continue;

            // 2. Full 16-pixel circle check for 9 contiguous pixels
            // We retrieve all 16 values
            int circle[16];
            for (int i = 0; i < 16; ++i) {
                circle[i] = src(x + offsets[i][0], y + offsets[i][1]);
            }

            // Look for 9 contiguous pixels that are either all > t_high or all < t_low
            bool is_corner = false;
            int active_type = 0; // 1 for brighter, -1 for darker

            // Check brighter contiguous segment
            for (int start = 0; start < 16; ++start) {
                bool ok = true;
                for (int len = 0; len < 9; ++len) {
                    if (circle[(start + len) % 16] <= t_high) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    is_corner = true;
                    active_type = 1;
                    break;
                }
            }

            if (!is_corner) {
                // Check darker contiguous segment
                for (int start = 0; start < 16; ++start) {
                    bool ok = true;
                    for (int len = 0; len < 9; ++len) {
                        if (circle[(start + len) % 16] >= t_low) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        is_corner = true;
                        active_type = -1;
                        break;
                    }
                }
            }

            if (is_corner) {
                // 3. Compute Corner Score (sum of absolute differences of contiguous segment)
                // Score metric: sum of absolute differences minus threshold
                int max_score = 0;
                for (int start = 0; start < 16; ++start) {
                    bool ok = true;
                    int current_score = 0;
                    for (int len = 0; len < 9; ++len) {
                        int val = circle[(start + len) % 16];
                        if (active_type == 1 && val <= t_high) { ok = false; break; }
                        if (active_type == -1 && val >= t_low) { ok = false; break; }
                        current_score += std::abs(val - ip) - threshold;
                    }
                    if (ok) {
                        max_score = std::max(max_score, current_score);
                    }
                }

                scores(x, y) = max_score;
                candidates.push_back({x, y});
            }
        }
    }

    keypoints.clear();

    if (nonmax_suppression) {
        // NMS: Check 3x3 neighborhood around candidates
        for (const auto& [cx, cy] : candidates) {
            int score = scores(cx, cy);
            bool is_max = true;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (scores(cx + dx, cy + dy) > score) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) break;
            }

            if (is_max) {
                keypoints.emplace_back(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(score), 7.0f);
            }
        }
    } else {
        for (const auto& [cx, cy] : candidates) {
            keypoints.emplace_back(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(scores(cx, cy)), 7.0f);
        }
    }
}

} // namespace visioncore

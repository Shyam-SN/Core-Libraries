#pragma once

#include <visioncore/core/image.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace visioncore {

/**
 * @brief Template matching similarity/difference metrics.
 */
enum class TemplateMetric {
    SQDIFF,        ///< Sum of squared differences (min is best match)
    CCORR,         ///< Cross-correlation (max is best match)
    CCOEFF_NORMED  ///< Normalized correlation coefficient (max is best match, range [-1, 1])
};

/**
 * @brief Slides a template over an image and computes a matching response map.
 * 
 * @param src Grayscale input image view.
 * @param templ Grayscale template patch view. Must be smaller than or equal to src.
 * @param result Floating-point response map view. Dimensions must be exactly 
 *               (src.width() - templ.width() + 1) x (src.height() - templ.height() + 1).
 * @param metric TemplateMetric choice.
 */
inline void match_template(const ImageView<const uint8_t, 1>& src,
                           const ImageView<const uint8_t, 1>& templ,
                           const ImageView<float, 1>& result,
                           TemplateMetric metric) {
    const size_t w = src.width();
    const size_t h = src.height();
    const size_t tw = templ.width();
    const size_t th = templ.height();

    if (tw > w || th > h) {
        throw std::invalid_argument("Template dimensions cannot exceed source dimensions.");
    }

    const size_t rw = w - tw + 1;
    const size_t rh = h - th + 1;

    if (result.width() != rw || result.height() != rh) {
        throw std::invalid_argument("Result image dimensions do not match the expected matching boundary.");
    }

    // Precomputations for CCOEFF_NORMED
    double templ_sum = 0.0;
    double templ_sq_sum = 0.0;
    const double templ_pixels = static_cast<double>(tw * th);

    if (metric == TemplateMetric::CCOEFF_NORMED) {
        for (size_t ty = 0; ty < th; ++ty) {
            for (size_t tx = 0; tx < tw; ++tx) {
                double val = templ(tx, ty);
                templ_sum += val;
                templ_sq_sum += val * val;
            }
        }
    }

    // Sum of (T - mean_T)^2
    double templ_variance = templ_sq_sum - (templ_sum * templ_sum) / templ_pixels;
    if (templ_variance < 1e-6) {
        templ_variance = 1e-6; // avoid division by zero
    }

    for (size_t y = 0; y < rh; ++y) {
        for (size_t x = 0; x < rw; ++x) {
            double sum = 0.0;

            if (metric == TemplateMetric::SQDIFF) {
                for (size_t ty = 0; ty < th; ++ty) {
                    for (size_t tx = 0; tx < tw; ++tx) {
                        double diff = static_cast<double>(templ(tx, ty)) - static_cast<double>(src(x + tx, y + ty));
                        sum += diff * diff;
                    }
                }
                result(x, y) = static_cast<float>(sum);

            } else if (metric == TemplateMetric::CCORR) {
                for (size_t ty = 0; ty < th; ++ty) {
                    for (size_t tx = 0; tx < tw; ++tx) {
                        sum += static_cast<double>(templ(tx, ty)) * static_cast<double>(src(x + tx, y + ty));
                    }
                }
                result(x, y) = static_cast<float>(sum);

            } else if (metric == TemplateMetric::CCOEFF_NORMED) {
                double patch_sum = 0.0;
                double patch_sq_sum = 0.0;
                double cross_sum = 0.0;

                for (size_t ty = 0; ty < th; ++ty) {
                    for (size_t tx = 0; tx < tw; ++tx) {
                        double val_i = src(x + tx, y + ty);
                        double val_t = templ(tx, ty);
                        patch_sum += val_i;
                        patch_sq_sum += val_i * val_i;
                        cross_sum += val_i * val_t;
                    }
                }

                double patch_variance = patch_sq_sum - (patch_sum * patch_sum) / templ_pixels;
                if (patch_variance < 1e-6) {
                    patch_variance = 1e-6;
                }

                // Covariance: sum((I - mean_I) * (T - mean_T)) = sum(I*T) - mean_I * sum(T) - mean_T * sum(I) + pixels * mean_I * mean_T
                // Which simplifies to: sum(I*T) - (sum(I)*sum(T))/pixels
                double covariance = cross_sum - (patch_sum * templ_sum) / templ_pixels;

                double denom = std::sqrt(templ_variance * patch_variance);
                result(x, y) = static_cast<float>(std::clamp(covariance / denom, -1.0, 1.0));
            }
        }
    }
}

/**
 * @brief Helper function to locate the global minimum and maximum intensity values in a response map.
 * Used to identify the best match coordinates in template matching.
 */
inline void min_max_loc(const ImageView<const float, 1>& response,
                        float& min_val, float& max_val,
                        size_t& min_x, size_t& min_y,
                        size_t& max_x, size_t& max_y) {
    min_val = std::numeric_limits<float>::max();
    max_val = -std::numeric_limits<float>::max();
    min_x = 0; min_y = 0;
    max_x = 0; max_y = 0;

    for (size_t y = 0; y < response.height(); ++y) {
        for (size_t x = 0; x < response.width(); ++x) {
            float val = response(x, y);
            if (val < min_val) {
                min_val = val;
                min_x = x;
                min_y = y;
            }
            if (val > max_val) {
                max_val = val;
                max_x = x;
                max_y = y;
            }
        }
    }
}

} // namespace visioncore

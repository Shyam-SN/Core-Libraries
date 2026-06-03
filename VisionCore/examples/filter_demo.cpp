#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <visioncore/algorithms/histogram.hpp>
#include <visioncore/algorithms/thresholding.hpp>
#include <visioncore/algorithms/morphology.hpp>
#include <visioncore/algorithms/gradients.hpp>
#include <visioncore/algorithms/canny.hpp>
#include <visioncore/algorithms/corner_detection.hpp>
#include <visioncore/core/keypoint.hpp>
#include <visioncore/algorithms/orb.hpp>
#include <visioncore/algorithms/feature_matching.hpp>
#include <visioncore/algorithms/template_matching.hpp>
#include <visioncore/algorithms/optical_flow.hpp>
#include <visioncore/core/point.hpp>
#include <visioncore/algorithms/homography.hpp>
#include <visioncore/algorithms/calibration.hpp>
#include <visioncore/algorithms/epipolar.hpp>
#include <visioncore/algorithms/stereo.hpp>
#include <visioncore/algorithms/pnp.hpp>
#include <visioncore/algorithms/ransac.hpp>
#include <visioncore/algorithms/bundle_adjustment.hpp>
#include <visioncore/utils/image_io.hpp>
#include <iostream>
#include <chrono>
#include <string>

using namespace visioncore;

// Generates a synthetic checkerboard image if no image path is provided
Image<uint8_t, 1> create_synthetic_image(size_t width, size_t height) {
    Image<uint8_t, 1> img(width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            // Checkerboard pattern with a vertical/horizontal cross
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            img(x, y) = checker ? 255 : 50;
            if (x == width / 2 || y == height / 2) {
                img(x, y) = 150; // Draw lines in the middle
            }
        }
    }
    return img;
}
// Draw detected keypoints as crosses on an output image for visualization
void draw_keypoints(const ImageView<const uint8_t, 1>& src, ImageView<uint8_t, 1> dst, const std::vector<Keypoint>& keypoints) {
    for (size_t y = 0; y < src.height(); ++y) {
        for (size_t x = 0; x < src.width(); ++x) {
            dst(x, y) = src(x, y);
        }
    }
    for (const auto& kp : keypoints) {
        int kx = static_cast<int>(std::round(kp.x));
        int ky = static_cast<int>(std::round(kp.y));
        for (int d = -2; d <= 2; ++d) {
            if (kx + d >= 0 && kx + d < static_cast<int>(src.width())) dst(kx + d, ky) = 255;
            if (ky + d >= 0 && ky + d < static_cast<int>(src.height())) dst(kx, ky + d) = 255;
        }
    }
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    if (argc > 1) {
        input_path = argv[1];
    }

    Image<uint8_t, 1> input_img;
    std::vector<Keypoint> fast_kps;

    if (!input_path.empty()) {
        std::cout << "[INFO] Loading image: " << input_path << "\n";
        auto loaded = utils::load_grayscale(input_path);
        if (!loaded) {
            std::cerr << "[ERROR] Failed to load image: " << input_path << ". Falling back to synthetic image.\n";
            input_img = create_synthetic_image(512, 512);
        } else {
            input_img = std::move(*loaded);
        }
    } else {
        std::cout << "[INFO] No input image provided. Generating 512x512 synthetic pattern...\n";
        input_img = create_synthetic_image(512, 512);
        // Save the synthetic original so the user can inspect it
        utils::save_image("synthetic_original.png", input_img.const_view());
    }

    std::cout << "[INFO] Image loaded successfully (" << input_img.width() << "x" << input_img.height() << ").\n";

    // Allocate output images
    Image<uint8_t, 1> img_box(input_img.width(), input_img.height());
    Image<uint8_t, 1> img_gauss(input_img.width(), input_img.height());
    
    // Sobel gradients need to store negative values, so we use float
    Image<float, 1> img_sobel_x(input_img.width(), input_img.height());
    Image<float, 1> img_sobel_y(input_img.width(), input_img.height());
    Image<uint8_t, 1> img_sobel_mag(input_img.width(), input_img.height());

    // 1. Box Filter
    {
        auto start = std::chrono::high_resolution_clock::now();
        box_filter(input_img.const_view(), img_box.view(), 7, BorderType::REPLICATE);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[BENCHMARK] Box Filter (7x7): " << duration.count() << " ms\n";
        utils::save_image("output_box.png", img_box.const_view());
    }

    // 2. Gaussian Blur
    {
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_blur(input_img.const_view(), img_gauss.view(), 7, 1.5, BorderType::REPLICATE);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[BENCHMARK] Gaussian Blur (7x7, sigma=1.5): " << duration.count() << " ms\n";
        utils::save_image("output_gaussian.png", img_gauss.const_view());
    }

    // 3. Sobel Derivatives & Magnitude
    {
        auto start_x = std::chrono::high_resolution_clock::now();
        sobel_x(input_img.const_view(), img_sobel_x.view(), BorderType::REPLICATE);
        auto end_x = std::chrono::high_resolution_clock::now();
        
        auto start_y = std::chrono::high_resolution_clock::now();
        sobel_y(input_img.const_view(), img_sobel_y.view(), BorderType::REPLICATE);
        auto end_y = std::chrono::high_resolution_clock::now();
        
        auto start_mag = std::chrono::high_resolution_clock::now();
        sobel_magnitude(img_sobel_x.const_view(), img_sobel_y.const_view(), img_sobel_mag.view());
        auto end_mag = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_x = end_x - start_x;
        std::chrono::duration<double, std::milli> dur_y = end_y - start_y;
        std::chrono::duration<double, std::milli> dur_mag = end_mag - start_mag;

        std::cout << "[BENCHMARK] Sobel X: " << dur_x.count() << " ms\n";
        std::cout << "[BENCHMARK] Sobel Y: " << dur_y.count() << " ms\n";
        std::cout << "[BENCHMARK] Sobel Magnitude: " << dur_mag.count() << " ms\n";

        utils::save_image("output_sobel_mag.png", img_sobel_mag.const_view());
        
        // Save normalized visual representations of dx and dy (map [-255, 255] to [0, 255])
        Image<uint8_t, 1> vis_dx(input_img.width(), input_img.height());
        Image<uint8_t, 1> vis_dy(input_img.width(), input_img.height());
        for (size_t y = 0; y < input_img.height(); ++y) {
            for (size_t x = 0; x < input_img.width(); ++x) {
                // Map Sobel outputs (which range from -1020 to 1020 for uint8 input but we clamp/normalize for display)
                vis_dx(x, y) = static_cast<uint8_t>(std::clamp((img_sobel_x(x, y) + 255.0f) * 0.5f, 0.0f, 255.0f));
                vis_dy(x, y) = static_cast<uint8_t>(std::clamp((img_sobel_y(x, y) + 255.0f) * 0.5f, 0.0f, 255.0f));
            }
        }
        utils::save_image("output_sobel_x.png", vis_dx.const_view());
        utils::save_image("output_sobel_y.png", vis_dy.const_view());
    }

    // 4. Histogram Equalization
    {
        Image<uint8_t, 1> img_equalized(input_img.width(), input_img.height());
        auto start = std::chrono::high_resolution_clock::now();
        equalize_histogram(input_img.const_view(), img_equalized.view());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[BENCHMARK] Histogram Equalization: " << duration.count() << " ms\n";
        utils::save_image("output_equalized.png", img_equalized.const_view());
    }

    // 5. Thresholding (Binary, Otsu, Adaptive)
    {
        Image<uint8_t, 1> img_thresh_bin(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_thresh_otsu(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_thresh_adapt(input_img.width(), input_img.height());

        // Global Binary (threshold = 127)
        auto start_bin = std::chrono::high_resolution_clock::now();
        threshold_binary(input_img.const_view(), img_thresh_bin.view(), 127, 255);
        auto end_bin = std::chrono::high_resolution_clock::now();
        
        // Otsu's Automatic
        auto start_otsu = std::chrono::high_resolution_clock::now();
        threshold_otsu(input_img.const_view(), img_thresh_otsu.view(), 255);
        auto end_otsu = std::chrono::high_resolution_clock::now();
        uint8_t otsu_val = compute_otsu_threshold(input_img.const_view());
        std::cout << "[INFO] Otsu calculated optimal threshold: " << static_cast<int>(otsu_val) << "\n";

        // Local Adaptive (Gaussian method, block size 15, constant C = 5.0)
        auto start_adapt = std::chrono::high_resolution_clock::now();
        threshold_adaptive(input_img.const_view(), img_thresh_adapt.view(), 15, 5.0, AdaptiveMethod::GAUSSIAN, 255);
        auto end_adapt = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_bin = end_bin - start_bin;
        std::chrono::duration<double, std::milli> dur_otsu = end_otsu - start_otsu;
        std::chrono::duration<double, std::milli> dur_adapt = end_adapt - start_adapt;

        std::cout << "[BENCHMARK] Global Binary Threshold: " << dur_bin.count() << " ms\n";
        std::cout << "[BENCHMARK] Otsu's Optimal Threshold: " << dur_otsu.count() << " ms\n";
        std::cout << "[BENCHMARK] Local Adaptive Threshold: " << dur_adapt.count() << " ms\n";

        utils::save_image("output_threshold_binary.png", img_thresh_bin.const_view());
        utils::save_image("output_threshold_otsu.png", img_thresh_otsu.const_view());
        utils::save_image("output_threshold_adaptive.png", img_thresh_adapt.const_view());
    }

    // 6. Morphological Operations (using 5x5 cross structuring element)
    {
        Image<uint8_t, 1> img_dilate(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_erode(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_open(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_close(input_img.width(), input_img.height());

        auto se = StructuringElement::cross(5, 5);

        auto start_dil = std::chrono::high_resolution_clock::now();
        dilate(input_img.const_view(), img_dilate.view(), se, BorderType::REPLICATE);
        auto end_dil = std::chrono::high_resolution_clock::now();

        auto start_ero = std::chrono::high_resolution_clock::now();
        erode(input_img.const_view(), img_erode.view(), se, BorderType::REPLICATE);
        auto end_ero = std::chrono::high_resolution_clock::now();

        auto start_op = std::chrono::high_resolution_clock::now();
        opening(input_img.const_view(), img_open.view(), se, BorderType::REPLICATE);
        auto end_op = std::chrono::high_resolution_clock::now();

        auto start_cl = std::chrono::high_resolution_clock::now();
        closing(input_img.const_view(), img_close.view(), se, BorderType::REPLICATE);
        auto end_cl = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_dil = end_dil - start_dil;
        std::chrono::duration<double, std::milli> dur_ero = end_ero - start_ero;
        std::chrono::duration<double, std::milli> dur_op = end_op - start_op;
        std::chrono::duration<double, std::milli> dur_cl = end_cl - start_cl;

        std::cout << "[BENCHMARK] Dilation (5x5 cross): " << dur_dil.count() << " ms\n";
        std::cout << "[BENCHMARK] Erosion (5x5 cross): " << dur_ero.count() << " ms\n";
        std::cout << "[BENCHMARK] Opening (5x5 cross): " << dur_op.count() << " ms\n";
        std::cout << "[BENCHMARK] Closing (5x5 cross): " << dur_cl.count() << " ms\n";

        utils::save_image("output_dilation.png", img_dilate.const_view());
        utils::save_image("output_erosion.png", img_erode.const_view());
        utils::save_image("output_opening.png", img_open.const_view());
        utils::save_image("output_closing.png", img_close.const_view());
    }

    // 7. Additional Gradients (Prewitt, Scharr, Laplacian)
    {
        Image<float, 1> img_prewitt_x(input_img.width(), input_img.height());
        Image<float, 1> img_prewitt_y(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_prewitt_mag(input_img.width(), input_img.height());

        Image<float, 1> img_scharr_x(input_img.width(), input_img.height());
        Image<float, 1> img_scharr_y(input_img.width(), input_img.height());
        Image<uint8_t, 1> img_scharr_mag(input_img.width(), input_img.height());

        Image<float, 1> img_laplacian(input_img.width(), input_img.height());
        Image<uint8_t, 1> vis_laplacian(input_img.width(), input_img.height());

        // Prewitt
        auto start_prewitt = std::chrono::high_resolution_clock::now();
        prewitt_x(input_img.const_view(), img_prewitt_x.view(), BorderType::REPLICATE);
        prewitt_y(input_img.const_view(), img_prewitt_y.view(), BorderType::REPLICATE);
        sobel_magnitude(img_prewitt_x.const_view(), img_prewitt_y.const_view(), img_prewitt_mag.view());
        auto end_prewitt = std::chrono::high_resolution_clock::now();

        // Scharr
        auto start_scharr = std::chrono::high_resolution_clock::now();
        scharr_x(input_img.const_view(), img_scharr_x.view(), BorderType::REPLICATE);
        scharr_y(input_img.const_view(), img_scharr_y.view(), BorderType::REPLICATE);
        sobel_magnitude(img_scharr_x.const_view(), img_scharr_y.const_view(), img_scharr_mag.view());
        auto end_scharr = std::chrono::high_resolution_clock::now();

        // Laplacian
        auto start_lap = std::chrono::high_resolution_clock::now();
        laplacian(input_img.const_view(), img_laplacian.view(), BorderType::REPLICATE);
        auto end_lap = std::chrono::high_resolution_clock::now();

        // Convert Laplacian for visualization (scale and shift to [0, 255])
        for (size_t y = 0; y < input_img.height(); ++y) {
            for (size_t x = 0; x < input_img.width(); ++x) {
                vis_laplacian(x, y) = static_cast<uint8_t>(std::clamp((img_laplacian(x, y) + 1020.0f) * (255.0f / 2040.0f), 0.0f, 255.0f));
            }
        }

        std::chrono::duration<double, std::milli> dur_prewitt = end_prewitt - start_prewitt;
        std::chrono::duration<double, std::milli> dur_scharr = end_scharr - start_scharr;
        std::chrono::duration<double, std::milli> dur_lap = end_lap - start_lap;

        std::cout << "[BENCHMARK] Prewitt Gradient & Mag: " << dur_prewitt.count() << " ms\n";
        std::cout << "[BENCHMARK] Scharr Gradient & Mag: " << dur_scharr.count() << " ms\n";
        std::cout << "[BENCHMARK] Laplacian Filter: " << dur_lap.count() << " ms\n";

        utils::save_image("output_prewitt_mag.png", img_prewitt_mag.const_view());
        utils::save_image("output_scharr_mag.png", img_scharr_mag.const_view());
        utils::save_image("output_laplacian.png", vis_laplacian.const_view());
    }

    // 8. Canny Edge Detection
    {
        Image<uint8_t, 1> img_canny(input_img.width(), input_img.height());

        auto start = std::chrono::high_resolution_clock::now();
        canny(input_img.const_view(), img_canny.view(), 50.0, 150.0, 3, 1.0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[BENCHMARK] Canny Edge Detection: " << duration.count() << " ms\n";

        utils::save_image("output_canny.png", img_canny.const_view());
    }

    // 9. Corner Detection (Harris and FAST)
    {
        std::vector<Keypoint> harris_kps;

        // Harris
        auto start_harris = std::chrono::high_resolution_clock::now();
        detect_harris(input_img.const_view(), harris_kps, 0.04, 1e5, 3, BorderType::REPLICATE);
        auto end_harris = std::chrono::high_resolution_clock::now();

        // FAST
        auto start_fast = std::chrono::high_resolution_clock::now();
        detect_fast(input_img.const_view(), fast_kps, 20, true);
        auto end_fast = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_harris = end_harris - start_harris;
        std::chrono::duration<double, std::milli> dur_fast = end_fast - start_fast;

        std::cout << "[BENCHMARK] Harris Corner Detection: " << dur_harris.count() << " ms (found " << harris_kps.size() << " corners)\n";
        std::cout << "[BENCHMARK] FAST-9 Corner Detection: " << dur_fast.count() << " ms (found " << fast_kps.size() << " corners)\n";

        Image<uint8_t, 1> vis_harris(input_img.width(), input_img.height());
        Image<uint8_t, 1> vis_fast(input_img.width(), input_img.height());

        draw_keypoints(input_img.const_view(), vis_harris.view(), harris_kps);
        draw_keypoints(input_img.const_view(), vis_fast.view(), fast_kps);

        utils::save_image("output_corners_harris.png", vis_harris.const_view());
        utils::save_image("output_corners_fast.png", vis_fast.const_view());

        // 10. ORB Descriptor Extraction
        std::vector<Keypoint> orb_kps = fast_kps; // Copy FAST keypoints
        std::vector<std::vector<uint8_t>> descriptors;

        auto start_orb = std::chrono::high_resolution_clock::now();
        compute_orb_descriptors(input_img.const_view(), orb_kps, descriptors);
        auto end_orb = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_orb = end_orb - start_orb;
        std::cout << "[BENCHMARK] ORB Descriptor Extraction: " << dur_orb.count() << " ms (processed " 
                  << orb_kps.size() << " valid oriented keypoints)\n";

        if (!descriptors.empty()) {
            std::cout << "[INFO] First ORB descriptor (32 bytes hex): ";
            for (uint8_t byte : descriptors[0]) {
                printf("%02x", byte);
            }
            std::cout << "\n";
        }

        // 11. Feature Matching Demo
        std::vector<std::vector<uint8_t>> query_descs = descriptors;
        std::vector<std::vector<uint8_t>> train_descs = descriptors;

        // Perturb the train set slightly to simulate real noise (flip 1 bit in some descriptors)
        for (size_t i = 0; i < train_descs.size(); ++i) {
            if (i % 2 == 0 && !train_descs[i].empty()) {
                train_descs[i][0] ^= 0x01; // flip 1 bit
            }
        }

        std::vector<Match> bf_matches;
        std::vector<Match> ratio_matches;

        auto start_bf = std::chrono::high_resolution_clock::now();
        match_brute_force(query_descs, train_descs, bf_matches);
        auto end_bf = std::chrono::high_resolution_clock::now();

        auto start_ratio = std::chrono::high_resolution_clock::now();
        match_ratio_test(query_descs, train_descs, ratio_matches, 0.8f);
        auto end_ratio = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_bf = end_bf - start_bf;
        std::chrono::duration<double, std::milli> dur_ratio = end_ratio - start_ratio;

        std::cout << "[BENCHMARK] Brute-Force Matching: " << dur_bf.count() << " ms (found " << bf_matches.size() << " matches)\n";
        std::cout << "[BENCHMARK] Ratio-Test Matching (0.8 threshold): " << dur_ratio.count() << " ms (found " << ratio_matches.size() << " matches)\n";
    }

    // 12. Template Matching Demo
    if (input_img.width() >= 160 && input_img.height() >= 160) {
        // Crop a 32x32 template from coordinates (64, 64)
        Image<uint8_t, 1> templ(32, 32);
        for (size_t y = 0; y < 32; ++y) {
            for (size_t x = 0; x < 32; ++x) {
                templ(x, y) = input_img(64 + x, 64 + y);
            }
        }

        size_t rw = input_img.width() - 32 + 1;
        size_t rh = input_img.height() - 32 + 1;
        Image<float, 1> res_sqdiff(rw, rh);
        Image<float, 1> res_coeff(rw, rh);

        // Benchmark SQDIFF
        auto start_sq = std::chrono::high_resolution_clock::now();
        match_template(input_img.const_view(), templ.const_view(), res_sqdiff.view(), TemplateMetric::SQDIFF);
        auto end_sq = std::chrono::high_resolution_clock::now();

        // Benchmark CCOEFF_NORMED
        auto start_coeff = std::chrono::high_resolution_clock::now();
        match_template(input_img.const_view(), templ.const_view(), res_coeff.view(), TemplateMetric::CCOEFF_NORMED);
        auto end_coeff = std::chrono::high_resolution_clock::now();

        float min_v1 = 0.0f, max_v1 = 0.0f;
        size_t min_x1 = 0, min_y1 = 0, max_x1 = 0, max_y1 = 0;
        min_max_loc(res_sqdiff.const_view(), min_v1, max_v1, min_x1, min_y1, max_x1, max_y1);

        float min_v2 = 0.0f, max_v2 = 0.0f;
        size_t min_x2 = 0, min_y2 = 0, max_x2 = 0, max_y2 = 0;
        min_max_loc(res_coeff.const_view(), min_v2, max_v2, min_x2, min_y2, max_x2, max_y2);

        std::chrono::duration<double, std::milli> dur_sq = end_sq - start_sq;
        std::chrono::duration<double, std::milli> dur_coeff = end_coeff - start_coeff;

        std::cout << "[BENCHMARK] Template Matching SQDIFF: " << dur_sq.count() << " ms\n";
        std::cout << "[INFO] SQDIFF Best Match (min difference) at (" << min_x1 << ", " << min_y1 << ") with score " << min_v1 << "\n";
        std::cout << "[BENCHMARK] Template Matching CCOEFF_NORMED: " << dur_coeff.count() << " ms\n";
        std::cout << "[INFO] CCOEFF_NORMED Best Match (max similarity) at (" << max_x2 << ", " << max_y2 << ") with score " << max_v2 << "\n";

        // Save normalized response map for visualization
        Image<uint8_t, 1> vis_coeff(rw, rh);
        for (size_t y = 0; y < rh; ++y) {
            for (size_t x = 0; x < rw; ++x) {
                // map CCOEFF_NORMED from [-1.0, 1.0] to [0, 255]
                vis_coeff(x, y) = static_cast<uint8_t>(std::clamp((res_coeff(x, y) + 1.0f) * 127.5f, 0.0f, 255.0f));
            }
        }
        utils::save_image("output_template_response.png", vis_coeff.const_view());
    }

    // 13. Lucas-Kanade Sparse Optical Flow Demo
    if (!fast_kps.empty()) {
        // Create a copy of the input image translated by +2.0 pixels horizontally and +1.0 pixels vertically
        Image<uint8_t, 1> next_frame(input_img.width(), input_img.height(), 0);
        for (size_t y = 0; y < input_img.height(); ++y) {
            for (size_t x = 0; x < input_img.width(); ++x) {
                size_t sx = (x >= 2) ? x - 2 : 0;
                size_t sy = (y >= 1) ? y - 1 : 0;
                next_frame(x, y) = input_img(sx, sy);
            }
        }

        std::vector<Keypoint> tracked_kps;
        std::vector<uint8_t> status;

        auto start_lk = std::chrono::high_resolution_clock::now();
        calc_optical_flow_lk(input_img.const_view(), next_frame.const_view(), fast_kps, tracked_kps, status, 15, 20);
        auto end_lk = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_lk = end_lk - start_lk;

        size_t tracked_count = 0;
        double sum_dx = 0.0;
        double sum_dy = 0.0;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] == 1) {
                tracked_count++;
                sum_dx += (tracked_kps[i].x - fast_kps[i].x);
                sum_dy += (tracked_kps[i].y - fast_kps[i].y);
            }
        }

        std::cout << "[BENCHMARK] Lucas-Kanade Sparse Optical Flow: " << dur_lk.count() << " ms\n";
        std::cout << "[INFO] Tracked " << tracked_count << " / " << fast_kps.size() << " keypoints successfully.\n";
        if (tracked_count > 0) {
            std::cout << "[INFO] Average estimated displacement: (" << (sum_dx / tracked_count) << ", " << (sum_dy / tracked_count) << ") - expected: (2.0, 1.0)\n";
        }
    }

    // 14. Homography Estimation Demo
    {
        std::vector<Point2D> src = {
            Point2D(10.0, 10.0),
            Point2D(110.0, 10.0),
            Point2D(110.0, 110.0),
            Point2D(10.0, 110.0)
        };

        std::vector<Point2D> dst = {
            Point2D(12.0, 15.0),
            Point2D(115.0, 8.0),
            Point2D(108.0, 112.0),
            Point2D(15.0, 105.0)
        };

        auto start_h = std::chrono::high_resolution_clock::now();
        math::Matrix H = find_homography(src, dst);
        auto end_h = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_h = end_h - start_h;
        std::cout << "[BENCHMARK] Homography Estimation (DLT): " << dur_h.count() << " ms\n";
        std::cout << "[INFO] Estimated Homography Matrix H:\n";
        for (size_t r = 0; r < 3; ++r) {
            std::cout << "  [ " << H(r, 0) << ", " << H(r, 1) << ", " << H(r, 2) << " ]\n";
        }
    }

    // 15. Epipolar Geometry Demo
    {
        std::vector<Point2D> pts1 = {
            Point2D(10.0, 20.0), Point2D(150.0, 30.0), Point2D(50.0, 120.0), Point2D(30.0, 150.0),
            Point2D(120.0, 90.0), Point2D(90.0, 60.0), Point2D(80.0, 40.0), Point2D(40.0, 80.0)
        };

        // Skew-symmetric ground truth to generate points
        math::Matrix F_gt(3, 3, 0.0);
        F_gt(0, 1) = -0.2;  F_gt(0, 2) = 0.3;
        F_gt(1, 0) = 0.2;   F_gt(1, 2) = -0.1;
        F_gt(2, 0) = -0.3;  F_gt(2, 1) = 0.1;

        std::vector<Point2D> pts2(8);
        for (size_t i = 0; i < 8; ++i) {
            double l2_a = F_gt(0, 0) * pts1[i].x + F_gt(0, 1) * pts1[i].y + F_gt(0, 2);
            double l2_b = F_gt(1, 0) * pts1[i].x + F_gt(1, 1) * pts1[i].y + F_gt(1, 2);
            double l2_c = F_gt(2, 0) * pts1[i].x + F_gt(2, 1) * pts1[i].y + F_gt(2, 2);
            double y2 = 50.0;
            double x2 = (-l2_c - l2_b * y2) / l2_a;
            pts2[i] = Point2D(x2, y2);
        }

        auto start_f = std::chrono::high_resolution_clock::now();
        math::Matrix F = find_fundamental(pts1, pts2);
        auto end_f = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_f = end_f - start_f;
        std::cout << "[BENCHMARK] Fundamental Matrix Estimation (8-point): " << dur_f.count() << " ms\n";
        std::cout << "[INFO] Recovered Fundamental Matrix F:\n";
        for (size_t r = 0; r < 3; ++r) {
            std::cout << "  [ " << F(r, 0) << ", " << F(r, 1) << ", " << F(r, 2) << " ]\n";
        }

        // Mock camera intrinsic matrix K
        math::Matrix K = math::Matrix::identity(3);
        K(0, 0) = 500.0; K(1, 1) = 500.0;
        K(0, 2) = 250.0; K(1, 2) = 250.0;

        math::Matrix E = compute_essential(F, K, K);
        math::Matrix E_proj;
        project_essential(E, E_proj);
        std::cout << "[INFO] Calculated Essential Matrix E (projected to singular values s,s,0):\n";
        for (size_t r = 0; r < 3; ++r) {
            std::cout << "  [ " << E_proj(r, 0) << ", " << E_proj(r, 1) << ", " << E_proj(r, 2) << " ]\n";
        }
    }

    // 16. Stereo Vision & Triangulation Demo
    {
        // 16.1 Triangulation Demo
        math::Matrix P1(3, 4, 0.0);
        P1(0, 0) = 1.0; P1(1, 1) = 1.0; P1(2, 2) = 1.0;
        math::Matrix P2(3, 4, 0.0);
        P2(0, 0) = 1.0; P2(1, 1) = 1.0; P2(2, 2) = 1.0; P2(0, 3) = -1.0; // translated by 1 unit along X

        // 3D point at (2.0, 3.0, 5.0)
        std::vector<Point2D> pts1 = { Point2D(0.4, 0.6) };
        std::vector<Point2D> pts2 = { Point2D(0.2, 0.6) };
        std::vector<Point3D> pts3d;

        triangulate_points(P1, P2, pts1, pts2, pts3d);
        if (!pts3d.empty()) {
            std::cout << "[INFO] Triangulated 3D Point: (" << pts3d[0].x << ", " << pts3d[0].y << ", " << pts3d[0].z << ") - expected: (2.0, 3.0, 5.0)\n";
        }

        // 16.2 Disparity Block Matching Demo
        // Create synthetic rectified left/right images
        Image<uint8_t, 1> left_img(160, 160, 100);
        Image<uint8_t, 1> right_img(160, 160, 100);

        // Draw a block at (60, 60) to (100, 100) in left
        for (size_t y = 60; y < 100; ++y) {
            for (size_t x = 60; x < 100; ++x) {
                left_img(x, y) = 220;
            }
        }
        // Draw the block shifted right (or left in right view, e.g. x-d, d=6 pixels) at (54, 60) to (94, 100) in right
        for (size_t y = 60; y < 100; ++y) {
            for (size_t x = 54; x < 94; ++x) {
                right_img(x, y) = 220;
            }
        }

        Image<uint8_t, 1> disparity_map(160, 160);
        auto start_stereo = std::chrono::high_resolution_clock::now();
        compute_disparity_sad(left_img.const_view(), right_img.const_view(), disparity_map.view(), 0, 16, 9);
        auto end_stereo = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_stereo = end_stereo - start_stereo;
        std::cout << "[BENCHMARK] Stereo Disparity Map SAD: " << dur_stereo.count() << " ms\n";
        std::cout << "[INFO] Computed Disparity at block center: " << static_cast<int>(disparity_map(80, 80)) << " - expected: 6\n";

        // Map disparity values (0-15) to [0, 255] for visual verification
        Image<uint8_t, 1> vis_disp(160, 160);
        for (size_t y = 0; y < 160; ++y) {
            for (size_t x = 0; x < 160; ++x) {
                vis_disp(x, y) = disparity_map(x, y) * 16;
            }
        }
        utils::save_image("output_stereo_disparity.png", vis_disp.const_view());
    }

    // 17. Perspective-n-Point (PnP) Solver Demo
    {
        math::Matrix K(3, 3);
        K(0, 0) = 800.0; K(0, 1) = 0.0;   K(0, 2) = 320.0;
        K(1, 0) = 0.0;   K(1, 1) = 800.0; K(1, 2) = 240.0;
        K(2, 0) = 0.0;   K(2, 1) = 0.0;   K(2, 2) = 1.0;

        // Ground-truth rotation matrix R (rotation by 10 degrees around Y-axis)
        double angle = 10.0 * M_PI / 180.0;
        double c = std::cos(angle);
        double s = std::sin(angle);
        math::Matrix R_gt(3, 3, 0.0);
        R_gt(0, 0) = c;   R_gt(0, 2) = s;
        R_gt(1, 1) = 1.0;
        R_gt(2, 0) = -s;  R_gt(2, 2) = c;

        std::vector<double> t_gt = { 1.5, -2.0, 10.0 };

        std::vector<Point3D> object_pts = {
            Point3D(-2.0, -2.0, 0.0), Point3D(2.0, -2.0, 0.0), Point3D(2.0, 2.0, 0.0),
            Point3D(-2.0, 2.0, 0.0),  Point3D(0.0, -1.0, 1.0), Point3D(1.0, 1.0, -2.0)
        };

        std::vector<Point2D> image_pts(6);
        for (size_t i = 0; i < 6; ++i) {
            double xc = R_gt(0, 0) * object_pts[i].x + R_gt(0, 1) * object_pts[i].y + R_gt(0, 2) * object_pts[i].z + t_gt[0];
            double yc = R_gt(1, 0) * object_pts[i].x + R_gt(1, 1) * object_pts[i].y + R_gt(1, 2) * object_pts[i].z + t_gt[1];
            double zc = R_gt(2, 0) * object_pts[i].x + R_gt(2, 1) * object_pts[i].y + R_gt(2, 2) * object_pts[i].z + t_gt[2];
            image_pts[i].x = (K(0, 0) * xc + K(0, 2) * zc) / zc;
            image_pts[i].y = (K(1, 1) * yc + K(1, 2) * zc) / zc;
        }

        math::Matrix R_est;
        std::vector<double> t_est;

        auto start_pnp = std::chrono::high_resolution_clock::now();
        solve_pnp_dlt(object_pts, image_pts, K, R_est, t_est);
        auto end_pnp = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_pnp = end_pnp - start_pnp;
        std::cout << "[BENCHMARK] DLT PnP Solver: " << dur_pnp.count() << " ms\n";
        std::cout << "[INFO] Recovered Translation Vector t: [ " << t_est[0] << ", " << t_est[1] << ", " << t_est[2] << " ] - expected: [ 1.5, -2.0, 10.0 ]\n";
    }

    // 18. RANSAC Demo
    {
        math::Matrix H_gt(3, 3);
        H_gt(0, 0) = 1.1;   H_gt(0, 1) = 0.2;   H_gt(0, 2) = 15.0;
        H_gt(1, 0) = -0.1;  H_gt(1, 1) = 0.9;   H_gt(1, 2) = -8.0;
        H_gt(2, 0) = 0.0005;H_gt(2, 1) = -0.001;H_gt(2, 2) = 1.0;

        const size_t num_inliers = 15;
        const size_t num_outliers = 5;
        const size_t total_points = num_inliers + num_outliers;

        std::vector<Point2D> src(total_points);
        std::vector<Point2D> dst(total_points);

        // Generate clean inliers
        for (size_t i = 0; i < num_inliers; ++i) {
            src[i] = Point2D(i * 10.0 + 5.0, i * 8.0 + 12.0);
            double w = H_gt(2, 0) * src[i].x + H_gt(2, 1) * src[i].y + H_gt(2, 2);
            dst[i].x = (H_gt(0, 0) * src[i].x + H_gt(0, 1) * src[i].y + H_gt(0, 2)) / w;
            dst[i].y = (H_gt(1, 0) * src[i].x + H_gt(1, 1) * src[i].y + H_gt(1, 2)) / w;
        }

        // Generate outliers
        for (size_t i = num_inliers; i < total_points; ++i) {
            src[i] = Point2D(i * 10.0, i * 8.0);
            dst[i] = Point2D(-999.0, 999.0);
        }

        std::vector<uint8_t> inliers_mask;
        auto start_ransac = std::chrono::high_resolution_clock::now();
        math::Matrix H_est = find_homography_ransac(src, dst, inliers_mask, 3.0, 500);
        auto end_ransac = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_ransac = end_ransac - start_ransac;
        size_t recovered_inliers = 0;
        for (uint8_t val : inliers_mask) {
            if (val == 1) recovered_inliers++;
        }

        std::cout << "[BENCHMARK] find_homography_ransac: " << dur_ransac.count() << " ms\n";
        std::cout << "[INFO] RANSAC inlier count: " << recovered_inliers << " / " << total_points 
                  << " (expected inliers: " << num_inliers << ", outliers rejected: " << num_outliers << ")\n";
    }

    // 14. Camera Calibration (Zhang's Method) Demo
    {
        std::cout << "\n=== 14. Camera Calibration (Zhang's Method) ===\n";
        
        math::Matrix K_gt(3, 3, 0.0);
        K_gt(0, 0) = 800.0; K_gt(0, 1) = 0.5;   K_gt(0, 2) = 320.0;
        K_gt(1, 1) = 800.0; K_gt(1, 2) = 240.0;
        K_gt(2, 2) = 1.0;

        auto get_rotation = [](double angle_x, double angle_y) {
            double cx = std::cos(angle_x); double sx = std::sin(angle_x);
            double cy = std::cos(angle_y); double sy = std::sin(angle_y);
            math::Matrix Rx(3, 3, 0.0);
            Rx(0, 0) = 1.0; Rx(1, 1) = cx; Rx(1, 2) = -sx; Rx(2, 1) = sx; Rx(2, 2) = cx;
            math::Matrix Ry(3, 3, 0.0);
            Ry(0, 0) = cy; Ry(0, 2) = sy; Ry(1, 1) = 1.0; Ry(2, 0) = -sy; Ry(2, 2) = cy;
            return Ry * Rx;
        };

        struct Pose {
            double rx, ry, tx, ty, tz;
        };
        std::vector<Pose> poses = {
            { 0.1, -0.15, -0.2, 0.1, 1.5 },
            {-0.08, 0.12, 0.1, -0.1, 1.8 },
            { 0.15, 0.05, -0.15, 0.2, 2.0 },
            {-0.05, -0.1, 0.05, 0.05, 1.4 }
        };

        std::vector<math::Matrix> homographies;
        for (const auto& pose : poses) {
            math::Matrix R = get_rotation(pose.rx, pose.ry);
            math::Matrix H(3, 3);
            
            H(0, 0) = K_gt(0, 0) * R(0, 0) + K_gt(0, 1) * R(1, 0) + K_gt(0, 2) * R(2, 0);
            H(1, 0) = K_gt(1, 1) * R(1, 0) + K_gt(1, 2) * R(2, 0);
            H(2, 0) = R(2, 0);

            H(0, 1) = K_gt(0, 0) * R(0, 1) + K_gt(0, 1) * R(1, 1) + K_gt(0, 2) * R(2, 1);
            H(1, 1) = K_gt(1, 1) * R(1, 1) + K_gt(1, 2) * R(2, 1);
            H(2, 1) = R(2, 1);

            H(0, 2) = K_gt(0, 0) * pose.tx + K_gt(0, 1) * pose.ty + K_gt(0, 2) * pose.tz;
            H(1, 2) = K_gt(1, 1) * pose.ty + K_gt(1, 2) * pose.tz;
            H(2, 2) = pose.tz;

            homographies.push_back(H);
        }

        auto start_calib = std::chrono::high_resolution_clock::now();
        math::Matrix K_est = calibrate_camera(homographies);
        auto end_calib = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur_calib = end_calib - start_calib;
        std::cout << "[BENCHMARK] calibrate_camera: " << dur_calib.count() << " ms\n";
        std::cout << "[INFO] Ground Truth K:\n";
        std::cout << "  fx: " << K_gt(0, 0) << ", fy: " << K_gt(1, 1) << ", skew: " << K_gt(0, 1)
                  << ", u0: " << K_gt(0, 2) << ", v0: " << K_gt(1, 2) << "\n";
        std::cout << "[INFO] Estimated K:\n";
        std::cout << "  fx: " << K_est(0, 0) << ", fy: " << K_est(1, 1) << ", skew: " << K_est(0, 1)
                  << ", u0: " << K_est(0, 2) << ", v0: " << K_est(1, 2) << "\n";
    }

    // 19. Bundle Adjustment (Levenberg-Marquardt) Demo
    {
        math::Matrix K(3, 3);
        K(0, 0) = 800.0; K(0, 1) = 0.0;   K(0, 2) = 320.0;
        K(1, 0) = 0.0;   K(1, 1) = 800.0; K(1, 2) = 240.0;
        K(2, 0) = 0.0;   K(2, 1) = 0.0;   K(2, 2) = 1.0;

        math::Matrix R2_gt = euler_to_rotation(0.05, -0.04, 0.03);
        std::vector<double> t2_gt = { 1.0, -0.5, 8.0 };

        std::vector<Point3D> pts3d_gt = {
            Point3D(-1.5, -1.0, 5.0), Point3D(1.5, -1.0, 6.0),
            Point3D(1.5, 1.2, 5.5),   Point3D(-1.2, 1.0, 5.8)
        };

        const size_t N = pts3d_gt.size();
        std::vector<Point2D> obs1(N);
        std::vector<Point2D> obs2(N);

        math::Matrix R1 = math::Matrix::identity(3);
        std::vector<double> t1 = { 0.0, 0.0, 0.0 };

        for (size_t i = 0; i < N; ++i) {
            obs1[i] = project_point(pts3d_gt[i], K, R1, t1);
            obs2[i] = project_point(pts3d_gt[i], K, R2_gt, t2_gt);
        }

        std::vector<Point3D> pts3d_est = pts3d_gt;
        for (size_t i = 0; i < N; ++i) {
            pts3d_est[i].x += 0.15;
            pts3d_est[i].y -= 0.12;
            pts3d_est[i].z += 0.20;
        }

        math::Matrix R2_est = euler_to_rotation(0.07, -0.03, 0.04);
        std::vector<double> t2_est = { 1.1, -0.4, 8.2 };

        double initial_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            Point2D p1 = project_point(pts3d_est[i], K, R1, t1);
            Point2D p2 = project_point(pts3d_est[i], K, R2_est, t2_est);
            double e1x = p1.x - obs1[i].x; double e1y = p1.y - obs1[i].y;
            double e2x = p2.x - obs2[i].x; double e2y = p2.y - obs2[i].y;
            initial_err += (e1x*e1x + e1y*e1y + e2x*e2x + e2y*e2y);
        }

        auto start_ba = std::chrono::high_resolution_clock::now();
        bundle_adjustment(K, obs1, obs2, R2_est, t2_est, pts3d_est, 30);
        auto end_ba = std::chrono::high_resolution_clock::now();

        double final_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            Point2D p1 = project_point(pts3d_est[i], K, R1, t1);
            Point2D p2 = project_point(pts3d_est[i], K, R2_est, t2_est);
            double e1x = p1.x - obs1[i].x; double e1y = p1.y - obs1[i].y;
            double e2x = p2.x - obs2[i].x; double e2y = p2.y - obs2[i].y;
            final_err += (e1x*e1x + e1y*e1y + e2x*e2x + e2y*e2y);
        }

        std::chrono::duration<double, std::milli> dur_ba = end_ba - start_ba;
        std::cout << "[BENCHMARK] Levenberg-Marquardt Bundle Adjustment: " << dur_ba.count() << " ms\n";
        std::cout << "[INFO] Initial Reprojection Error: " << initial_err << " px^2\n";
        std::cout << "[INFO] Final Reprojection Error: " << final_err << " px^2\n";
    }

    std::cout << "[INFO] Processed images saved to directory.\n";
    return 0;
}

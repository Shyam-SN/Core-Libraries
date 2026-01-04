#include <visioncore/utils/image_io.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>

#ifdef VISIONCORE_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace visioncore::utils {

#ifndef VISIONCORE_HAS_OPENCV
// Helper to skip whitespace and comments in PPM/PGM
static void skip_comments(std::ifstream& f) {
    while (f >> std::ws && f.peek() == '#') {
        std::string dummy;
        std::getline(f, dummy);
    }
}
#endif

std::optional<Image<uint8_t, 1>> load_grayscale(const std::string& file_path) {
#ifdef VISIONCORE_HAS_OPENCV
    cv::Mat mat = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    if (mat.empty()) {
        return std::nullopt;
    }
    
    Image<uint8_t, 1> img(static_cast<size_t>(mat.cols), static_cast<size_t>(mat.rows));
    for (size_t y = 0; y < img.height(); ++y) {
        const uint8_t* row_ptr = mat.ptr<uint8_t>(static_cast<int>(y));
        std::copy(row_ptr, row_ptr + img.width(), &img(0, y, 0));
    }
    return img;
#else
    // Fallback PGM reader
    std::ifstream f(file_path, std::ios::binary);
    if (!f) return std::nullopt;

    std::string header;
    f >> header;
    if (header != "P5") return std::nullopt; // Not binary PGM

    skip_comments(f);
    int w = 0, h = 0, max_val = 0;
    f >> w >> h;
    skip_comments(f);
    f >> max_val;
    
    // Read the single trailing whitespace/newline after max_val
    f.get(); 

    if (w <= 0 || h <= 0 || max_val != 255) return std::nullopt;

    Image<uint8_t, 1> img(static_cast<size_t>(w), static_cast<size_t>(h));
    f.read(reinterpret_cast<char*>(img.data()), w * h);
    if (!f) return std::nullopt;

    return img;
#endif
}

std::optional<Image<uint8_t, 3>> load_rgb(const std::string& file_path) {
#ifdef VISIONCORE_HAS_OPENCV
    cv::Mat mat = cv::imread(file_path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        return std::nullopt;
    }
    
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    
    Image<uint8_t, 3> img(static_cast<size_t>(rgb_mat.cols), static_cast<size_t>(rgb_mat.rows));
    for (size_t y = 0; y < img.height(); ++y) {
        const uint8_t* row_ptr = rgb_mat.ptr<uint8_t>(static_cast<int>(y));
        std::copy(row_ptr, row_ptr + img.width() * 3, &img(0, y, 0));
    }
    return img;
#else
    // Fallback PPM reader
    std::ifstream f(file_path, std::ios::binary);
    if (!f) return std::nullopt;

    std::string header;
    f >> header;
    if (header != "P6") return std::nullopt; // Not binary PPM

    skip_comments(f);
    int w = 0, h = 0, max_val = 0;
    f >> w >> h;
    skip_comments(f);
    f >> max_val;
    
    f.get(); // Single trailing whitespace/newline

    if (w <= 0 || h <= 0 || max_val != 255) return std::nullopt;

    Image<uint8_t, 3> img(static_cast<size_t>(w), static_cast<size_t>(h));
    f.read(reinterpret_cast<char*>(img.data()), w * h * 3);
    if (!f) return std::nullopt;

    return img;
#endif
}

bool save_image(const std::string& file_path, const ImageView<const uint8_t, 1>& image) {
#ifdef VISIONCORE_HAS_OPENCV
    cv::Mat mat(static_cast<int>(image.height()), static_cast<int>(image.width()), CV_8UC1);
    for (size_t y = 0; y < image.height(); ++y) {
        uint8_t* row_ptr = mat.ptr<uint8_t>(static_cast<int>(y));
        const uint8_t* src_ptr = image.data() + y * image.stride_elements();
        std::copy(src_ptr, src_ptr + image.width(), row_ptr);
    }
    return cv::imwrite(file_path, mat);
#else
    // Fallback PGM writer
    std::string actual_path = file_path;
    if (actual_path.size() > 4 && actual_path.substr(actual_path.size() - 4) == ".png") {
        actual_path = actual_path.substr(0, actual_path.size() - 4) + ".pgm";
    }
    std::ofstream f(actual_path, std::ios::binary);
    if (!f) return false;

    f << "P5\n" << image.width() << " " << image.height() << "\n255\n";
    for (size_t y = 0; y < image.height(); ++y) {
        const uint8_t* src_ptr = image.data() + y * image.stride_elements();
        f.write(reinterpret_cast<const char*>(src_ptr), image.width());
    }
    return f.good();
#endif
}

bool save_image(const std::string& file_path, const ImageView<const uint8_t, 3>& image) {
#ifdef VISIONCORE_HAS_OPENCV
    cv::Mat rgb_mat(static_cast<int>(image.height()), static_cast<int>(image.width()), CV_8UC3);
    for (size_t y = 0; y < image.height(); ++y) {
        uint8_t* row_ptr = rgb_mat.ptr<uint8_t>(static_cast<int>(y));
        const uint8_t* src_ptr = image.data() + y * image.stride_elements();
        std::copy(src_ptr, src_ptr + image.width() * 3, row_ptr);
    }
    
    cv::Mat bgr_mat;
    cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
    return cv::imwrite(file_path, bgr_mat);
#else
    // Fallback PPM writer
    std::string actual_path = file_path;
    if (actual_path.size() > 4 && actual_path.substr(actual_path.size() - 4) == ".png") {
        actual_path = actual_path.substr(0, actual_path.size() - 4) + ".ppm";
    }
    std::ofstream f(actual_path, std::ios::binary);
    if (!f) return false;

    f << "P6\n" << image.width() << " " << image.height() << "\n255\n";
    for (size_t y = 0; y < image.height(); ++y) {
        const uint8_t* src_ptr = image.data() + y * image.stride_elements();
        f.write(reinterpret_cast<const char*>(src_ptr), image.width() * 3);
    }
    return f.good();
#endif
}

} // namespace visioncore::utils

#include <benchmark/benchmark.h>
#include <visioncore/core/image.hpp>
#include <visioncore/algorithms/filtering.hpp>
#include <visioncore/algorithms/convolution.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace visioncore;

// Helper to create a dummy test image
static Image<uint8_t, 1> create_test_image(size_t width, size_t height) {
    Image<uint8_t, 1> img(width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            img(x, y) = static_cast<uint8_t>((x + y) % 256);
        }
    }
    return img;
}

// 1. Benchmark Separable vs Direct 2D Convolution
static void BM_DirectConvolution(benchmark::State& state) {
    const size_t size = state.range(0);
    Image<uint8_t, 1> src = create_test_image(size, size);
    Image<uint8_t, 1> dst(size, size);
    
    // 9x9 kernel (81 elements)
    std::vector<double> kernel(81, 1.0 / 81.0);

    for (auto _ : state) {
        convolve2d(src.view(), dst.view(), kernel, 9, 9, BorderType::REPLICATE);
        benchmark::DoNotOptimize(dst.data());
    }
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_DirectConvolution)->Arg(256)->Arg(512);

static void BM_SeparableConvolution(benchmark::State& state) {
    const size_t size = state.range(0);
    Image<uint8_t, 1> src = create_test_image(size, size);
    Image<uint8_t, 1> dst(size, size);
    
    // 9x9 separable kernels (9 + 9 = 18 elements)
    std::vector<double> row_kernel(9, 1.0 / 9.0);
    std::vector<double> col_kernel(9, 1.0 / 9.0);

    for (auto _ : state) {
        convolve_separable(src.view(), dst.view(), row_kernel, col_kernel, BorderType::REPLICATE);
        benchmark::DoNotOptimize(dst.data());
    }
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_SeparableConvolution)->Arg(256)->Arg(512);

// 2. Benchmark VisionCore Gaussian Blur vs OpenCV
static void BM_VisionCore_GaussianBlur(benchmark::State& state) {
    const size_t size = state.range(0);
    Image<uint8_t, 1> src = create_test_image(size, size);
    Image<uint8_t, 1> dst(size, size);

    for (auto _ : state) {
        gaussian_blur(src.view(), dst.view(), 7, 1.5, BorderType::REPLICATE);
        benchmark::DoNotOptimize(dst.data());
    }
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_VisionCore_GaussianBlur)->Arg(512)->Arg(1024);

static void BM_OpenCV_GaussianBlur(benchmark::State& state) {
    const int size = static_cast<int>(state.range(0));
    
    // Create OpenCV Mat matching the test image
    cv::Mat src(size, size, CV_8UC1);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            src.at<uint8_t>(y, x) = static_cast<uint8_t>((x + y) % 256);
        }
    }
    cv::Mat dst;

    for (auto _ : state) {
        cv::GaussianBlur(src, dst, cv::Size(7, 7), 1.5, 1.5, cv::BORDER_REPLICATE);
        benchmark::DoNotOptimize(dst.data);
    }
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_OpenCV_GaussianBlur)->Arg(512)->Arg(1024);

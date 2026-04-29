# VisionCore: Modern C++20 Computer Vision & 3D Geometry Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![Build & Test Status](https://img.shields.io/badge/Tests-54%20Passed-success.svg)](#)

**VisionCore** is a high-performance, educationally clean, and mathematically rigorous computer vision library written in Modern C++20. It implements fundamental image processing pipelines, feature extraction, and 3D multi-view geometry algorithms from scratch, prioritizing zero-copy design, compile-time type safety, and SIMD alignment.

---

## 🚀 Key Architectural Principles

1.  **Zero-Copy Design & Views**: Designed with strict separation between owning memory containers (`Image`) and non-owning reference structures (`ImageView`). Sub-regions of interest (ROIs) are represented without memory duplication.
2.  **Modern C++20 Standard Library**: Utilizes standard attributes (`[[nodiscard]]`, `[[unlikely]]`), `std::span` for safe contiguous memory accesses, and `std::concepts` for compile-time template constraints.
3.  **SIMD-Ready Alignment**: Features a custom memory allocator aligned to 32-byte (AVX2) and 64-byte (AVX-512) boundaries, optimizing cache line efficiency and allowing compilers to perform auto-vectorization during convolution passes.
4.  **Mathematical & Physics-Grounded Rigor**: Every algorithm is implemented directly from its physical/trigonometric derivations (e.g. advective material derivatives for optical flow, Eikonal wave propagation for morphology, and spring-mass-damper network dynamics for bundle adjustment).

---

## 🛠️ Repository Structure

```
VisionCore/
├── include/
│   └── visioncore/
│       ├── core/               # Image/ImageView structures, AlignedAllocator, Concepts
│       ├── math/               # SVD solver, homogeneous linear systems, matrix utilities
│       └── algorithms/         # Convolution, filtering, features, 3D geometry, calibration
├── src/
│   └── utils/                  # OpenCV-backed I/O helper (optional dependency)
├── tests/                      # 54 Unit tests covering all algorithms via Google Test
├── benchmarks/                 # Performance comparison harness vs OpenCV using Google Benchmark
├── examples/                   # Interactive demonstrations
├── VisionCore_Mathematical_Foundations.md  # Detailed physics & math handbook
└── README.md                   # This documentation
```

---

## 📦 Implemented Modules

### 1. Core & Foundation
*   **Image Containers**: Owns aligned, contiguous 2D grid pixels using C++20 concepts (`std::is_arithmetic`).
*   **ImageView**: Non-owning sub-region views with read-only/write promotion safety.
*   **SVD Solver**: Custom Hestenes-Jacobi singular value decomposition for solving general rectangular linear systems and homogeneous equations $A \mathbf{x} = \mathbf{0}$.

### 2. Image Processing & Filters
*   **2D Spatial Convolution**: Supports Direct and Separable convolution with `ZERO`, `REPLICATE`, and `REFLECT` border padding.
*   **Filters**: Box and Gaussian blur kernels.
*   **Histograms**: Single-channel intensity histogram calculation and CDF-based equalization.
*   **Thresholding**: Adaptive local mean/Gaussian thresholds and Otsu's global variance minimization.
*   **Mathematical Morphology**: Structuring elements (rectangle/cross) supporting Dilation, Erosion, Opening, and Closing.

### 3. Feature Detection & Tracking
*   **Gradients**: Sobel, Prewitt, Scharr, and Laplacian derivative operators.
*   **Canny Edge Detection**: Multi-stage pipeline with non-maximum suppression, double thresholding, and stack-based hysteresis tracking.
*   **Corners**: FAST-9 (accelerated circle test) and Harris Corner detectors with local NMS.
*   **ORB Feature Descriptors**: oFAST intensity centroids for keypoint orientation, BRIEF descriptor matching, and rBRIEF (rotation-invariant rotated BRIEF).
*   **Feature Matching**: Hamming distance matcher and Lowe's Ratio Test filtering.
*   **Template Matching**: SQDIFF, CCORR, and Normalized CCOEFF template sliding search.
*   **Optical Flow**: Lucas-Kanade subpixel tracker using iterative Gauss-Newton optimization.

### 4. 3D Multi-View Geometry & SfM
*   **Direct Linear Transform (DLT)**: Normalized homography estimation for coplanar correspondences.
*   **Zhang's Calibration**: Multi-view planar calibration solving for the absolute conic matrix $B$ using SVD constraints to analytically extract intrinsics $K$ and extrinsics $[R \mid \mathbf{t}]$.
*   **Epipolar Geometry**: Fundamental matrix ($F$) 8-point solver with rank-2 singularity projection, and Essential matrix ($E = K'^T F K$) decomposition into 4 relative pose options.
*   **Triangulation**: Linear ray intersection solver.
*   **Stereo Disparity**: Scanline-aligned SAD block matcher mapping disparities to physical depth coordinates.
*   **PnP (Perspective-n-Point)**: DLT projection solver and rigid $SO(3)$ pose orthonormalization.
*   **RANSAC**: Robust randomized subset sampling for homography fitting under high-outlier ratios.
*   **Bundle Adjustment**: Joint camera-pose/3D-point optimization utilizing a Levenberg-Marquardt optimizer and SVD solver.

---

## ⚡ Building & Getting Started

### Prerequisites
*   **Operating System**: macOS (Apple Silicon / Intel) or Linux (Ubuntu, Debian, Fedora, Arch, etc.).
*   **Compiler**: GCC 10+ or Clang 12+ / Apple Clang (Full C++20 support required).
*   **Build System**: CMake 3.16+
*   **Optional Dependencies**:
    *   OpenCV (Only utilized for loading/saving PNG/JPG images in image I/O helpers and example demos).

On macOS (via Homebrew):
```bash
brew install cmake opencv
```

### Automated Quick Start & Testing

Run the cross-platform `test.sh` script located inside `VisionCore`:

```bash
cd VisionCore
./test.sh
```

### Manual Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/VisionCore.git
cd VisionCore

# Create a build directory
mkdir -p build && cd build

# Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile targets (Library, Tests, Examples, Benchmarks)
# Uses sysctl on macOS or nproc on Linux for parallel build
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
```

### Running Tests
To run the full suite of unit tests:

```bash
./build/tests/visioncore_tests
```

### Running Benchmarks
To compare the performance of VisionCore's aligned memory convolution against OpenCV:

```bash
./build/benchmarks/visioncore_benchmarks
```

---

## 📚 Theory & Physics HandBook

For a deep, graduate-level dive into the mathematical derivations and physical equations governing each algorithm (e.g. how Gaussian smoothing represents physical thermal diffusion, or how the Lucas-Kanade optical flow formulation is derived from the advective material derivative in fluid mechanics), see the accompanying reference manual:

👉 **[VisionCore Mathematical & Physics Foundations](VisionCore_Mathematical_Foundations.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

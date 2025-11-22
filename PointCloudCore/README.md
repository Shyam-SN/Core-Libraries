# PointCloudCore: Research-Grade 3D Perception & Geometric Deep Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/credential)
[![Build & Test Status](https://img.shields.io/badge/Tests-Passed-success.svg)](#)

**PointCloudCore** is a high-performance 3D perception and geometric deep learning framework written from scratch in Modern C++20. It bridges classical 3D vision geometry (PCA Surface Normal Estimation, 33D FPFH Descriptors, Farthest Point Sampling, RANSAC Global Registration, SVD ICP Alignment) and state-of-the-art 3D Deep Learning architectures (PointNet++ Set Abstraction & Feature Propagation layers, Point Transformer Vector Attention blocks, PointNeXt inverted residual bottleneck blocks, and Chamfer Distance loss).

---

## 🛠️ Repository Structure

```
PointCloudCore/
├── include/
│   └── pointcloudcore/
│       ├── core/               # PointCloud container, Tensor representations, C++20 Concepts
│       ├── ops/                # PCA Normals, FPS Sampling, k-NN & Ball Query, FPFH, Metrics
│       ├── registration/       # SVD ICP & FPFH RANSAC Global Registration
│       ├── nn/                 # PointNet++, Point Transformer, PointNeXt modules
│       └── utils/              # High-resolution Timers
├── src/                        # Core C++20 implementation modules
├── tests/                      # Unit test suite powered by GoogleTest
├── benchmarks/                 # Performance benchmark suite powered by Google Benchmark
├── examples/                   # Interactive perception & deep learning demo
├── PointCloudCore_Mathematical_Foundations.md  # Math & Physics Derivation Handbook
├── CMakeLists.txt              # Cross-platform CMake build configuration
└── test.sh                     # Automated build, test, and benchmark script
```

---

## ⚡ Building & Getting Started

### Prerequisites
*   **Operating System**: macOS (Apple Silicon ARM64 / Intel) or Linux (Ubuntu, Debian, Fedora, Arch, etc.).
*   **Compiler**: GCC 10+ or Clang 12+ / Apple Clang (Full C++20 standard support).
*   **Build System**: CMake 3.16+

### Quick Start & Automated Test Suite

Run the cross-platform `test.sh` script inside `PointCloudCore`:

```bash
cd PointCloudCore
./test.sh
```

### Manual Build Instructions

```bash
cd PointCloudCore
mkdir -p build && cd build

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile targets (Library, Tests, Examples, Benchmarks)
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run Unit Tests
./tests/pointcloudcore_tests

# Run Example Demo
./examples/demo_registration_segmentation

# Run Benchmarks
./benchmarks/pointcloudcore_benchmarks
```

---

## 📚 Mathematical & Physics Foundations Handbook

For exhaustive mathematical derivations and physical intuitions (PCA surface normal covariance Eigen-decomposition, FPFH Darboux frame angles, Farthest Point Sampling greedy bounds, PointNet++ Set Abstraction, Point Transformer Vector Self-Attention equations, and Chamfer Distance calculus), see:

👉 **[PointCloudCore Mathematical & Physics Foundations](PointCloudCore_Mathematical_Foundations.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

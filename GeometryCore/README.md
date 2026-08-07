# GeometryCore: Research-Grade 3D Vision & Geometry Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/credential)
[![Build & Test Status](https://img.shields.io/badge/Tests-Passed-success.svg)](#)

**GeometryCore** is a high-performance, educationally clean, and mathematically rigorous 3D computer vision and geometry processing library written from scratch in Modern C++20. It implements 3D linear algebra primitives (Vectors, Matrices, Quaternions, $SO(3)$ / $SE(3)$ Lie Groups & Lie Algebras), Pinhole Camera Models with distortion, Multi-View Geometry (Triangulation, Epipolar Geometry, Essential/Fundamental matrix, PnP), Point Cloud processing & SVD-based ICP registration, Spatial data structures (KD-Tree, Octree, VoxelGrid, TSDF Volumetric Fusion), and Iso-surface Mesh Generation via Marching Cubes.

---

## 🛠️ Repository Structure

```
GeometryCore/
├── include/
│   └── geometrycore/
│       ├── math/               # Vector3, Matrix3, Matrix4, Quaternion, Lie Groups SO(3)/SE(3)
│       ├── camera/             # Pinhole Camera Model with Distortion & Pixel Projection
│       ├── geometry/           # Triangulation, Epipolar Geometry, PnP Solver
│       ├── pointcloud/         # PointCloud, SVD ICP Registration, 3D KD-Tree
│       ├── spatial/            # Octree Partitioning, VoxelGrid Downsampling, TSDF Volume
│       ├── mesh/               # Triangle Mesh, Marching Cubes Iso-Surface Extraction
│       └── utils/              # High-resolution Timers
├── src/                        # Core C++20 implementation modules
├── tests/                      # Unit test suite powered by GoogleTest
├── benchmarks/                 # Performance benchmark suite powered by Google Benchmark
├── examples/                   # Interactive ICP & Marching Cubes demonstration
├── GeometryCore_Mathematical_Foundations.md  # Math & Physics Derivation Handbook
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

Run the cross-platform `test.sh` script inside `GeometryCore`:

```bash
cd GeometryCore
./test.sh
```

### Manual Build Instructions

```bash
cd GeometryCore
mkdir -p build && cd build

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile targets (Library, Tests, Examples, Benchmarks)
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run Unit Tests
./tests/geometrycore_tests

# Run Example Demo
./examples/demo_icp_marching_cubes

# Run Benchmarks
./benchmarks/geometrycore_benchmarks
```

---

## 📚 Mathematical & Physics Foundations Handbook

For exhaustive mathematical derivations and physical intuitions (Quaternions, Lie group exponential/logarithmic maps, Pinhole projection calculus, SVD Kabsch-Umeyama closed-form registration, KD-Tree spatial bounds, TSDF fusion, and Marching Cubes lookup tables), see:

👉 **[GeometryCore Mathematical & Physics Foundations](GeometryCore_Mathematical_Foundations.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

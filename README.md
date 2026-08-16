# OmniCore (Core Libraries) 🧠

Welcome to **OmniCore**—a unified monorepo housing the foundational C++ libraries and algorithms that power high-performance Artificial Intelligence, Computer Vision, and 3D Geometry systems.

This repository consolidates several robust, highly optimized, and decoupled subsystems into a single ecosystem designed for extreme performance, scalability, and modularity.

---

## 📚 Libraries Included

### 1. `DeepLearningCore`
A custom, lightweight neural network and tensor operations library built from scratch in C++.
- Automatic Differentiation (Autograd)
- highly-optimized matrix multiplication (SIMD/BLAS integration)
- Custom optimizers (Adam, SGD) and loss functions.
- Designed for edge devices where deploying massive frameworks like PyTorch or TensorFlow is not viable.

### 2. `GeometryCore`
A high-precision 3D mathematical library.
- Quaternions, Euler angles, and affine transformations.
- Ray-casting, intersection testing, and spatial partitioning (Octrees, KD-Trees).
- Bounding Volume Hierarchies (BVH) for accelerated spatial queries.

### 3. `VisionCore`
Computer Vision primitives and image processing pipelines.
- Multi-threaded image convolution, filtering, and edge detection (Sobel, Canny).
- Feature extraction and matching algorithms.
- Camera calibration, intrinsic/extrinsic matrix handling, and stereo vision depth estimation.

### 4. `PointCloudCore`
Optimized algorithms for handling massive 3D point cloud datasets (millions of points).
- Voxel grid downsampling and Statistical Outlier Removal (SOR).
- Iterative Closest Point (ICP) for point cloud registration.
- Normal estimation and surface reconstruction algorithms.

### 5. `GraphCore`
High-performance graph theory and traversal algorithms.
- Custom adjacency list/matrix implementations optimized for cache locality.
- A*, Dijkstra, and Bellman-Ford shortest path routing.
- Strongly Connected Components (Tarjan's algorithm) and Cycle detection.

---

## 🛠️ Setup & Installation

### Prerequisites
- A modern C++17 or C++20 compatible compiler (GCC, Clang, or MSVC).
- **CMake** (v3.15+)

### Build Instructions
Since these are foundational libraries, they are built as static or shared libraries to be linked by downstream applications (like `CodeAtlas` or `FedEdge-Async`).

```bash
# Clone the repository
git clone git@github.com:Shyam-SN/Core-Libraries.git
cd Core-Libraries

# Create a build directory
mkdir build && cd build

# Configure and compile all core modules
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 🔬 Architecture Philosophy
- **Zero-Cost Abstractions**: Modern C++ features are used heavily but carefully to ensure there is no runtime overhead.
- **Header-Only when Possible**: heavily templated code (like linear algebra matrices) is provided as header-only to allow aggressive compiler inlining.
- **No Heavy Dependencies**: We strictly avoid bloating these libraries with massive third-party dependencies. They are self-contained, mathematically rigorous, and deterministic.

# DeepLearningCore: Research-Grade C++20 Deep Learning Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![Build & Test Status](https://img.shields.io/badge/Tests-Passed-success.svg)](#)

**DeepLearningCore** is a high-performance, educationally clean, and mathematically rigorous deep learning framework written from scratch in Modern C++20. It implements multi-dimensional strided Tensors, reverse-mode Automatic Differentiation (Autograd), Neural Network Modules (Linear, Conv2D, BatchNorm, Dropout, Residual Blocks, Transformer primitives), Optimizers (SGD, Adam, RMSProp), Loss Functions, DataLoader, Model Checkpoint Serialization, and Inference Engine.

---

## 🛠️ Repository Structure

```
DeepLearningCore/
├── include/
│   └── deeplearningcore/
│       ├── core/               # Tensor, Storage, Shape, Autograd, Concepts
│       ├── ops/                # Arithmetic, GEMM, Activations (ReLU, Sigmoid, Softmax, GELU)
│       ├── nn/                 # Module, Linear, Conv2D, Pooling, BatchNorm, Dropout, Residual, Transformer
│       ├── optim/              # SGD, Adam, RMSProp Optimizers
│       ├── losses/             # MSE, CrossEntropy, BCE, L1 Losses
│       ├── data/               # Dataset, DataLoader
│       └── utils/              # Model Checkpoint Serialization, High-res Timers
├── src/                        # Core C++20 implementation modules
├── tests/                      # Unit test suite powered by GoogleTest
├── benchmarks/                 # Performance benchmark suite powered by Google Benchmark
├── examples/                   # Interactive training demonstration
├── DeepLearningCore_Mathematical_Foundations.md  # Theory & Derivation Handbook
├── CMakeLists.txt              # Cross-platform CMake build configuration
└── test.sh                     # Automated build, test, and benchmark script
```

---

## ⚡ Building & Getting Started

### Prerequisites
*   **Operating System**: macOS (Apple Silicon ARM64 / Intel) or Linux (Ubuntu, Debian, Fedora, Arch, etc.).
*   **Compiler**: GCC 10+ or Clang 12+ / Apple Clang (C++20 standard support).
*   **Build System**: CMake 3.16+

### Quick Start & Automated Test Suite

Run the cross-platform `test.sh` script inside `DeepLearningCore`:

```bash
cd DeepLearningCore
./test.sh
```

### Manual Build Instructions

```bash
cd DeepLearningCore
mkdir -p build && cd build

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile targets (Library, Tests, Examples, Benchmarks)
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run Unit Tests
./tests/deeplearningcore_tests

# Run Example Training Demo
./examples/demo_mlp_mnist

# Run Benchmarks
./benchmarks/deeplearningcore_benchmarks
```

---

## 📚 Mathematical Foundations Handbook

For detailed mathematical derivations (chain rule, matrix calculus, autograd computational DAG, backpropagation through convolution, batch normalization, and transformer attention mechanisms), see:

👉 **[DeepLearningCore Mathematical Foundations](DeepLearningCore_Mathematical_Foundations.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

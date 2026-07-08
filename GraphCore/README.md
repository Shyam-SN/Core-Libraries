# GraphCore: Research-Grade Graph Processing Framework & Qt Interactive Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/credential)
[![Build & Test Status](https://img.shields.io/badge/Tests-Passed-success.svg)](#)

**GraphCore** is a high-performance graph algorithms framework and desktop visualization studio written from scratch in Modern C++20. It features high-performance graph representations (Adjacency List, Adjacency Matrix, CSR, CSC, Edge List), random/scale-free graph generators, force-directed 2D layout physics, 30+ graph algorithms (Dijkstra, A*, Tarjan SCC, Kruskal/Prim MST, Dinic Max Flow, Hopcroft-Karp, Bron-Kerbosch, Louvain Community Detection, PageRank Centrality), and a Qt 6 / Qt 5 desktop studio complete with an **Interactive Onboarding Guided Tour**.

---

## 🛠️ Repository Structure

```
GraphCore/
├── include/
│   └── graphcore/
│       ├── graph/              # Graph, CSRGraph, C++20 Concepts
│       ├── algorithms/         # Traversals, Shortest Paths, DAG, MST, Flow, Matching, Cliques, Community, Centrality
│       ├── generators/         # Erdos-Renyi, Grid, Scale-Free, Random DAG Generators
│       ├── layouts/            # Fruchterman-Reingold 2D Force-Directed Layout
│       ├── visualization/      # Step-by-Step Animation Engine
│       └── utils/              # Union-Find (DSU) & Timers
├── src/                        # Core C++20 implementation modules
├── qt_app/                     # Qt Desktop Visualizer & Interactive Tour Guide Wizard
├── tests/                      # Unit test suite powered by GoogleTest
├── benchmarks/                 # Performance benchmark suite powered by Google Benchmark
├── examples/                   # Interactive CLI graph studio demonstration
├── GraphCore_Mathematical_Foundations.md  # Proofs of Correctness & Math Foundations
├── CMakeLists.txt              # Cross-platform CMake build configuration
└── test.sh                     # Automated build, test, and benchmark script
```

---

## ⚡ Building & Getting Started

### Prerequisites
*   **Operating System**: macOS (Apple Silicon ARM64 / Intel) or Linux (Ubuntu, Debian, Fedora, Arch, etc.).
*   **Compiler**: GCC 10+ or Clang 12+ / Apple Clang (Full C++20 standard support).
*   **Build System**: CMake 3.16+
*   **Optional GUI**: Qt 6 or Qt 5 Widgets (for Qt Interactive Studio Desktop App)

### Quick Start & Automated Test Suite

Run the cross-platform `test.sh` script inside `GraphCore`:

```bash
cd GraphCore
./test.sh
```

### Manual Build Instructions

```bash
cd GraphCore
mkdir -p build && cd build

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile targets (Library, Tests, Examples, Benchmarks, Qt Studio App)
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run Unit Tests
./tests/graphcore_tests

# Run Example Demo
./examples/demo_graph_studio

# Run Benchmarks
./benchmarks/graphcore_benchmarks

# Launch Qt Interactive Studio & Tour Guide (if Qt is available)
./qt_app/graphcore_qt_app
```

---

## 🎓 Qt Interactive Onboarding Tour Guide

When launching `graphcore_qt_app`, an interactive **Tour Guide Wizard Dialog** automatically pops up to introduce the UI layout:
1. **Interactive Graph Canvas**: Click to spawn nodes, drag to construct weighted edges, pan/zoom.
2. **Algorithm Controls**: Select from Dijkstra, A*, Tarjan SCC, Dinic Flow, Louvain, PageRank.
3. **Step Animation Engine**: Play, Pause, Step Next, Step Prev through queue/stack state steps.
4. **Force Layout & Benchmarks**: Physics repulsion simulation and real-time complexity panel.

---

## 📚 Mathematical Foundations Handbook

For detailed proofs of correctness and mathematical derivations (Dijkstra non-negative edge optimality proof, A* admissible heuristic consistency, Tarjan SCC low-link invariants, Dinic blocking flow analysis, Hopcroft-Karp augmenting paths, Bron-Kerbosch pivot optimization, PageRank power iteration convergence, and Louvain modularity $Q$), see:

👉 **[GraphCore Mathematical Foundations](GraphCore_Mathematical_Foundations.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

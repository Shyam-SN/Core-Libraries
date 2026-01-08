#!/bin/bash
set -e

# Change directory to the script's directory (VisionCore project root)
cd "$(dirname "$0")"

echo "=== 1. Configuring CMake (Release Mode) ==="
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

echo "=== 2. Building VisionCore ==="
# Detect available CPU cores portably across macOS, Linux, and BSD
if command -v nproc >/dev/null 2>&1; then
    NPROC=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NPROC=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NPROC=4
fi

cmake --build build -j"$NPROC"

echo "=== 3. Running Unit Tests ==="
./build/tests/visioncore_tests

echo "=== 4. Running Example Demo ==="
# Run the demo in the build/examples directory so output images are generated there
cd build/examples
./filter_demo
cd ../..

echo "=== 5. Running Benchmarks ==="
if [ -f ./build/benchmarks/visioncore_benchmarks ]; then
    ./build/benchmarks/visioncore_benchmarks
else
    echo "Benchmarks binary not built (OpenCV missing or benchmarks disabled)."
fi

echo "=== Build, Test, and Benchmarking Completed Successfully! ==="

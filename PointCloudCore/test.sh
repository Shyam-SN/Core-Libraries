#!/bin/bash
set -e

# Change directory to the script's directory (PointCloudCore project root)
cd "$(dirname "$0")"

echo "=== 1. Configuring CMake (Release Mode) ==="
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

echo "=== 2. Building PointCloudCore ==="
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
./build/tests/pointcloudcore_tests

echo "=== 4. Running Example Demo ==="
./build/examples/demo_registration_segmentation

echo "=== 5. Running Benchmarks ==="
./build/benchmarks/pointcloudcore_benchmarks

echo "=== PointCloudCore Build, Test, and Benchmarking Completed Successfully! ==="

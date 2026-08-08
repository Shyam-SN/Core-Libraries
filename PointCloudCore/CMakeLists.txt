cmake_minimum_required(VERSION 3.16)
project(PointCloudCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warning and optimization flags
include(CheckCXXCompilerFlag)
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)")
        check_cxx_compiler_flag("-mcpu=native" COMPILER_SUPPORTS_MCPU_NATIVE)
        if(COMPILER_SUPPORTS_MCPU_NATIVE)
            add_compile_options(-mcpu=native)
        endif()
    else()
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

# Include directories
include_directories(include)

# Create library
add_library(pointcloudcore STATIC
    src/core/point_cloud.cpp
    src/core/tensor.cpp
    src/ops/pca_normals.cpp
    src/ops/sampling.cpp
    src/ops/neighborhood.cpp
    src/ops/fpfh_descriptor.cpp
    src/ops/metrics.cpp
    src/registration/icp.cpp
    src/registration/ransac_registration.cpp
    src/nn/pointnet_plusplus.cpp
    src/nn/point_transformer.cpp
    src/nn/pointnext.cpp
)

target_include_directories(pointcloudcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable Testing
enable_testing()
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch Google Benchmark
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Add Subdirectories
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)

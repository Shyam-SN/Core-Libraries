cmake_minimum_required(VERSION 3.16)
project(VisionCore VERSION 0.1.0 LANGUAGES CXX)

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

# Find OpenCV (only for image I/O and comparison checks, as per blueprint)
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
endif()
find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    add_compile_definitions(VISIONCORE_HAS_OPENCV)
else()
    message(WARNING "OpenCV not found. Building with fallback PPM/PGM image I/O.")
endif()

# Include directories
include_directories(include)

# Create library
add_library(visioncore STATIC
    src/utils/image_io.cpp
)
target_include_directories(visioncore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(OpenCV_FOUND)
    target_link_libraries(visioncore PUBLIC ${OpenCV_LIBS})
endif()

# Enable Testing
enable_testing()
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch Google Benchmark
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Add subdirectories
add_subdirectory(tests)
if(OpenCV_FOUND)
    add_subdirectory(benchmarks)
endif()
add_subdirectory(examples)


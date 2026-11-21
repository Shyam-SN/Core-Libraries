cmake_minimum_required(VERSION 3.16)
project(OmniCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_subdirectory(VisionCore)
add_subdirectory(GeometryCore)
add_subdirectory(PointCloudCore)
add_subdirectory(DeepLearningCore)
add_subdirectory(GraphCore)

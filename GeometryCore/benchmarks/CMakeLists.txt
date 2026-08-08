add_executable(geometrycore_benchmarks bench_icp_kdtree.cpp)
target_link_libraries(geometrycore_benchmarks PRIVATE
    geometrycore
    benchmark::benchmark_main
)

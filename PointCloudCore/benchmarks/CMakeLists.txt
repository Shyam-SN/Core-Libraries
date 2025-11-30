add_executable(pointcloudcore_benchmarks bench_sampling_fpfh.cpp)
target_link_libraries(pointcloudcore_benchmarks PRIVATE
    pointcloudcore
    benchmark::benchmark_main
)

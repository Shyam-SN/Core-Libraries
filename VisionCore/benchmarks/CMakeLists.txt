add_executable(visioncore_benchmarks
    bench_filtering.cpp
)

target_link_libraries(visioncore_benchmarks PRIVATE
    visioncore
    benchmark::benchmark
    benchmark::benchmark_main
)

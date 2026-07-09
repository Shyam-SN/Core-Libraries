add_executable(graphcore_benchmarks bench_algorithms.cpp)
target_link_libraries(graphcore_benchmarks PRIVATE
    graphcore
    benchmark::benchmark_main
)

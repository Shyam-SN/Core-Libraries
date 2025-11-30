add_executable(deeplearningcore_benchmarks bench_tensor_ops.cpp)
target_link_libraries(deeplearningcore_benchmarks PRIVATE
    deeplearningcore
    benchmark::benchmark_main
)

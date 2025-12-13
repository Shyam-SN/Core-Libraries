#include <benchmark/benchmark.h>
#include <deeplearningcore/deeplearningcore.hpp>

using namespace deeplearningcore;

static void BM_TensorMatmul(benchmark::State& state) {
    size_t dim = state.range(0);
    Tensor a = Tensor::rand(Shape{dim, dim}, -1.0f, 1.0f);
    Tensor b = Tensor::rand(Shape{dim, dim}, -1.0f, 1.0f);

    for (auto _ : state) {
        Tensor c = a.matmul(b);
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_TensorMatmul)->Arg(32)->Arg(64)->Arg(128);

static void BM_TensorAutogradBackward(benchmark::State& state) {
    size_t dim = state.range(0);
    for (auto _ : state) {
        Tensor a = Tensor::rand(Shape{dim, dim}, -1.0f, 1.0f, true);
        Tensor b = Tensor::rand(Shape{dim, dim}, -1.0f, 1.0f, true);
        Tensor c = a.matmul(b).relu().sum();
        c.backward();
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_TensorAutogradBackward)->Arg(32)->Arg(64);

BENCHMARK_MAIN();

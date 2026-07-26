#include <benchmark/benchmark.h>
#include <graphcore/graphcore.hpp>

using namespace graphcore;

static void BM_DijkstraShortestPath(benchmark::State& state) {
    size_t num_v = state.range(0);
    Graph g = GraphGenerators::erdos_renyi(num_v, 0.1, false);

    for (auto _ : state) {
        auto res = ShortestPath::dijkstra(g, 0);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DijkstraShortestPath)->Arg(100)->Arg(500);

static void BM_PageRankCentrality(benchmark::State& state) {
    size_t num_v = state.range(0);
    Graph g = GraphGenerators::erdos_renyi(num_v, 0.1, true);

    for (auto _ : state) {
        auto pr = Centrality::pagerank(g, 0.85, 20);
        benchmark::DoNotOptimize(pr);
    }
}
BENCHMARK(BM_PageRankCentrality)->Arg(100)->Arg(500);

BENCHMARK_MAIN();

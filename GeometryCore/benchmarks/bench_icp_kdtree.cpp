#include <benchmark/benchmark.h>
#include <geometrycore/geometrycore.hpp>

using namespace geometrycore;

static void BM_KDTreeBuild(benchmark::State& state) {
    size_t num_pts = state.range(0);
    std::vector<Vector3> pts;
    pts.reserve(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
        pts.push_back({static_cast<double>(i), static_cast<double>(i * 2), static_cast<double>(i * 3)});
    }

    for (auto _ : state) {
        KDTree kdtree(pts);
        benchmark::DoNotOptimize(kdtree);
    }
}
BENCHMARK(BM_KDTreeBuild)->Arg(1000)->Arg(5000);

static void BM_KDTreeNearestNeighbor(benchmark::State& state) {
    size_t num_pts = state.range(0);
    std::vector<Vector3> pts;
    pts.reserve(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
        pts.push_back({static_cast<double>(i % 100), static_cast<double>((i / 100) % 100), static_cast<double>(i / 10000)});
    }
    KDTree kdtree(pts);
    Vector3 query(50.5, 50.5, 0.5);

    for (auto _ : state) {
        double dist_sq = 0.0;
        size_t idx = kdtree.nearest_neighbor(query, dist_sq);
        benchmark::DoNotOptimize(idx);
    }
}
BENCHMARK(BM_KDTreeNearestNeighbor)->Arg(1000)->Arg(5000);

BENCHMARK_MAIN();

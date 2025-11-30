#include <benchmark/benchmark.h>
#include <pointcloudcore/pointcloudcore.hpp>

using namespace pointcloudcore;

static void BM_FarthestPointSampling(benchmark::State& state) {
    size_t num_pts = state.range(0);
    PointCloud cloud;
    for (size_t i = 0; i < num_pts; ++i) {
        cloud.points.push_back({static_cast<double>(i), static_cast<double>(i * 2), static_cast<double>(i * 3)});
    }

    for (auto _ : state) {
        PointCloud sampled = Sampling::farthest_point_sampling(cloud, num_pts / 4);
        benchmark::DoNotOptimize(sampled);
    }
}
BENCHMARK(BM_FarthestPointSampling)->Arg(500)->Arg(1000);

static void BM_PCANormalsEstimation(benchmark::State& state) {
    size_t num_pts = state.range(0);
    PointCloud cloud;
    for (size_t i = 0; i < num_pts; ++i) {
        cloud.points.push_back({static_cast<double>(i % 50), static_cast<double>((i / 50) % 50), 0.0});
    }

    for (auto _ : state) {
        PCANormals::estimate_normals(cloud, 15);
        benchmark::DoNotOptimize(cloud);
    }
}
BENCHMARK(BM_PCANormalsEstimation)->Arg(500)->Arg(1000);

BENCHMARK_MAIN();

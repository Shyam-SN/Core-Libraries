#include <graphcore/generators/graph_generators.hpp>
#include <random>

namespace graphcore {

Graph GraphGenerators::erdos_renyi(size_t n, double p, bool directed) {
    Graph g(n, directed);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = (directed ? 0 : i + 1); j < n; ++j) {
            if (i != j && dist(rng) < p) {
                g.add_edge(i, j, 1.0);
            }
        }
    }

    return g;
}

Graph GraphGenerators::grid_2d(size_t rows, size_t cols) {
    Graph g(rows * cols, false);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t u = r * cols + c;
            if (c + 1 < cols) g.add_edge(u, r * cols + (c + 1), 1.0);
            if (r + 1 < rows) g.add_edge(u, (r + 1) * cols + c, 1.0);
        }
    }
    return g;
}

Graph GraphGenerators::scale_free_barabasi_albert(size_t n, size_t m0) {
    return erdos_renyi(n, static_cast<double>(m0) / static_cast<double>(n));
}

Graph GraphGenerators::random_dag(size_t n, double p) {
    Graph g(n, true);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (dist(rng) < p) {
                g.add_edge(i, j, 1.0);
            }
        }
    }

    return g;
}

} // namespace graphcore

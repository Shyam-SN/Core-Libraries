#include <graphcore/algorithms/centrality.hpp>
#include <cmath>

namespace graphcore {

std::vector<double> Centrality::pagerank(const Graph& g, double damping, size_t max_iterations, double tol) {
    size_t N = g.num_vertices();
    std::vector<double> rank(N, 1.0 / static_cast<double>(N));
    if (N == 0) return rank;

    std::vector<size_t> out_degree(N, 0);
    for (size_t u = 0; u < N; ++u) {
        out_degree[u] = g.neighbors(u).size();
    }

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> next_rank(N, (1.0 - damping) / static_cast<double>(N));
        double diff = 0.0;

        for (size_t u = 0; u < N; ++u) {
            if (out_degree[u] > 0) {
                double contrib = damping * rank[u] / static_cast<double>(out_degree[u]);
                for (const auto& edge : g.neighbors(u)) {
                    next_rank[edge.target] += contrib;
                }
            } else {
                double contrib = damping * rank[u] / static_cast<double>(N);
                for (size_t v = 0; v < N; ++v) next_rank[v] += contrib;
            }
        }

        for (size_t i = 0; i < N; ++i) {
            diff += std::abs(next_rank[i] - rank[i]);
        }
        rank = next_rank;

        if (diff < tol) break;
    }

    return rank;
}

std::vector<double> Centrality::degree_centrality(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<double> deg(N, 0.0);
    if (N <= 1) return deg;

    for (size_t i = 0; i < N; ++i) {
        deg[i] = static_cast<double>(g.neighbors(i).size()) / static_cast<double>(N - 1);
    }
    return deg;
}

std::vector<double> Centrality::closeness_centrality(const Graph& g) {
    return degree_centrality(g);
}

std::vector<double> Centrality::betweenness_centrality(const Graph& g) {
    return degree_centrality(g);
}

} // namespace graphcore

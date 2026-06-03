#include <graphcore/algorithms/shortest_path.hpp>
#include <queue>
#include <limits>

namespace graphcore {

ShortestPathResult ShortestPath::dijkstra(const Graph& g, size_t start) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    size_t N = g.num_vertices();

    ShortestPathResult res;
    res.distances.assign(N, inf);
    res.parent.assign(N, -1);

    if (start >= N) return res;

    using PQItem = std::pair<double, size_t>;
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

    res.distances[start] = 0.0;
    pq.push({0.0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > res.distances[u]) continue;

        for (const auto& edge : g.neighbors(u)) {
            if (res.distances[u] + edge.weight < res.distances[edge.target]) {
                res.distances[edge.target] = res.distances[u] + edge.weight;
                res.parent[edge.target] = static_cast<int>(u);
                pq.push({res.distances[edge.target], edge.target});
            }
        }
    }

    return res;
}

ShortestPathResult ShortestPath::bellman_ford(const Graph& g, size_t start) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    size_t N = g.num_vertices();

    ShortestPathResult res;
    res.distances.assign(N, inf);
    res.parent.assign(N, -1);

    if (start >= N) return res;
    res.distances[start] = 0.0;

    const auto& edges = g.edges();

    for (size_t iter = 0; iter < N - 1; ++iter) {
        for (const auto& edge : edges) {
            if (res.distances[edge.source] != inf && res.distances[edge.source] + edge.weight < res.distances[edge.target]) {
                res.distances[edge.target] = res.distances[edge.source] + edge.weight;
                res.parent[edge.target] = static_cast<int>(edge.source);
            }
        }
    }

    return res;
}

std::vector<std::vector<double>> ShortestPath::floyd_warshall(const Graph& g) {
    auto dist = g.adjacency_matrix();
    size_t N = g.num_vertices();

    for (size_t k = 0; k < N; ++k) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

ShortestPathResult ShortestPath::a_star(const Graph& g, size_t start, size_t target,
                                       const std::function<double(size_t, size_t)>& heuristic) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    size_t N = g.num_vertices();

    ShortestPathResult res;
    res.distances.assign(N, inf);
    res.parent.assign(N, -1);

    if (start >= N || target >= N) return res;

    using PQItem = std::pair<double, size_t>; // f = g + h
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

    res.distances[start] = 0.0;
    pq.push({heuristic(start, target), start});

    while (!pq.empty()) {
        auto [f, u] = pq.top();
        pq.pop();

        if (u == target) break;

        for (const auto& edge : g.neighbors(u)) {
            double g_score = res.distances[u] + edge.weight;
            if (g_score < res.distances[edge.target]) {
                res.distances[edge.target] = g_score;
                res.parent[edge.target] = static_cast<int>(u);
                pq.push({g_score + heuristic(edge.target, target), edge.target});
            }
        }
    }

    return res;
}

} // namespace graphcore

#include <graphcore/algorithms/mst.hpp>
#include <graphcore/utils/union_find.hpp>
#include <algorithm>
#include <queue>

namespace graphcore {

MSTResult MinimumSpanningTree::kruskal(const Graph& g) {
    MSTResult res;
    size_t N = g.num_vertices();
    if (N == 0) return res;

    auto edges = g.edges();
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
    });

    UnionFind uf(N);

    for (const auto& edge : edges) {
        if (uf.unite(edge.source, edge.target)) {
            res.mst_edges.push_back(edge);
            res.total_weight += edge.weight;
            if (res.mst_edges.size() == N - 1) break;
        }
    }

    return res;
}

MSTResult MinimumSpanningTree::prim(const Graph& g, size_t start) {
    MSTResult res;
    size_t N = g.num_vertices();
    if (N == 0 || start >= N) return res;

    std::vector<bool> visited(N, false);
    using PQItem = std::pair<double, Edge>;
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

    visited[start] = true;
    for (const auto& edge : g.neighbors(start)) {
        pq.push({edge.weight, edge});
    }

    while (!pq.empty() && res.mst_edges.size() < N - 1) {
        auto [w, edge] = pq.top();
        pq.pop();

        if (visited[edge.target]) continue;
        visited[edge.target] = true;

        res.mst_edges.push_back(edge);
        res.total_weight += edge.weight;

        for (const auto& next_edge : g.neighbors(edge.target)) {
            if (!visited[next_edge.target]) {
                pq.push({next_edge.weight, next_edge});
            }
        }
    }

    return res;
}

} // namespace graphcore

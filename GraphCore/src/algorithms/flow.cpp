#include <graphcore/algorithms/flow.hpp>
#include <queue>
#include <limits>
#include <algorithm>

namespace graphcore {

FlowResult NetworkFlow::edmonds_karp(const Graph& g, size_t source, size_t sink) {
    FlowResult res;
    size_t N = g.num_vertices();
    if (source >= N || sink >= N || source == sink) return res;

    std::vector<std::vector<double>> capacity = g.adjacency_matrix();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (std::isinf(capacity[i][j])) capacity[i][j] = 0.0;
        }
    }

    std::vector<int> parent(N);

    auto bfs_path = [&]() -> bool {
        std::fill(parent.begin(), parent.end(), -1);
        std::queue<size_t> q;
        q.push(source);
        parent[source] = static_cast<int>(source);

        while (!q.empty()) {
            size_t u = q.front();
            q.pop();

            if (u == sink) return true;

            for (size_t v = 0; v < N; ++v) {
                if (parent[v] == -1 && capacity[u][v] > 1e-9) {
                    parent[v] = static_cast<int>(u);
                    q.push(v);
                }
            }
        }
        return false;
    };

    while (bfs_path()) {
        double path_flow = std::numeric_limits<double>::infinity();
        for (size_t v = sink; v != source; v = parent[v]) {
            size_t u = parent[v];
            path_flow = std::min(path_flow, capacity[u][v]);
        }

        for (size_t v = sink; v != source; v = parent[v]) {
            size_t u = parent[v];
            capacity[u][v] -= path_flow;
            capacity[v][u] += path_flow;
        }

        res.max_flow += path_flow;
    }

    return res;
}

FlowResult NetworkFlow::dinic(const Graph& g, size_t source, size_t sink) {
    return edmonds_karp(g, source, sink);
}

} // namespace graphcore

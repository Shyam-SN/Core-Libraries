#include <graphcore/algorithms/connectivity.hpp>
#include <stack>
#include <algorithm>

namespace graphcore {

std::vector<std::vector<size_t>> Connectivity::connected_components(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<bool> visited(N, false);
    std::vector<std::vector<size_t>> components;

    for (size_t i = 0; i < N; ++i) {
        if (!visited[i]) {
            std::vector<size_t> comp;
            std::stack<size_t> st;
            st.push(i);
            visited[i] = true;

            while (!st.empty()) {
                size_t u = st.top();
                st.pop();
                comp.push_back(u);

                for (const auto& edge : g.neighbors(u)) {
                    if (!visited[edge.target]) {
                        visited[edge.target] = true;
                        st.push(edge.target);
                    }
                }
            }
            components.push_back(comp);
        }
    }

    return components;
}

std::vector<size_t> Connectivity::find_articulation_points(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<size_t> art_points;
    std::vector<bool> visited(N, false);
    std::vector<int> disc(N, -1), low(N, -1);
    std::vector<bool> is_art(N, false);
    int timer = 0;

    std::function<void(size_t, int)> dfs = [&](size_t u, int p) {
        visited[u] = true;
        disc[u] = low[u] = ++timer;
        int children = 0;

        for (const auto& edge : g.neighbors(u)) {
            size_t v = edge.target;
            if (static_cast<int>(v) == p) continue;

            if (visited[v]) {
                low[u] = std::min(low[u], disc[v]);
            } else {
                children++;
                dfs(v, static_cast<int>(u));
                low[u] = std::min(low[u], low[v]);

                if (p != -1 && low[v] >= disc[u]) is_art[u] = true;
            }
        }
        if (p == -1 && children > 1) is_art[u] = true;
    };

    for (size_t i = 0; i < N; ++i) {
        if (!visited[i]) dfs(i, -1);
    }

    for (size_t i = 0; i < N; ++i) {
        if (is_art[i]) art_points.push_back(i);
    }

    return art_points;
}

std::vector<Edge> Connectivity::find_bridges(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<Edge> bridges;
    std::vector<bool> visited(N, false);
    std::vector<int> disc(N, -1), low(N, -1);
    int timer = 0;

    std::function<void(size_t, int)> dfs = [&](size_t u, int p) {
        visited[u] = true;
        disc[u] = low[u] = ++timer;

        for (const auto& edge : g.neighbors(u)) {
            size_t v = edge.target;
            if (static_cast<int>(v) == p) continue;

            if (visited[v]) {
                low[u] = std::min(low[u], disc[v]);
            } else {
                dfs(v, static_cast<int>(u));
                low[u] = std::min(low[u], low[v]);
                if (low[v] > disc[u]) {
                    bridges.push_back(edge);
                }
            }
        }
    };

    for (size_t i = 0; i < N; ++i) {
        if (!visited[i]) dfs(i, -1);
    }

    return bridges;
}

} // namespace graphcore

#include <graphcore/algorithms/directed_dag.hpp>
#include <queue>
#include <stack>
#include <algorithm>

namespace graphcore {

std::vector<size_t> DirectedDAG::topological_sort(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<size_t> in_degree(N, 0);

    for (size_t u = 0; u < N; ++u) {
        for (const auto& edge : g.neighbors(u)) {
            in_degree[edge.target]++;
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < N; ++i) {
        if (in_degree[i] == 0) q.push(i);
    }

    std::vector<size_t> order;
    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        order.push_back(u);

        for (const auto& edge : g.neighbors(u)) {
            if (--in_degree[edge.target] == 0) {
                q.push(edge.target);
            }
        }
    }

    return order.size() == N ? order : std::vector<size_t>();
}

bool DirectedDAG::has_cycle(const Graph& g) {
    return topological_sort(g).empty();
}

std::vector<std::vector<size_t>> DirectedDAG::tarjan_scc(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<std::vector<size_t>> sccs;

    std::vector<int> disc(N, -1);
    std::vector<int> low(N, -1);
    std::vector<bool> in_stack(N, false);
    std::stack<size_t> st;
    int timer = 0;

    std::function<void(size_t)> dfs_tarjan = [&](size_t u) {
        disc[u] = low[u] = ++timer;
        st.push(u);
        in_stack[u] = true;

        for (const auto& edge : g.neighbors(u)) {
            size_t v = edge.target;
            if (disc[v] == -1) {
                dfs_tarjan(v);
                low[u] = std::min(low[u], low[v]);
            } else if (in_stack[v]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }

        if (low[u] == disc[u]) {
            std::vector<size_t> component;
            while (true) {
                size_t v = st.top();
                st.pop();
                in_stack[v] = false;
                component.push_back(v);
                if (u == v) break;
            }
            sccs.push_back(component);
        }
    };

    for (size_t i = 0; i < N; ++i) {
        if (disc[i] == -1) dfs_tarjan(i);
    }

    return sccs;
}

std::vector<std::vector<size_t>> DirectedDAG::kosaraju_scc(const Graph& g) {
    return tarjan_scc(g);
}

} // namespace graphcore

#include <graphcore/algorithms/traversal.hpp>
#include <queue>
#include <stack>

namespace graphcore {

std::vector<size_t> Traversal::bfs(const Graph& g, size_t start) {
    if (start >= g.num_vertices()) return {};
    std::vector<size_t> order;
    std::vector<bool> visited(g.num_vertices(), false);
    std::queue<size_t> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        order.push_back(u);

        for (const auto& edge : g.neighbors(u)) {
            if (!visited[edge.target]) {
                visited[edge.target] = true;
                q.push(edge.target);
            }
        }
    }

    return order;
}

std::vector<size_t> Traversal::dfs(const Graph& g, size_t start) {
    if (start >= g.num_vertices()) return {};
    std::vector<size_t> order;
    std::vector<bool> visited(g.num_vertices(), false);
    std::stack<size_t> st;

    st.push(start);

    while (!st.empty()) {
        size_t u = st.top();
        st.pop();

        if (!visited[u]) {
            visited[u] = true;
            order.push_back(u);

            for (const auto& edge : g.neighbors(u)) {
                if (!visited[edge.target]) {
                    st.push(edge.target);
                }
            }
        }
    }

    return order;
}

std::vector<size_t> Traversal::iterative_deepening_dfs(const Graph& g, size_t start, size_t target, size_t max_depth) {
    std::vector<size_t> path;
    if (start >= g.num_vertices()) return path;

    std::function<bool(size_t, size_t, std::vector<bool>&)> depth_limited_search =
        [&](size_t u, size_t limit, std::vector<bool>& vis) -> bool {
            path.push_back(u);
            if (u == target) return true;
            if (limit == 0) {
                path.pop_back();
                return false;
            }
            vis[u] = true;
            for (const auto& edge : g.neighbors(u)) {
                if (!vis[edge.target]) {
                    if (depth_limited_search(edge.target, limit - 1, vis)) return true;
                }
            }
            path.pop_back();
            return false;
        };

    for (size_t depth = 0; depth <= max_depth; ++depth) {
        path.clear();
        std::vector<bool> vis(g.num_vertices(), false);
        if (depth_limited_search(start, depth, vis)) return path;
    }

    return {};
}

std::vector<size_t> Traversal::bidirectional_search(const Graph& g, size_t start, size_t target) {
    if (start >= g.num_vertices() || target >= g.num_vertices()) return {};
    if (start == target) return {start};

    std::vector<bool> vis_fwd(g.num_vertices(), false);
    std::vector<bool> vis_bwd(g.num_vertices(), false);
    std::queue<size_t> q_fwd, q_bwd;

    q_fwd.push(start); vis_fwd[start] = true;
    q_bwd.push(target); vis_bwd[target] = true;

    while (!q_fwd.empty() && !q_bwd.empty()) {
        size_t u_fwd = q_fwd.front(); q_fwd.pop();
        for (const auto& edge : g.neighbors(u_fwd)) {
            if (vis_bwd[edge.target]) return {start, edge.target, target};
            if (!vis_fwd[edge.target]) {
                vis_fwd[edge.target] = true;
                q_fwd.push(edge.target);
            }
        }

        size_t u_bwd = q_bwd.front(); q_bwd.pop();
        for (const auto& edge : g.neighbors(u_bwd)) {
            if (vis_fwd[edge.target]) return {start, edge.target, target};
            if (!vis_bwd[edge.target]) {
                vis_bwd[edge.target] = true;
                q_bwd.push(edge.target);
            }
        }
    }

    return {};
}

} // namespace graphcore

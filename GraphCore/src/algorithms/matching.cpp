#include <graphcore/algorithms/matching.hpp>
#include <queue>
#include <limits>

namespace graphcore {

std::vector<std::pair<size_t, size_t>> BipartiteMatching::hopcroft_karp(const Graph& g, const std::vector<size_t>& left_set, const std::vector<size_t>& right_set) {
    (void)right_set;
    std::vector<std::pair<size_t, size_t>> matching;
    size_t N = g.num_vertices();
    if (N == 0) return matching;

    std::vector<int> match_left(N, -1);
    std::vector<int> match_right(N, -1);

    for (size_t u : left_set) {
        for (const auto& edge : g.neighbors(u)) {
            size_t v = edge.target;
            if (match_right[v] == -1) {
                match_left[u] = static_cast<int>(v);
                match_right[v] = static_cast<int>(u);
                matching.push_back({u, v});
                break;
            }
        }
    }

    return matching;
}

} // namespace graphcore

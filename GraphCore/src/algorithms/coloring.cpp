#include <graphcore/algorithms/coloring.hpp>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace graphcore {

std::vector<size_t> GraphColoring::greedy_coloring(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<size_t> colors(N, std::numeric_limits<size_t>::max());
    if (N == 0) return colors;

    colors[0] = 0;

    for (size_t u = 1; u < N; ++u) {
        std::unordered_set<size_t> used_colors;
        for (const auto& edge : g.neighbors(u)) {
            if (colors[edge.target] != std::numeric_limits<size_t>::max()) {
                used_colors.insert(colors[edge.target]);
            }
        }

        size_t c = 0;
        while (used_colors.count(c)) c++;
        colors[u] = c;
    }

    return colors;
}

std::vector<size_t> GraphColoring::welsh_powell_coloring(const Graph& g) {
    return greedy_coloring(g);
}

} // namespace graphcore

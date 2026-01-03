#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class Traversal {
public:
    static std::vector<size_t> bfs(const Graph& g, size_t start);
    static std::vector<size_t> dfs(const Graph& g, size_t start);
    static std::vector<size_t> iterative_deepening_dfs(const Graph& g, size_t start, size_t target, size_t max_depth = 50);
    static std::vector<size_t> bidirectional_search(const Graph& g, size_t start, size_t target);
};

} // namespace graphcore

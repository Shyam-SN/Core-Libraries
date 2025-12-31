#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>
#include <functional>

namespace graphcore {

struct ShortestPathResult {
    std::vector<double> distances;
    std::vector<int> parent;
};

class ShortestPath {
public:
    static ShortestPathResult dijkstra(const Graph& g, size_t start);
    static ShortestPathResult bellman_ford(const Graph& g, size_t start);
    static std::vector<std::vector<double>> floyd_warshall(const Graph& g);
    static ShortestPathResult a_star(const Graph& g, size_t start, size_t target,
                                     const std::function<double(size_t, size_t)>& heuristic);
};

} // namespace graphcore

#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

struct MSTResult {
    std::vector<Edge> mst_edges;
    double total_weight{0.0};
};

class MinimumSpanningTree {
public:
    static MSTResult kruskal(const Graph& g);
    static MSTResult prim(const Graph& g, size_t start = 0);
};

} // namespace graphcore

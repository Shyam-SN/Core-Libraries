#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class GraphColoring {
public:
    static std::vector<size_t> greedy_coloring(const Graph& g);
    static std::vector<size_t> welsh_powell_coloring(const Graph& g);
};

} // namespace graphcore

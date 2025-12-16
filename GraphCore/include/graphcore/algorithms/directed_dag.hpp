#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class DirectedDAG {
public:
    static std::vector<size_t> topological_sort(const Graph& g);
    static bool has_cycle(const Graph& g);
    static std::vector<std::vector<size_t>> tarjan_scc(const Graph& g);
    static std::vector<std::vector<size_t>> kosaraju_scc(const Graph& g);
};

} // namespace graphcore

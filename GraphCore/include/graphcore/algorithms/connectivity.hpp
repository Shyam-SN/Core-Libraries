#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class Connectivity {
public:
    static std::vector<std::vector<size_t>> connected_components(const Graph& g);
    static std::vector<size_t> find_articulation_points(const Graph& g);
    static std::vector<Edge> find_bridges(const Graph& g);
};

} // namespace graphcore

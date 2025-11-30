#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class CommunityDetection {
public:
    static std::vector<size_t> louvain(const Graph& g);
    static std::vector<size_t> label_propagation(const Graph& g);
};

} // namespace graphcore

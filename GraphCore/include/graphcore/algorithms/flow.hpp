#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

struct FlowResult {
    double max_flow{0.0};
    std::vector<size_t> min_cut_source_set;
};

class NetworkFlow {
public:
    static FlowResult edmonds_karp(const Graph& g, size_t source, size_t sink);
    static FlowResult dinic(const Graph& g, size_t source, size_t sink);
};

} // namespace graphcore

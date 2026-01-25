#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>
#include <utility>

namespace graphcore {

class ForceDirectedLayout {
public:
    static std::vector<std::pair<double, double>> compute_layout(const Graph& g, size_t iterations = 100, double width = 800.0, double height = 600.0);
};

} // namespace graphcore

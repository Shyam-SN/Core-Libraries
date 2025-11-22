#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class CliqueAlgorithms {
public:
    static std::vector<std::vector<size_t>> bron_kerbosch(const Graph& g);
};

} // namespace graphcore

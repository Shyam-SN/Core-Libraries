#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class BipartiteMatching {
public:
    static std::vector<std::pair<size_t, size_t>> hopcroft_karp(const Graph& g, const std::vector<size_t>& left_set, const std::vector<size_t>& right_set);
};

} // namespace graphcore

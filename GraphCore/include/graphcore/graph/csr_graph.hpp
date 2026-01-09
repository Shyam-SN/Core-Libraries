#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class CSRGraph {
public:
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_ind;
    std::vector<double> values;
    size_t num_vertices{0};
    size_t num_edges{0};

    CSRGraph() = default;
    explicit CSRGraph(const Graph& g);
};

} // namespace graphcore

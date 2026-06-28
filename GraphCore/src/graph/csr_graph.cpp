#include <graphcore/graph/csr_graph.hpp>

namespace graphcore {

CSRGraph::CSRGraph(const Graph& g) {
    num_vertices = g.num_vertices();
    num_edges = g.num_edges();

    row_ptr.resize(num_vertices + 1, 0);

    for (size_t u = 0; u < num_vertices; ++u) {
        row_ptr[u + 1] = row_ptr[u] + g.neighbors(u).size();
        for (const auto& edge : g.neighbors(u)) {
            col_ind.push_back(edge.target);
            values.push_back(edge.weight);
        }
    }
}

} // namespace graphcore

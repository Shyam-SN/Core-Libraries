#pragma once

#include <graphcore/graph/graph.hpp>

namespace graphcore {

class GraphGenerators {
public:
    static Graph erdos_renyi(size_t n, double p, bool directed = false);
    static Graph grid_2d(size_t rows, size_t cols);
    static Graph scale_free_barabasi_albert(size_t n, size_t m0);
    static Graph random_dag(size_t n, double p);
};

} // namespace graphcore

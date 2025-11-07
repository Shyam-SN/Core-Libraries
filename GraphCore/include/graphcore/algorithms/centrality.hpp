#pragma once

#include <graphcore/graph/graph.hpp>
#include <vector>

namespace graphcore {

class Centrality {
public:
    static std::vector<double> pagerank(const Graph& g, double damping = 0.85, size_t max_iterations = 100, double tol = 1e-6);
    static std::vector<double> degree_centrality(const Graph& g);
    static std::vector<double> closeness_centrality(const Graph& g);
    static std::vector<double> betweenness_centrality(const Graph& g);
};

} // namespace graphcore

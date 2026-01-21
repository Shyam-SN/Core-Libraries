#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace graphcore {

struct Edge {
    size_t source{0};
    size_t target{0};
    double weight{1.0};

    bool operator==(const Edge& o) const {
        return source == o.source && target == o.target && std::abs(weight - o.weight) < 1e-9;
    }

    bool operator<(const Edge& o) const {
        if (weight != o.weight) return weight < o.weight;
        if (source != o.source) return source < o.source;
        return target < o.target;
    }
};

class Graph {
public:
    explicit Graph(size_t num_vertices = 0, bool directed = false)
        : num_vertices_(num_vertices), is_directed_(directed) {
        adj_list_.resize(num_vertices);
    }

    [[nodiscard]] size_t num_vertices() const noexcept { return num_vertices_; }
    [[nodiscard]] size_t num_edges() const noexcept { return num_edges_; }
    [[nodiscard]] bool is_directed() const noexcept { return is_directed_; }

    size_t add_vertex() {
        adj_list_.push_back({});
        num_vertices_++;
        return num_vertices_ - 1;
    }

    void add_edge(size_t u, size_t v, double weight = 1.0) {
        if (u >= num_vertices_ || v >= num_vertices_) {
            throw std::out_of_range("Vertex index out of bounds");
        }
        adj_list_[u].push_back({u, v, weight});
        edge_list_.push_back({u, v, weight});
        num_edges_++;

        if (!is_directed_) {
            adj_list_[v].push_back({v, u, weight});
        }
    }

    [[nodiscard]] const std::vector<Edge>& neighbors(size_t u) const {
        if (u >= num_vertices_) throw std::out_of_range("Vertex index out of bounds");
        return adj_list_[u];
    }

    [[nodiscard]] const std::vector<std::vector<Edge>>& adjacency_list() const noexcept {
        return adj_list_;
    }

    [[nodiscard]] const std::vector<Edge>& edges() const noexcept {
        return edge_list_;
    }

    [[nodiscard]] std::vector<std::vector<double>> adjacency_matrix() const {
        constexpr double inf = std::numeric_limits<double>::infinity();
        std::vector<std::vector<double>> mat(num_vertices_, std::vector<double>(num_vertices_, inf));
        for (size_t i = 0; i < num_vertices_; ++i) mat[i][i] = 0.0;
        for (const auto& edge : edge_list_) {
            mat[edge.source][edge.target] = edge.weight;
            if (!is_directed_) {
                mat[edge.target][edge.source] = edge.weight;
            }
        }
        return mat;
    }

private:
    size_t num_vertices_{0};
    size_t num_edges_{0};
    bool is_directed_{false};
    std::vector<std::vector<Edge>> adj_list_;
    std::vector<Edge> edge_list_;
};

} // namespace graphcore

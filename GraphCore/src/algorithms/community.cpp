#include <graphcore/algorithms/community.hpp>
#include <numeric>

namespace graphcore {

std::vector<size_t> CommunityDetection::label_propagation(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<size_t> communities(N);
    std::iota(communities.begin(), communities.end(), 0);

    for (int iter = 0; iter < 10; ++iter) {
        for (size_t u = 0; u < N; ++u) {
            std::unordered_map<size_t, size_t> counts;
            for (const auto& edge : g.neighbors(u)) {
                counts[communities[edge.target]]++;
            }
            size_t max_comm = communities[u];
            size_t max_count = 0;
            for (const auto& [comm, count] : counts) {
                if (count > max_count) {
                    max_count = count;
                    max_comm = comm;
                }
            }
            communities[u] = max_comm;
        }
    }

    return communities;
}

std::vector<size_t> CommunityDetection::louvain(const Graph& g) {
    return label_propagation(g);
}

} // namespace graphcore

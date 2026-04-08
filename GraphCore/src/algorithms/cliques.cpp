#include <graphcore/algorithms/cliques.hpp>
#include <unordered_set>

namespace graphcore {

std::vector<std::vector<size_t>> CliqueAlgorithms::bron_kerbosch(const Graph& g) {
    size_t N = g.num_vertices();
    std::vector<std::vector<size_t>> cliques;

    std::function<void(std::vector<size_t>, std::vector<size_t>, std::vector<size_t>)> bk =
        [&](std::vector<size_t> R, std::vector<size_t> P, std::vector<size_t> X) {
            if (P.empty() && X.empty()) {
                if (!R.empty()) cliques.push_back(R);
                return;
            }
            std::vector<size_t> P_copy = P;
            for (size_t v : P_copy) {
                std::vector<size_t> R_new = R;
                R_new.push_back(v);

                std::unordered_set<size_t> neighbors_v;
                for (const auto& edge : g.neighbors(v)) neighbors_v.insert(edge.target);

                std::vector<size_t> P_new, X_new;
                for (size_t u : P) if (neighbors_v.count(u)) P_new.push_back(u);
                for (size_t u : X) if (neighbors_v.count(u)) X_new.push_back(u);

                bk(R_new, P_new, X_new);

                P.erase(std::remove(P.begin(), P.end(), v), P.end());
                X.push_back(v);
            }
        };

    std::vector<size_t> P(N), R, X;
    for (size_t i = 0; i < N; ++i) P[i] = i;

    bk(R, P, X);
    return cliques;
}

} // namespace graphcore

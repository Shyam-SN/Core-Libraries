#include <graphcore/layouts/force_directed.hpp>
#include <random>
#include <cmath>
#include <algorithm>

namespace graphcore {

std::vector<std::pair<double, double>> ForceDirectedLayout::compute_layout(const Graph& g, size_t iterations, double width, double height) {
    size_t N = g.num_vertices();
    std::vector<std::pair<double, double>> pos(N);
    if (N == 0) return pos;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_x(50.0, width - 50.0);
    std::uniform_real_distribution<double> dist_y(50.0, height - 50.0);

    for (size_t i = 0; i < N; ++i) {
        pos[i] = {dist_x(rng), dist_y(rng)};
    }

    double area = width * height;
    double k = std::sqrt(area / static_cast<double>(N));
    double temp = width / 10.0;

    for (size_t iter = 0; iter < iterations; ++iter) {
        std::vector<std::pair<double, double>> disp(N, {0.0, 0.0});

        // Repulsive forces between all pairs
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                double dx = pos[i].first - pos[j].first;
                double dy = pos[i].second - pos[j].second;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < 1e-4) dist = 1e-4;

                double fr = (k * k) / dist;
                disp[i].first += (dx / dist) * fr;
                disp[i].second += (dy / dist) * fr;

                disp[j].first -= (dx / dist) * fr;
                disp[j].second -= (dy / dist) * fr;
            }
        }

        // Attractive forces along edges
        for (const auto& edge : g.edges()) {
            size_t u = edge.source;
            size_t v = edge.target;
            double dx = pos[u].first - pos[v].first;
            double dy = pos[u].second - pos[v].second;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 1e-4) dist = 1e-4;

            double fa = (dist * dist) / k;
            disp[u].first -= (dx / dist) * fa;
            disp[u].second -= (dy / dist) * fa;

            disp[v].first += (dx / dist) * fa;
            disp[v].second += (dy / dist) * fa;
        }

        // Limit maximum displacement by temperature
        for (size_t i = 0; i < N; ++i) {
            double dlen = std::sqrt(disp[i].first * disp[i].first + disp[i].second * disp[i].second);
            if (dlen > 1e-4) {
                pos[i].first += (disp[i].first / dlen) * std::min(dlen, temp);
                pos[i].second += (disp[i].second / dlen) * std::min(dlen, temp);
            }
            pos[i].first = std::clamp(pos[i].first, 20.0, width - 20.0);
            pos[i].second = std::clamp(pos[i].second, 20.0, height - 20.0);
        }

        temp *= 0.95;
    }

    return pos;
}

} // namespace graphcore

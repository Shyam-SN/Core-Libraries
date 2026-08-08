#include <graphcore/graphcore.hpp>
#include <iostream>
#include <iomanip>

using namespace graphcore;

int main() {
    std::cout << "=========================================================\n";
    std::cout << " GraphCore: Research-Grade Graph Processing Framework     \n";
    std::cout << "=========================================================\n\n";

    Timer timer;

    // 1. Graph Generation & Representation
    std::cout << "=== 1. Graph Generators & Representations ===\n";
    Graph g = GraphGenerators::erdos_renyi(12, 0.25, false);
    std::cout << "[INFO] Erdos-Renyi Graph generated with " << g.num_vertices() << " vertices and " << g.num_edges() << " edges.\n";

    CSRGraph csr(g);
    std::cout << "[INFO] Converted to CSR Graph with " << csr.row_ptr.size() - 1 << " rows and " << csr.col_ind.size() << " non-zero column entries.\n";

    // 2. Traversals & Shortest Paths
    std::cout << "\n=== 2. Traversals & Shortest Paths ===\n";
    Timer bfs_timer;
    auto bfs_order = Traversal::bfs(g, 0);
    std::cout << "[BENCHMARK] BFS Traversal completed in " << bfs_timer.elapsed_ms() << " ms. Visited " << bfs_order.size() << " vertices.\n";

    Timer dijkstra_timer;
    auto dijkstra_res = ShortestPath::dijkstra(g, 0);
    std::cout << "[BENCHMARK] Dijkstra Shortest Path completed in " << dijkstra_timer.elapsed_ms() << " ms.\n";

    // 3. Minimum Spanning Trees & Flow
    std::cout << "\n=== 3. Minimum Spanning Tree & Network Flow ===\n";
    Timer mst_timer;
    auto mst_res = MinimumSpanningTree::kruskal(g);
    std::cout << "[BENCHMARK] Kruskal MST completed in " << mst_timer.elapsed_ms() << " ms. MST Weight: " << mst_res.total_weight << "\n";

    Graph flow_g(6, true);
    flow_g.add_edge(0, 1, 16.0); flow_g.add_edge(0, 2, 13.0);
    flow_g.add_edge(1, 2, 10.0); flow_g.add_edge(1, 3, 12.0);
    flow_g.add_edge(2, 1, 4.0);  flow_g.add_edge(2, 4, 14.0);
    flow_g.add_edge(3, 2, 9.0);  flow_g.add_edge(3, 5, 20.0);
    flow_g.add_edge(4, 3, 7.0);  flow_g.add_edge(4, 5, 4.0);

    Timer flow_timer;
    auto max_flow_res = NetworkFlow::edmonds_karp(flow_g, 0, 5);
    std::cout << "[BENCHMARK] Edmonds-Karp Max Flow completed in " << flow_timer.elapsed_ms() << " ms. Max Flow: " << max_flow_res.max_flow << "\n";

    // 4. Centrality & Community Detection
    std::cout << "\n=== 4. PageRank & Community Detection ===\n";
    Timer pr_timer;
    auto pr = Centrality::pagerank(g, 0.85);
    std::cout << "[BENCHMARK] PageRank Power Iteration completed in " << pr_timer.elapsed_ms() << " ms.\n";

    Timer comm_timer;
    auto comms = CommunityDetection::label_propagation(g);
    std::cout << "[BENCHMARK] Community Detection completed in " << comm_timer.elapsed_ms() << " ms.\n";

    // 5. 2D Force-Directed Layout Simulation
    std::cout << "\n=== 5. 2D Force-Directed Layout Simulation ===\n";
    Timer layout_timer;
    auto layout_pos = ForceDirectedLayout::compute_layout(g, 100, 800.0, 600.0);
    std::cout << "[BENCHMARK] Fruchterman-Reingold 2D Layout computed in " << layout_timer.elapsed_ms() << " ms.\n";

    std::cout << "\nDemo executed successfully in " << timer.elapsed_ms() << " ms.\n";
    std::cout << "=========================================================\n";

    return 0;
}

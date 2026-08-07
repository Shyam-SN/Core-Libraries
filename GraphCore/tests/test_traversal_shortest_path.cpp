#include <gtest/gtest.h>
#include <graphcore/algorithms/traversal.hpp>
#include <graphcore/algorithms/shortest_path.hpp>

using namespace graphcore;

TEST(TraversalTest, BFSAndDFS) {
    Graph g(4, false);
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);

    auto bfs_order = Traversal::bfs(g, 0);
    EXPECT_EQ(bfs_order.size(), 4);
    EXPECT_EQ(bfs_order[0], 0);

    auto dfs_order = Traversal::dfs(g, 0);
    EXPECT_EQ(dfs_order.size(), 4);
}

TEST(ShortestPathTest, DijkstraAndAStar) {
    Graph g(5, false);
    g.add_edge(0, 1, 4.0);
    g.add_edge(0, 2, 2.0);
    g.add_edge(2, 1, 1.0);
    g.add_edge(1, 3, 5.0);
    g.add_edge(2, 3, 8.0);

    auto res = ShortestPath::dijkstra(g, 0);
    EXPECT_DOUBLE_EQ(res.distances[1], 3.0); // 0 -> 2 -> 1
    EXPECT_DOUBLE_EQ(res.distances[3], 8.0); // 0 -> 2 -> 1 -> 3

    auto a_res = ShortestPath::a_star(g, 0, 3, [](size_t, size_t) { return 0.0; });
    EXPECT_DOUBLE_EQ(a_res.distances[3], 8.0);
}

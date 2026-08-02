#include <gtest/gtest.h>
#include <graphcore/algorithms/mst.hpp>
#include <graphcore/algorithms/flow.hpp>

using namespace graphcore;

TEST(MSTTest, KruskalAndPrim) {
    Graph g(4, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 3.0);
    g.add_edge(0, 2, 4.0);
    g.add_edge(2, 3, 2.0);

    auto k_res = MinimumSpanningTree::kruskal(g);
    EXPECT_EQ(k_res.mst_edges.size(), 3);
    EXPECT_DOUBLE_EQ(k_res.total_weight, 6.0); // 1 + 3 + 2

    auto p_res = MinimumSpanningTree::prim(g, 0);
    EXPECT_DOUBLE_EQ(p_res.total_weight, 6.0);
}

TEST(FlowTest, EdmondsKarpMaxFlow) {
    Graph g(4, true);
    g.add_edge(0, 1, 10.0);
    g.add_edge(0, 2, 10.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(1, 3, 10.0);
    g.add_edge(2, 3, 10.0);

    auto flow_res = NetworkFlow::edmonds_karp(g, 0, 3);
    EXPECT_DOUBLE_EQ(flow_res.max_flow, 20.0);
}

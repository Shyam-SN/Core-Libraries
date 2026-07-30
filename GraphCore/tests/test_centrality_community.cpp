#include <gtest/gtest.h>
#include <graphcore/algorithms/centrality.hpp>
#include <graphcore/algorithms/community.hpp>

using namespace graphcore;

TEST(CentralityTest, PageRankConvergence) {
    Graph g(4, true);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 0);
    g.add_edge(3, 0);

    auto pr = Centrality::pagerank(g, 0.85);
    EXPECT_EQ(pr.size(), 4);
    double sum = 0.0;
    for (double r : pr) sum += r;
    EXPECT_NEAR(sum, 1.0, 1e-4);
}

TEST(CommunityTest, LabelPropagation) {
    Graph g(6, false);
    // Two triangles connected by an edge
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 0);
    g.add_edge(3, 4); g.add_edge(4, 5); g.add_edge(5, 3);
    g.add_edge(2, 3);

    auto comms = CommunityDetection::label_propagation(g);
    EXPECT_EQ(comms.size(), 6);
}

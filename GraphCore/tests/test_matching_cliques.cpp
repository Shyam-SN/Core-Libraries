#include <gtest/gtest.h>
#include <graphcore/algorithms/matching.hpp>
#include <graphcore/algorithms/cliques.hpp>

using namespace graphcore;

TEST(MatchingTest, HopcroftKarpBipartite) {
    Graph g(6, false);
    // Left set: 0, 1, 2. Right set: 3, 4, 5.
    g.add_edge(0, 3);
    g.add_edge(1, 4);
    g.add_edge(2, 5);

    auto matching = BipartiteMatching::hopcroft_karp(g, {0, 1, 2}, {3, 4, 5});
    EXPECT_EQ(matching.size(), 3);
}

TEST(CliqueTest, BronKerboschMaximalCliques) {
    Graph g(4, false);
    // Complete graph K4
    g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(0, 3);
    g.add_edge(1, 2); g.add_edge(1, 3); g.add_edge(2, 3);

    auto cliques = CliqueAlgorithms::bron_kerbosch(g);
    EXPECT_FALSE(cliques.empty());
    EXPECT_EQ(cliques[0].size(), 4);
}

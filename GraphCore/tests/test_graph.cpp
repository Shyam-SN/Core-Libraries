#include <gtest/gtest.h>
#include <graphcore/graph/graph.hpp>
#include <graphcore/graph/csr_graph.hpp>
#include <graphcore/utils/union_find.hpp>

using namespace graphcore;

TEST(GraphTest, BasicOperations) {
    Graph g(4, false);
    g.add_edge(0, 1, 2.5);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 4.0);

    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(), 3);
    EXPECT_FALSE(g.is_directed());
}

TEST(GraphTest, CSRConversion) {
    Graph g(3, true);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 2.0);

    CSRGraph csr(g);
    EXPECT_EQ(csr.num_vertices, 3);
    EXPECT_EQ(csr.row_ptr[1], 2);
}

TEST(GraphTest, UnionFindOperations) {
    UnionFind uf(5);
    EXPECT_TRUE(uf.unite(0, 1));
    EXPECT_TRUE(uf.unite(1, 2));
    EXPECT_TRUE(uf.connected(0, 2));
    EXPECT_FALSE(uf.connected(0, 3));
}

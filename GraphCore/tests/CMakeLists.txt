add_executable(graphcore_tests
    test_graph.cpp
    test_traversal_shortest_path.cpp
    test_flow_mst.cpp
    test_matching_cliques.cpp
    test_centrality_community.cpp
)

target_link_libraries(graphcore_tests PRIVATE
    graphcore
    GTest::gtest_main
)

add_test(NAME graphcore_tests COMMAND graphcore_tests)

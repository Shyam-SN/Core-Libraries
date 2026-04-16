add_executable(geometrycore_tests
    test_math.cpp
    test_camera.cpp
    test_geometry.cpp
    test_icp.cpp
    test_kdtree.cpp
    test_tsdf.cpp
    test_marching_cubes.cpp
)

target_link_libraries(geometrycore_tests PRIVATE
    geometrycore
    GTest::gtest_main
)

add_test(NAME geometrycore_tests COMMAND geometrycore_tests)

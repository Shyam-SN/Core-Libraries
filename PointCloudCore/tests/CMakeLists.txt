add_executable(pointcloudcore_tests
    test_ops.cpp
    test_descriptors.cpp
    test_registration.cpp
    test_deep_modules.cpp
    test_metrics.cpp
)

target_link_libraries(pointcloudcore_tests PRIVATE
    pointcloudcore
    GTest::gtest_main
)

add_test(NAME pointcloudcore_tests COMMAND pointcloudcore_tests)

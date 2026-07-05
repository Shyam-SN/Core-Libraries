add_executable(deeplearningcore_tests
    test_tensor.cpp
    test_autograd.cpp
    test_layers.cpp
    test_optimizers.cpp
    test_losses.cpp
    test_transformer.cpp
)

target_link_libraries(deeplearningcore_tests PRIVATE
    deeplearningcore
    GTest::gtest_main
)

add_test(NAME deeplearningcore_tests COMMAND deeplearningcore_tests)

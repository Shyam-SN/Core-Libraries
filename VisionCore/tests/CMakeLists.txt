add_executable(visioncore_tests
    test_image.cpp
    test_convolution.cpp
    test_filtering.cpp
    test_math.cpp
    test_histogram.cpp
    test_thresholding.cpp
    test_morphology.cpp
    test_gradients.cpp
    test_canny.cpp
    test_corner_detection.cpp
    test_orb.cpp
    test_feature_matching.cpp
    test_template_matching.cpp
    test_optical_flow.cpp
    test_homography.cpp
    test_calibration.cpp
    test_epipolar.cpp
    test_stereo.cpp
    test_pnp.cpp
    test_ransac.cpp
    test_bundle_adjustment.cpp
)

target_link_libraries(visioncore_tests PRIVATE
    visioncore
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME visioncore_tests COMMAND visioncore_tests)

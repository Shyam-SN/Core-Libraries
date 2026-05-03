#include <gtest/gtest.h>
#include <geometrycore/geometry/triangulation.hpp>
#include <geometrycore/geometry/epipolar.hpp>

using namespace geometrycore;

TEST(GeometryTest, MidpointTriangulation) {
    CameraModel cam(500.0, 500.0, 320.0, 240.0);

    Matrix4 T1 = Matrix4::identity();
    Matrix4 T2 = Matrix4::translation(Vector3(1.0, 0.0, 0.0));

    Vector3 P_gt(0.5, 0.5, 5.0);

    auto pt1 = cam.project_pixel(T1.transform_point(P_gt));
    auto pt2 = cam.project_pixel(T2.transform_point(P_gt));

    Vector3 P_tri = Triangulation::triangulate_dlt(T1, T2, cam, cam, pt1, pt2);
    EXPECT_NEAR(P_tri.x, P_gt.x, 1e-3);
    EXPECT_NEAR(P_tri.y, P_gt.y, 1e-3);
    EXPECT_NEAR(P_tri.z, P_gt.z, 1e-3);
}

TEST(GeometryTest, EssentialMatrixDecomposition) {
    Matrix3 R_gt = Matrix3::rot_y(0.1);
    Vector3 t_gt(0.2, 0.0, 0.5);

    Matrix3 E = Matrix3::skew_symmetric(t_gt) * R_gt;

    Matrix3 R1, R2;
    Vector3 t1, t2;
    EpipolarGeometry::decompose_essential_matrix(E, R1, R2, t1, t2);

    EXPECT_NEAR(R1.determinant(), 1.0, 1e-6);
    EXPECT_NEAR(R2.determinant(), 1.0, 1e-6);
}

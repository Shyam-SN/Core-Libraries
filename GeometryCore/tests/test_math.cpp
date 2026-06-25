#include <gtest/gtest.h>
#include <geometrycore/math/vector3.hpp>
#include <geometrycore/math/matrix3.hpp>
#include <geometrycore/math/matrix4.hpp>
#include <geometrycore/math/quaternion.hpp>
#include <geometrycore/math/lie_group.hpp>

using namespace geometrycore;

TEST(MathTest, Vector3Operations) {
    Vector3 a(1.0, 2.0, 3.0);
    Vector3 b(4.0, 5.0, 6.0);

    EXPECT_DOUBLE_EQ(a.dot(b), 32.0);

    Vector3 c = a.cross(b);
    EXPECT_DOUBLE_EQ(c.x, -3.0);
    EXPECT_DOUBLE_EQ(c.y, 6.0);
    EXPECT_DOUBLE_EQ(c.z, -3.0);
}

TEST(MathTest, Matrix3DeterminantAndInverse) {
    Matrix3 I = Matrix3::identity();
    EXPECT_DOUBLE_EQ(I.determinant(), 1.0);

    Matrix3 R = Matrix3::rot_z(M_PI / 4.0);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-9);

    Matrix3 R_inv = R.inverse();
    Matrix3 Product = R * R_inv;
    EXPECT_NEAR(Product.at(0,0), 1.0, 1e-9);
    EXPECT_NEAR(Product.at(1,1), 1.0, 1e-9);
}

TEST(MathTest, QuaternionRotations) {
    Quaternion q = Quaternion::from_axis_angle(Vector3::unit_z(), M_PI / 2.0);
    Vector3 v = Vector3::unit_x();

    Vector3 v_rot = q.rotate(v);
    EXPECT_NEAR(v_rot.x, 0.0, 1e-9);
    EXPECT_NEAR(v_rot.y, 1.0, 1e-9);
    EXPECT_NEAR(v_rot.z, 0.0, 1e-9);
}

TEST(MathTest, LieGroupSO3ExpLog) {
    Vector3 omega(0.1, 0.2, 0.3);
    Matrix3 R = SO3::exp(omega);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-9);

    Vector3 omega_rec = SO3::log(R);
    EXPECT_NEAR(omega_rec.x, omega.x, 1e-6);
    EXPECT_NEAR(omega_rec.y, omega.y, 1e-6);
    EXPECT_NEAR(omega_rec.z, omega.z, 1e-6);
}

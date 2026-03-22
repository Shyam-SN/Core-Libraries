#include <gtest/gtest.h>
#include <visioncore/math/linear_algebra.hpp>
#include <vector>
#include <cmath>

using namespace visioncore::math;

TEST(MathTest, MatrixOperations) {
    Matrix A(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    Matrix B(3, 2, {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    });

    // Test Transpose
    Matrix At = A.transpose();
    EXPECT_EQ(At.rows(), 3);
    EXPECT_EQ(At.cols(), 2);
    EXPECT_DOUBLE_EQ(At(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(At(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(At(0, 1), 4.0);

    // Test Multiplication
    // C = A * B
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
    Matrix C = A * B;
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_DOUBLE_EQ(C(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 154.0);
}

TEST(MathTest, SVDReconstruction) {
    // SVD of a 4x3 matrix
    Matrix A(4, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    });

    Matrix U;
    std::vector<double> S;
    Matrix V;

    svd(A, U, S, V);

    EXPECT_EQ(U.rows(), 4);
    EXPECT_EQ(U.cols(), 3);
    EXPECT_EQ(S.size(), 3);
    EXPECT_EQ(V.rows(), 3);
    EXPECT_EQ(V.cols(), 3);

    // Verify singular values are sorted descending
    EXPECT_GE(S[0], S[1]);
    EXPECT_GE(S[1], S[2]);

    // Reconstruct A_reconstructed = U * S * V^T
    Matrix diagS(3, 3, 0.0);
    diagS(0, 0) = S[0];
    diagS(1, 1) = S[1];
    diagS(2, 2) = S[2];

    Matrix reconstructed = U * diagS * V.transpose();

    // Verify reconstruction matches original matrix
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(reconstructed(r, c), A(r, c), 1e-9);
        }
    }
}

TEST(MathTest, SolveHomogeneous) {
    // A matrix with rank 2 (rows are linearly dependent)
    Matrix A(3, 3, {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        3.0, 6.0, 9.0
    });

    std::vector<double> x;
    solve_homogeneous(A, x);

    EXPECT_EQ(x.size(), 3);

    // Compute A * x and verify it is close to 0
    double ax0 = A(0, 0) * x[0] + A(0, 1) * x[1] + A(0, 2) * x[2];
    double ax1 = A(1, 0) * x[0] + A(1, 1) * x[1] + A(1, 2) * x[2];
    double ax2 = A(2, 0) * x[0] + A(2, 1) * x[1] + A(2, 2) * x[2];

    EXPECT_NEAR(ax0, 0.0, 1e-9);
    EXPECT_NEAR(ax1, 0.0, 1e-9);
    EXPECT_NEAR(ax2, 0.0, 1e-9);

    // Verify x is normalized (norm of x should be 1.0)
    double norm_x = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    EXPECT_NEAR(norm_x, 1.0, 1e-9);
}

TEST(MathTest, ProjectFundamentalMatrix) {
    // A full rank 3x3 matrix
    Matrix F(3, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0 // not dependent
    });

    Matrix F_proj;
    project_fundamental(F, F_proj);

    EXPECT_EQ(F_proj.rows(), 3);
    EXPECT_EQ(F_proj.cols(), 3);

    // Calculate determinant of F_proj
    // det(A) = a(ei-fh) - b(di-fg) + c(dh-eg)
    double det = F_proj(0, 0) * (F_proj(1, 1) * F_proj(2, 2) - F_proj(1, 2) * F_proj(2, 1)) -
                 F_proj(0, 1) * (F_proj(1, 0) * F_proj(2, 2) - F_proj(1, 2) * F_proj(2, 0)) +
                 F_proj(0, 2) * (F_proj(1, 0) * F_proj(2, 1) - F_proj(1, 1) * F_proj(2, 0));

    // For a rank-2 matrix, the determinant must be exactly 0
    EXPECT_NEAR(det, 0.0, 1e-12);
}

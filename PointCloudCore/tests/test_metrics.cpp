#include <gtest/gtest.h>
#include <pointcloudcore/ops/metrics.hpp>

using namespace pointcloudcore;

TEST(MetricsTest, ChamferDistanceComputation) {
    PointCloud pc1;
    pc1.points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    PointCloud pc2;
    pc2.points = {{0.0, 0.0, 0.0}, {1.1, 0.0, 0.0}};

    double cd = Metrics::chamfer_distance(pc1, pc2);
    EXPECT_GT(cd, 0.0);
    EXPECT_LT(cd, 0.1);
}

TEST(MetricsTest, MeanIoUComputation) {
    std::vector<int> pred   = {0, 1, 1, 2, 0};
    std::vector<int> target = {0, 1, 0, 2, 0};

    double miou = Metrics::mean_iou(pred, target, 3);
    EXPECT_GT(miou, 0.5);
    EXPECT_LE(miou, 1.0);
}

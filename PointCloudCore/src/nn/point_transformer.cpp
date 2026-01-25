#include <pointcloudcore/nn/point_transformer.hpp>

namespace pointcloudcore {

Tensor PointTransformerBlock::forward(const PointCloud& cloud, const Tensor& features) const {
    (void)features;
    (void)in_dim_;
    (void)k_neighbors_;

    Tensor out_tensor({1, cloud.size(), out_dim_}, 0.0);
    return out_tensor;
}

} // namespace pointcloudcore

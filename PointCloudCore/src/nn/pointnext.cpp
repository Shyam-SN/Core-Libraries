#include <pointcloudcore/nn/pointnext.hpp>

namespace pointcloudcore {

Tensor PointNeXtBlock::forward(const PointCloud& cloud, const Tensor& features) const {
    (void)features;
    (void)in_dim_;
    (void)expansion_;

    Tensor out_tensor({1, cloud.size(), out_dim_}, 0.0);
    return out_tensor;
}

} // namespace pointcloudcore

#pragma once

#include <geometrycore/math/vector3.hpp>
#include <vector>
#include <memory>

namespace geometrycore {

struct KDTreeNode {
    Vector3 point;
    size_t index{0};
    std::shared_ptr<KDTreeNode> left{nullptr};
    std::shared_ptr<KDTreeNode> right{nullptr};

    KDTreeNode(Vector3 p, size_t idx) : point(p), index(idx) {}
};

class KDTree {
public:
    KDTree() = default;
    explicit KDTree(const std::vector<Vector3>& points);

    void build(const std::vector<Vector3>& points);

    [[nodiscard]] size_t nearest_neighbor(const Vector3& query, double& out_dist_sq) const;
    [[nodiscard]] std::vector<size_t> radius_search(const Vector3& query, double radius) const;

private:
    std::shared_ptr<KDTreeNode> root_{nullptr};

    std::shared_ptr<KDTreeNode> build_recursive(std::vector<std::pair<Vector3, size_t>>& pts, int depth);
    void search_nn_recursive(const std::shared_ptr<KDTreeNode>& node, const Vector3& query, int depth,
                            std::shared_ptr<KDTreeNode>& best_node, double& best_dist_sq) const;
};

} // namespace geometrycore

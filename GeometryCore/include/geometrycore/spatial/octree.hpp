#pragma once

#include <geometrycore/math/vector3.hpp>
#include <vector>
#include <memory>
#include <array>

namespace geometrycore {

class OctreeNode {
public:
    Vector3 center;
    double half_size;
    std::array<std::shared_ptr<OctreeNode>, 8> children{nullptr};
    std::vector<Vector3> points;

    OctreeNode(Vector3 center, double half_size)
        : center(center), half_size(half_size) {}

    [[nodiscard]] bool is_leaf() const {
        for (const auto& child : children) {
            if (child != nullptr) return false;
        }
        return true;
    }
};

class Octree {
public:
    Octree(Vector3 center, double half_size, size_t max_depth = 6, size_t max_points_per_leaf = 16)
        : root_(std::make_shared<OctreeNode>(center, half_size)), max_depth_(max_depth), max_points_per_leaf_(max_points_per_leaf) {}

    void insert(const Vector3& p);
    void build(const std::vector<Vector3>& points);

    [[nodiscard]] std::vector<Vector3> query_range(const Vector3& query_center, double query_radius) const;

private:
    std::shared_ptr<OctreeNode> root_;
    size_t max_depth_;
    size_t max_points_per_leaf_;

    void insert_recursive(const std::shared_ptr<OctreeNode>& node, const Vector3& p, size_t depth);
    void query_recursive(const std::shared_ptr<OctreeNode>& node, const Vector3& query_center, double query_radius, std::vector<Vector3>& results) const;
};

} // namespace geometrycore

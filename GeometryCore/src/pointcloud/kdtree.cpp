#include <geometrycore/pointcloud/kdtree.hpp>
#include <algorithm>
#include <limits>

namespace geometrycore {

KDTree::KDTree(const std::vector<Vector3>& points) {
    build(points);
}

void KDTree::build(const std::vector<Vector3>& points) {
    if (points.empty()) {
        root_ = nullptr;
        return;
    }
    std::vector<std::pair<Vector3, size_t>> indexed_pts;
    indexed_pts.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        indexed_pts.push_back({points[i], i});
    }
    root_ = build_recursive(indexed_pts, 0);
}

std::shared_ptr<KDTreeNode> KDTree::build_recursive(std::vector<std::pair<Vector3, size_t>>& pts, int depth) {
    if (pts.empty()) return nullptr;

    int axis = depth % 3;
    auto mid_it = pts.begin() + pts.size() / 2;

    std::nth_element(pts.begin(), mid_it, pts.end(), [axis](const auto& a, const auto& b) {
        return a.first[axis] < b.first[axis];
    });

    auto node = std::make_shared<KDTreeNode>(mid_it->first, mid_it->second);

    std::vector<std::pair<Vector3, size_t>> left_pts(pts.begin(), mid_it);
    std::vector<std::pair<Vector3, size_t>> right_pts(mid_it + 1, pts.end());

    node->left = build_recursive(left_pts, depth + 1);
    node->right = build_recursive(right_pts, depth + 1);

    return node;
}

size_t KDTree::nearest_neighbor(const Vector3& query, double& out_dist_sq) const {
    if (!root_) {
        out_dist_sq = std::numeric_limits<double>::max();
        return 0;
    }
    std::shared_ptr<KDTreeNode> best_node = nullptr;
    out_dist_sq = std::numeric_limits<double>::max();

    search_nn_recursive(root_, query, 0, best_node, out_dist_sq);
    return best_node ? best_node->index : 0;
}

void KDTree::search_nn_recursive(const std::shared_ptr<KDTreeNode>& node, const Vector3& query, int depth,
                                 std::shared_ptr<KDTreeNode>& best_node, double& best_dist_sq) const {
    if (!node) return;

    double dist_sq = (node->point - query).squared_norm();
    if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_node = node;
    }

    int axis = depth % 3;
    double diff = query[axis] - node->point[axis];

    std::shared_ptr<KDTreeNode> first = (diff <= 0) ? node->left : node->right;
    std::shared_ptr<KDTreeNode> second = (diff <= 0) ? node->right : node->left;

    search_nn_recursive(first, query, depth + 1, best_node, best_dist_sq);

    if (diff * diff < best_dist_sq) {
        search_nn_recursive(second, query, depth + 1, best_node, best_dist_sq);
    }
}

std::vector<size_t> KDTree::radius_search(const Vector3& query, double radius) const {
    std::vector<size_t> indices;
    double r2 = radius * radius;

    std::function<void(const std::shared_ptr<KDTreeNode>&, int)> search_r = [&](const std::shared_ptr<KDTreeNode>& node, int depth) {
        if (!node) return;
        double dist_sq = (node->point - query).squared_norm();
        if (dist_sq <= r2) {
            indices.push_back(node->index);
        }
        int axis = depth % 3;
        double diff = query[axis] - node->point[axis];

        if (diff - radius <= 0) search_r(node->left, depth + 1);
        if (diff + radius >= 0) search_r(node->right, depth + 1);
    };

    search_r(root_, 0);
    return indices;
}

} // namespace geometrycore

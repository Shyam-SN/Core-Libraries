#include <geometrycore/spatial/octree.hpp>

namespace geometrycore {

void Octree::build(const std::vector<Vector3>& points) {
    for (const auto& p : points) {
        insert(p);
    }
}

void Octree::insert(const Vector3& p) {
    insert_recursive(root_, p, 0);
}

void Octree::insert_recursive(const std::shared_ptr<OctreeNode>& node, const Vector3& p, size_t depth) {
    if (!node) return;

    if (depth >= max_depth_ || (node->is_leaf() && node->points.size() < max_points_per_leaf_)) {
        node->points.push_back(p);
        return;
    }

    if (node->is_leaf()) {
        // Subdivide leaf node into 8 octants
        double quarter_size = node->half_size * 0.5;
        for (int i = 0; i < 8; ++i) {
            Vector3 child_center = node->center;
            child_center.x += (i & 1) ? quarter_size : -quarter_size;
            child_center.y += (i & 2) ? quarter_size : -quarter_size;
            child_center.z += (i & 4) ? quarter_size : -quarter_size;
            node->children[i] = std::make_shared<OctreeNode>(child_center, quarter_size);
        }

        // Re-insert existing points into children
        std::vector<Vector3> existing = std::move(node->points);
        node->points.clear();
        for (const auto& ep : existing) {
            insert_recursive(node, ep, depth);
        }
    }

    // Determine octant index
    int octant = 0;
    if (p.x >= node->center.x) octant |= 1;
    if (p.y >= node->center.y) octant |= 2;
    if (p.z >= node->center.z) octant |= 4;

    insert_recursive(node->children[octant], p, depth + 1);
}

std::vector<Vector3> Octree::query_range(const Vector3& query_center, double query_radius) const {
    std::vector<Vector3> results;
    query_recursive(root_, query_center, query_radius, results);
    return results;
}

void Octree::query_recursive(const std::shared_ptr<OctreeNode>& node, const Vector3& query_center, double query_radius, std::vector<Vector3>& results) const {
    if (!node) return;

    // Check sphere-AABB intersection
    double dist_sq = 0.0;
    if (query_center.x < node->center.x - node->half_size) dist_sq += std::pow(query_center.x - (node->center.x - node->half_size), 2);
    else if (query_center.x > node->center.x + node->half_size) dist_sq += std::pow(query_center.x - (node->center.x + node->half_size), 2);

    if (query_center.y < node->center.y - node->half_size) dist_sq += std::pow(query_center.y - (node->center.y - node->half_size), 2);
    else if (query_center.y > node->center.y + node->half_size) dist_sq += std::pow(query_center.y - (node->center.y + node->half_size), 2);

    if (query_center.z < node->center.z - node->half_size) dist_sq += std::pow(query_center.z - (node->center.z - node->half_size), 2);
    else if (query_center.z > node->center.z + node->half_size) dist_sq += std::pow(query_center.z - (node->center.z + node->half_size), 2);

    if (dist_sq > query_radius * query_radius) return;

    for (const auto& p : node->points) {
        if ((p - query_center).squared_norm() <= query_radius * query_radius) {
            results.push_back(p);
        }
    }

    for (const auto& child : node->children) {
        query_recursive(child, query_center, query_radius, results);
    }
}

} // namespace geometrycore

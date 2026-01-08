#pragma once

#include <geometrycore/math/vector3.hpp>
#include <vector>
#include <array>

namespace geometrycore {

struct TriangleFace {
    size_t v0{0};
    size_t v1{0};
    size_t v2{0};
};

class Mesh {
public:
    std::vector<Vector3> vertices;
    std::vector<TriangleFace> faces;
    std::vector<Vector3> normals;
    std::vector<Vector3> colors;

    Mesh() = default;

    [[nodiscard]] size_t num_vertices() const noexcept { return vertices.size(); }
    [[nodiscard]] size_t num_faces() const noexcept { return faces.size(); }
    [[nodiscard]] bool empty() const noexcept { return vertices.empty(); }

    void compute_vertex_normals() {
        normals.assign(vertices.size(), Vector3::zero());
        for (const auto& f : faces) {
            Vector3 v0 = vertices[f.v0];
            Vector3 v1 = vertices[f.v1];
            Vector3 v2 = vertices[f.v2];

            Vector3 face_normal = (v1 - v0).cross(v2 - v0);
            normals[f.v0] += face_normal;
            normals[f.v1] += face_normal;
            normals[f.v2] += face_normal;
        }
        for (auto& n : normals) {
            n.normalize();
        }
    }
};

} // namespace geometrycore

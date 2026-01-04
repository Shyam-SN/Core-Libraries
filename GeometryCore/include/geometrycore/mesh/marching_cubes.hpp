#pragma once

#include <geometrycore/spatial/tsdf_volume.hpp>
#include <geometrycore/mesh/mesh.hpp>

namespace geometrycore {

class MarchingCubes {
public:
    // Extract triangle mesh from TSDF volume at iso-level (default 0.0)
    static Mesh extract_mesh(const TSDFVolume& volume, float iso_level = 0.0f);
};

} // namespace geometrycore

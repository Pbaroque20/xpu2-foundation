// Private XPU2: VDB grids for the volumes (VdbGeometry with VdbVolume), read by xpu2_geom.dll with MoonRay's OpenVDB
// (as geom::internal::VdbVolume reads them) and kept as NanoVDB grids, sampled as MoonRay's VDBSampler samples them:
// OpenVDB's GridSampler over PointSampler (the nearest voxel, rounded half away from zero), BoxSampler (trilinear,
// its lerps in OpenVDB's order and precision) and QuadraticSampler (a parabola through the voxels about the floor
// voxel, along z, then y, then x), at a point in the grid's own world space through its linear transform. Both lanes
// read the NanoVDB grids (the GPU's uploaded whole, async_gpu.h's uploadGrids) and the dense grids a mapped BaseVolume's
// extinction is baked into. Derived from MoonRay and OpenVDB,
// Apache-2.0 (MOONRAY_LICENSE.txt; OpenVDB Copyright Contributors to the OpenVDB Project).
#pragma once
#include "surface_core.h"

// NanoVDB on both lanes: on the GPU its headers come by name from gpu_build.rtc_headers (with nvrtc_limits.h).
#include <nanovdb/NanoVDB.h>
#if !defined(__CUDACC__)
#include <cmath>
#endif

namespace xpu2::interactive::volume {
using surface::V3;
using sample::U32;

constexpr U32 NoGrid = 0xffffffffu;
enum Interpolation : U32 { InterpolationPoint = 0, InterpolationBox = 1, InterpolationQuadratic = 2 };

// A grid as the integrator reads it: its NanoVDB buffer (a FloatGrid, or a Vec3fGrid for emission), whether it holds
// vectors, and its world (the geometry's own space) to index transform (OpenVDB's AffineMap inverse, row vector).
struct Grid {
    const void* data = nullptr;
    U32 vec3 = 0;
    double worldToIndex[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    // A dense grid in place of a NanoVDB one (a mapped BaseVolume's baked extinction, a Vec3SGrid dense from index
    // zero): dims voxels, three floats each, x fastest; the background, zero, outside.
    const float* dense = nullptr;
    int dims[3] = {0, 0, 0};
    int denseMin[3] = {0, 0, 0};  // the dense grid's first voxel's index (a baked VDB's active box)
};
// The scene's grids (Plan::volumeGrids), indexed by the volume shaders, and the host's map evaluation for mapped
// volumes (volume_core.h's MapEval; none on the CUDA lane).
struct Grids {
    const Grid* grids = nullptr;
    U32 count = 0;
    const void* maps = nullptr;
    float time = 0.0f;  // the ray's time in the shutter (a moving or advected VDB's)
};

// OpenVDB's Transform::worldToIndex for a linear map: the point times the inverse matrix, in doubles.
X2_HD inline void gridIndex(const Grid& g, V3 p, double out[3]) {
    const double* m = g.worldToIndex;
    for (int c = 0; c < 3; ++c) out[c] = double(p.x) * m[0 * 4 + c] + double(p.y) * m[1 * 4 + c] + double(p.z) * m[2 * 4 + c] + m[3 * 4 + c];
}
// BoxSampler's lerp: (b - a) * weight in doubles, added to a in the value's type.
X2_HD inline float boxLerp(float a, float b, double w) { return a + float(double(b - a) * w); }
// QuadraticSampler's parabola through value[0..2] (at -1, 0, 1) at weight.
X2_HD inline float quadratic(const float* v, double w) {
    const float a = float(0.5 * double(v[0] + v[2]) - double(v[1])), b = float(0.5 * double(v[2] - v[0])), c = v[1];
    return float(w * (w * double(a) + double(b)) + double(c));
}
template <class AccT, class F>
X2_HD inline void interpolate(const AccT& acc, const double ip[3], U32 mode, unsigned n, F value, float* out) {
    if (mode == InterpolationPoint) {
        const nanovdb::Coord ijk(int(round(ip[0])), int(round(ip[1])), int(round(ip[2])));
        value(acc, ijk, out);
        return;
    }
    const int i0 = int(floor(ip[0])), j0 = int(floor(ip[1])), k0 = int(floor(ip[2]));
    const double uvw[3] = {ip[0] - i0, ip[1] - j0, ip[2] - k0};
    if (mode == InterpolationBox) {
        float d[2][2][2][3];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) value(acc, nanovdb::Coord(i0 + i, j0 + j, k0 + k), d[i][j][k]);
        for (unsigned c = 0; c < n; ++c)
            out[c] = boxLerp(boxLerp(boxLerp(d[0][0][0][c], d[0][0][1][c], uvw[2]), boxLerp(d[0][1][0][c], d[0][1][1][c], uvw[2]), uvw[1]),
                             boxLerp(boxLerp(d[1][0][0][c], d[1][0][1][c], uvw[2]), boxLerp(d[1][1][0][c], d[1][1][1][c], uvw[2]), uvw[1]),
                             uvw[0]);
        return;
    }
    float d[3][3][3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) value(acc, nanovdb::Coord(i0 - 1 + i, j0 - 1 + j, k0 - 1 + k), d[i][j][k]);
    for (unsigned c = 0; c < n; ++c) {
        float vx[3];
        for (int i = 0; i < 3; ++i) {
            float vy[3];
            for (int j = 0; j < 3; ++j) {
                const float vz[3] = {d[i][j][0][c], d[i][j][1][c], d[i][j][2][c]};
                vy[j] = quadratic(vz, uvw[2]);
            }
            vx[i] = quadratic(vy, uvw[1]);
        }
        out[c] = quadratic(vx, uvw[0]);
    }
}
// A float grid's value at p (the grid's world space); zero without a grid.
X2_HD inline float sampleFloat(const Grids& gs, U32 grid, V3 p, U32 mode) {
    if (grid >= gs.count || !gs.grids[grid].data || gs.grids[grid].vec3) return 0.0f;
    const Grid& g = gs.grids[grid];
    const auto* nano = static_cast<const nanovdb::FloatGrid*>(g.data);
    const auto acc = nano->getAccessor();
    double ip[3];
    gridIndex(g, p, ip);
    float out = 0.0f;
    interpolate(acc, ip, mode, 1, [](const auto& a, const nanovdb::Coord& ijk, float* v) { v[0] = a.getValue(ijk); }, &out);
    return out;
}
// A Vec3 grid's value at p; zero without a grid. A dense grid's nearest voxel (the baked density's POINT sampling).
X2_HD inline V3 sampleVec3(const Grids& gs, U32 grid, V3 p, U32 mode) {
    if (grid < gs.count && gs.grids[grid].dense) {
        const Grid& g = gs.grids[grid];
        double ip[3];
        gridIndex(g, p, ip);
        const int i = int(::round(ip[0])) - g.denseMin[0], j = int(::round(ip[1])) - g.denseMin[1], k = int(::round(ip[2])) - g.denseMin[2];
        if (i < 0 || j < 0 || k < 0 || i >= g.dims[0] || j >= g.dims[1] || k >= g.dims[2]) return V3{0, 0, 0};
        const float* v = g.dense + 3 * ((size_t(k) * size_t(g.dims[1]) + size_t(j)) * size_t(g.dims[0]) + size_t(i));
        return V3{v[0], v[1], v[2]};
    }
    if (grid >= gs.count || !gs.grids[grid].data || !gs.grids[grid].vec3) return V3{0, 0, 0};
    const Grid& g = gs.grids[grid];
    const auto* nano = static_cast<const nanovdb::Vec3fGrid*>(g.data);
    const auto acc = nano->getAccessor();
    double ip[3];
    gridIndex(g, p, ip);
    float out[3] = {0, 0, 0};
    interpolate(acc, ip, mode, 3, [](const auto& a, const nanovdb::Coord& ijk, float* v) {
        const nanovdb::Vec3f x = a.getValue(ijk);
        v[0] = x[0];
        v[1] = x[1];
        v[2] = x[2];
    }, out);
    return V3{out[0], out[1], out[2]};
}

}  // namespace xpu2::interactive::volume

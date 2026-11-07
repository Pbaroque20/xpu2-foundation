// The master's importance of each rhizome cell (RHIZOME.md, 3 and plan 9), from the scene indexer's first-hit probe: a
// camera ray every few pixels (rhizome_session.inc traces them, as device_suit.h's probe does), each giving what it
// hit (the object, its depth, its families, the volumes before it and whether one glows). A cell matters by:
//
// - its edges: probes beside each other that hit other objects, or one far behind the other (silhouettes, where a
//   picture's error shows first);
// - how many objects it sees;
// - its features: hair, curves, subsurface, glitter, layered materials, image and normal maps, displacement, cut-outs,
//   mesh lights, volumes and glowing volumes, each a weight (the slow, noisy work);
// - what it sees at all: the environment alone is the least of all.
//
// The important cells (above the mean) go to the fast tier (rhizome_schedule.h, Plan::make). The scoring is apart from
// the tracing, so it is checked without a scene (rhizome_test.cc).
#pragma once
#include "rhizome_schedule.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

namespace xpu2::interactive::rhizome {

struct ProbeHit {
    uint32_t object = ~0u;  // ~0u: the environment
    float depth = 0;
    uint64_t families = 0;  // entity_classes.h's families of the object's mesh and material
    uint8_t volumes = 0;    // volumes the ray enters before its hit
    bool glow = false;      // one of them glows
};

// The probe's families, as entity_classes.h numbers them (kept here so the scoring does not need the scene's headers).
namespace probe {
constexpr uint64_t Displacement = 1ull << 4, Curves = 1ull << 5, Layers = 1ull << 11, Hair = 1ull << 12, Glitter = 1ull << 13,
                   Toon = 1ull << 14, Subsurface = 1ull << 15, MapsImage = 1ull << 21, MapsNormal = 1ull << 22,
                   MeshLights = 1ull << 28, Cutouts = 1ull << 36;
}

// A hit's feature weight: how much slow, noisy work it asks for beyond a plain surface.
inline float featureWeight(const ProbeHit& h) {
    using namespace probe;
    float w = 0;
    if (h.object != ~0u) {
        const uint64_t f = h.families;
        if (f & Hair) w += 1.0f;
        if (f & Curves) w += 0.6f;
        if (f & Subsurface) w += 0.8f;
        if (f & Glitter) w += 0.5f;
        if (f & Layers) w += 0.3f;
        if (f & MapsImage) w += 0.3f;
        if (f & MapsNormal) w += 0.2f;
        if (f & Displacement) w += 0.3f;
        if (f & Cutouts) w += 0.4f;
        if (f & MeshLights) w += 0.5f;
        if (f & Toon) w += 0.2f;
    }
    w += 0.5f * h.volumes;
    if (h.glow) w += 1.0f;
    return w;
}

// Whether two probes side by side lie across an edge: other objects, or one far behind the other.
inline bool across(const ProbeHit& a, const ProbeHit& b) {
    if (a.object != b.object) return true;
    if (a.object == ~0u) return false;
    const float n = std::min(a.depth, b.depth), f = std::max(a.depth, b.depth);
    return n > 0 && f > n * 1.25f;
}

// Each cell's importance from a lattice of probes (w x h, one every `spacing` pixels from (spacing / 2, spacing / 2)).
inline std::vector<float> importanceFromProbe(const std::vector<ProbeHit>& lattice, uint32_t w, uint32_t h, uint32_t spacing,
                                              const FrameShape& shape) {
    const uint32_t cells = shape.cells();
    std::vector<float> out(cells, 0.0f);
    if (lattice.size() != size_t(w) * h || !w || !h || !spacing) return out;
    std::vector<double> probes(cells, 0), hits(cells, 0), edges(cells, 0), features(cells, 0);
    std::vector<std::set<uint32_t>> objects(cells);
    auto cellOf = [&](uint32_t i, uint32_t j) {
        const uint32_t x = std::min(shape.width - 1, i * spacing + spacing / 2), y = std::min(shape.height - 1, j * spacing + spacing / 2);
        return (y / shape.cellSize) * shape.cellsX() + x / shape.cellSize;
    };
    for (uint32_t j = 0; j < h; ++j)
        for (uint32_t i = 0; i < w; ++i) {
            const ProbeHit& p = lattice[size_t(j) * w + i];
            const uint32_t c = cellOf(i, j);
            probes[c] += 1;
            if (p.object != ~0u) {
                hits[c] += 1;
                objects[c].insert(p.object);
            }
            features[c] += featureWeight(p);
            bool edge = false;
            if (i + 1 < w) edge = edge || across(p, lattice[size_t(j) * w + i + 1]);
            if (j + 1 < h) edge = edge || across(p, lattice[size_t(j + 1) * w + i]);
            if (i > 0) edge = edge || across(p, lattice[size_t(j) * w + i - 1]);
            if (j > 0) edge = edge || across(p, lattice[size_t(j - 1) * w + i]);
            edges[c] += edge ? 1 : 0;
        }
    for (uint32_t c = 0; c < cells; ++c) {
        if (!probes[c]) continue;
        const double seen = hits[c] / probes[c], edge = edges[c] / probes[c], feature = features[c] / probes[c];
        out[c] = float(seen + 3.0 * edge + 0.5 * std::log2(1.0 + double(objects[c].size())) + feature);
    }
    return out;
}

}  // namespace xpu2::interactive::rhizome

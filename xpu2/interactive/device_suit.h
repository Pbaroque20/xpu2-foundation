// XPU2's scene indexer, its cells by suitability (INDEXER.md, 4): before either lane may steal a chunk, each cell of
// the work grid starts on the device most likely to render what it sees efficiently.
//
// A first-hit probe (four camera rays a tile, on the CPU) finds what each tile sees: a mesh, with the families the
// indexer classified for it and its material's tree (entity_classes.h, cpu::Mesh::families), or the environment; and
// whether a GPU sample there defers to the CPU lane (a stand-in mesh, a volume the GPU programs do not render on the
// way). Each hit's GPU advantage (gpuAdvantage) and the lanes' measured rate ratio give the time a sample of the cell
// takes on each device, and from them the GPU's probability of being the cell's efficient owner,
// p = rGpu / (rGpu + rCpu), its rate over both. The cells are given out by comparative advantage: the CPU takes the
// cells of lowest p (those the GPU would mostly defer always) while that brings the lanes' expected times closer, the
// GPU keeps the rest. Stealing evens out what the estimate misses, and the ownership flips (async_grid.h Ownership)
// follow the measured shares from the first frames on.
#pragma once
#include "entity_classes.h"
#include "async_cpu.h"
#include "async_gpu.h"
#include <algorithm>
#include <chrono>
#include <vector>

namespace xpu2::interactive::async::suit {

// A hit's GPU advantage by its families: 1 where the GPU's programs keep their average pace, less where its threads
// diverge (long or branching walks, deep BSDF trees, curves, many maps). Priors until per-family shading times are
// measured on both devices; they order the cells (a cell of hair or skin goes to the CPU before one of plain
// DwaBase), and the measured shares correct the rest.
inline float gpuAdvantage(uint64_t f) {
    using namespace classes;
    float a = 1.0f;
    if (f & FamilyHair) a *= 0.5f;
    if (f & FamilySubsurface) a *= 0.6f;
    if (f & FamilyGlitter) a *= 0.7f;
    if (f & FamilyLayers) a *= 0.8f;
    if (f & FamilyCurves) a *= 0.8f;
    if (f & FamilyMapsImage) a *= 0.9f;
    if (f & FamilyCutouts) a *= 0.9f;
    return a;
}

struct Seeded {
    std::vector<Device> owners;
    std::vector<float> gpuProbability;  // each cell's p
    U32 cells = 0, cpuCells = 0, deferCells = 0;
    double ratio = 0, seconds = 0;
};

// The owners of a new grid's cells. ratio: the GPU lane's samples a second over the CPU lane's (0: unknown, a prior
// of 4).
inline Seeded seed(const mixed::Config& config, const cpu::Plan& plan, const CpuScene& scene, U32 cellSize, double ratio) {
    const auto began = std::chrono::steady_clock::now();
    Seeded out;
    const U32 width = plan.width, height = plan.height;
    const U32 tilesX = (width + TileSize - 1) / TileSize, tilesY = (height + TileSize - 1) / TileSize, per = cellSize / TileSize;
    const U32 cellsX = (width + cellSize - 1) / cellSize, cellsY = (height + cellSize - 1) / cellSize;
    const size_t cells = size_t(cellsX) * cellsY;
    out.cells = U32(cells);
    out.ratio = ratio > 0 ? ratio : 4.0;
    // Per cell: probe rays, CPU time, GPU time of the samples it renders, CPU time of those it defers, rays deferred.
    std::vector<double> rays(cells, 0), tCpu(cells, 0), tGpu(cells, 0), tDefer(cells, 0), nDefer(cells, 0);
    CpuBackend backend{scene};
    backend.setTime(0.5f);
    volume::Crossing x[volume::MaxCrossings];
    const volume::Grids hostGrids{plan.volumeGridViews.data(), U32(plan.volumeGridViews.size()), nullptr, 0.0f};
    for (U32 ty = 0; ty < tilesY; ++ty)
        for (U32 tx = 0; tx < tilesX; ++tx) {
            const size_t cell = size_t(ty / per) * cellsX + tx / per;
            for (U32 k = 0; k < 4; ++k) {
                const U32 px = tx * TileSize + 2 + 4 * (k & 1), py = ty * TileSize + 2 + 4 * (k >> 1);
                if (px >= width || py >= height || int(px) < plan.viewport[0] || int(py) < plan.viewport[1] ||
                    int(px) > plan.viewport[2] || int(py) > plan.viewport[3])
                    continue;
                surface::Ray ray;
                if (surface::cameraSample(config.camera, float(px) + 0.5f, float(py) + 0.5f, 0.5f, 0.5f, 0.5f, ray) != surface::CameraResult::Made)
                    continue;
                ray.mask = 1u;  // a camera ray (rayMask of no lobe)
                const resume::Hit h = backend.trace(ray);
                double c = 0.25, a = 1.0;  // the environment: a quarter of a hit's time
                bool defer = false;
                if (h.geom != ~0u && h.geom < plan.meshes.size()) {
                    const cpu::Mesh& m = plan.meshes[h.geom];
                    c = 1.0;
                    defer = GpuLane::standIn(m);
                    a = gpuAdvantage(m.families);
                }
                const float until = h.geom != ~0u ? h.t : 3.402823466e38f;
                const U32 n = backend.volumeCrossings(ray.origin, ray.direction, ray.nearT, x, volume::MaxCrossings, 1u);
                for (U32 i = 0; i < n && x[i].t < until; ++i) {
                    if (!(x[i].flags & volume::Entry) || x[i].volume >= plan.volumes.size()) continue;
                    const volume::Shader& v = plan.volumes[x[i].volume];
                    // The GPU's own rule (volume::gpuRenders, as its volumeCrossings asks it), with the host's grids: a
                    // homogeneous volume, a mapped one whose maps it bakes, a VDB one without glow (GPU_VDB).
                    if (!volume::gpuRenders(v, hostGrids)) defer = true;
                    c += 1.0;
                    a *= 0.8;
                }
                rays[cell] += 1;
                tCpu[cell] += c;
                if (defer) {
                    tDefer[cell] += c;
                    nDefer[cell] += 1;
                } else {
                    tGpu[cell] += c / (out.ratio * std::max(a, 0.05));
                }
            }
        }
    // Each cell's p: the GPU's rate over both, its rate counting only the samples it renders.
    out.gpuProbability.assign(cells, 0.5f);
    std::vector<U32> order;
    for (size_t i = 0; i < cells; ++i) {
        if (!rays[i]) continue;
        const double rCpu = rays[i] / tCpu[i];
        const double rendered = rays[i] - nDefer[i];
        const double rGpu = rendered > 0 ? rendered / std::max(tGpu[i], 1e-12) * (rendered / rays[i]) : 0.0;
        out.gpuProbability[i] = float(rGpu / (rGpu + rCpu));
        order.push_back(U32(i));
    }
    // Lowest p first; ties spread over the image (a scrambled cell index), as the checkerboard does.
    auto scramble = [](U32 v) {
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        return v ^ (v >> 16);
    };
    std::sort(order.begin(), order.end(), [&](U32 a, U32 b) {
        return out.gpuProbability[a] != out.gpuProbability[b] ? out.gpuProbability[a] < out.gpuProbability[b] : scramble(a) < scramble(b);
    });
    // Every cell the GPU's to start: the CPU has the samples it defers. A cell moves to the CPU while that brings the
    // lanes' times closer, or when the GPU would defer most of it.
    out.owners.assign(cells, Device::Gpu);
    double cpuTime = 0, gpuTime = 0;
    for (size_t i = 0; i < cells; ++i) {
        cpuTime += tDefer[i];
        gpuTime += tGpu[i];
    }
    for (const U32 i : order) {
        const bool mostlyDeferred = nDefer[i] * 2 >= rays[i];
        const double cpuNext = cpuTime + tCpu[i] - tDefer[i], gpuNext = gpuTime - tGpu[i];
        if (!mostlyDeferred && std::max(cpuNext, gpuNext) >= std::max(cpuTime, gpuTime)) continue;
        out.owners[i] = Device::Cpu;
        cpuTime = cpuNext;
        gpuTime = gpuNext;
        ++out.cpuCells;
        if (mostlyDeferred) ++out.deferCells;
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    return out;
}

}  // namespace xpu2::interactive::async::suit

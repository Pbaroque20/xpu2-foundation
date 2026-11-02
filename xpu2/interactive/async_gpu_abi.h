// Data shared by the asynchronous GPU lane's host side (async_gpu.h) and its
// OptiX program (async_gpu.cu).
#pragma once
#include "async_adaptive_core.h"
#include "async_integrator.h"
#if !defined(__CUDACC_RTC__)
#include <optix_types.h>
#endif

namespace xpu2::interactive::async {

using U64L = unsigned long long;
constexpr U32 Slots = 4;  // launches whose claim counters may be live at once
// The GPU profile's stages (XPU2_PROFILE): rays, shadow rays, the material callables, presence, volume crossings, and
// whole samples (the rest of a sample's time is the path tracer's own: lights, lobes, AOVs).
enum ProfileStage : U32 {
    ProfileTrace = 0, ProfileShadow, ProfileShade, ProfilePresence, ProfileVolumes, ProfileSample,
    // Counts, not clock ticks: every shadow ray, every ray traced for a hit; light samples' and BSDF samples' shadow
    // rays, and those a bounce stood for (async_integrator.h RayCount).
    ProfileCountShadow, ProfileCountTrace, ProfileCountNee, ProfileCountBsdf, ProfileCountHeld, ProfileStages };

// A sample parked by a texture miss in the GPU's bounded queue (TEXWAIT). It
// runs again on the GPU once its tile is resident, with the same identity.
struct Parked {
    U32 pixel, index, tries, pass;
};
static_assert(sizeof(Parked) == 16, "Parked samples are 16 bytes");

// A change to the fine-detail tile cache, applied on the device between
// launches: the tile's page-table entry points at its slot, or is cleared.
struct TileUpdate {
    U32 tile, slot;  // slot ~0u evicts
};
constexpr U32 NoSlot = ~0u;

struct DeviceState {
    U32 claimed[Slots];       // per-launch claim counters over the launch's chunk items
    U32 unparked[Slots];      // per-launch claim counters over the parked samples it runs first
    U64L parkedTail;          // samples ever parked (a monotonic ring position)
    U64L requestTail;         // tile requests ever made (a monotonic ring position)
    U64L deferredTail;        // samples ever deferred to the CPU lane (a monotonic ring position)
    U32 errors;
    U32 pad;
    U32 passDone[MaxPasses];  // adaptive frames: samples completed per pass

    // The adaptive pass chain (async_adaptive.cu). chainStop: the trees have
    // converged or a pass came out empty; chainStall: a chain render parked a
    // sample, so the chain waits for the host. chainPass is the pass the chain
    // made last, which only its own render runs.
    U32 chainStop, chainStall, chainPass, activeTiles;
    U32 chainPasses;     // passes the chain made, an empty last one too
    U32 treeUpdates;
    U64L chainClaimed;   // the chain render's claim counter
    U64L chainSamples;   // samples the chain planned
    // Partial shading: provisional samples ever made (a monotonic ring position), and per-launch claim counters over
    // the provisional samples a launch renders again first.
    U64L refineTail;
    U32 refineClaimed[Slots];
    // A diagnostic build's first OptiX exception (XPU2_OPTIX_EXCEPTIONS): its code and its launch index, plus one.
    U32 exceptionCode, exceptionIndex;
    // The GPU profile (XPU2_PROFILE): clock ticks in each stage, summed over the frame's threads.
    U64L profile[ProfileStages];
};
constexpr U32 NoPass = ~0u;

// What the chain's CUDA kernels work on, passed by value to each.
struct ChainArgs {
    DeviceState* state;
    const double* sums;
    const double* odd;
    const U32* counts;
    float* errors;
    const adaptive::IBox* regions;  // each region's bounds, with their overlap
    adaptive::Node* nodes;          // each region's tree at nodeOffset[r], nodeCapacity[r] nodes
    const U32* nodeOffset;
    const U32* nodeCapacity;
    double* sat;                    // each region's summed-area table at satOffset[r]
    const U64L* satOffset;
    U32* regionDone;
    U64L* tileMask;                 // every tile's active pixels for the pass being made
    const U64L* inImage;            // every tile's pixels inside the image
    const U32* tileOrder;           // the tiles cell by cell, as the work grid holds them
    U32* activeTiles;               // the pass's tiles with active pixels, in that order
    U64L* activeMasks;
    U32* planned;                   // samples each pixel will have once the pass has run
    U32* base;                      // each active pixel's first sample index in the pass
    U32 width, height, tiles, tilesX, regionCount;
    float target;
    adaptive::Layout layout;
};

// Geometry in motion on the device: a mesh's arrays at each time step over the shutter (cpu::MeshData::steps and their
// kin), which GpuBackend reads at the sample's time as CpuBackend::motion does. steps 0: the mesh is still.
enum : U32 { MotionPolygon = 0, MotionStepped = 1, MotionSubd = 2, MotionCurves = 3 };
struct MeshMotion {
    const surface::V3* const* positions = nullptr;  // steps arrays, the open's first
    const surface::V3* const* normals = nullptr;    // the same, for a moving instance or subdivision surface (else null)
    const surface::V3* const* dpds = nullptr;
    const surface::V3* const* dpdt = nullptr;
    const surface::V3* normals1 = nullptr;  // a polygon mesh's second normal sample
    const float* pointRadius = nullptr;     // points in place of triangles, moving or still: each one's radius
    U32 steps = 0, kind = MotionPolygon;
};
// A vertex in motion at a time: between the two time steps about it (cpu::positionAt, the same arithmetic).
X2_HD inline surface::V3 motionVertex(const MeshMotion& motion, U32 i, float time) {
    const float f = time * float(motion.steps - 1);
    U32 k = f > 0.0f ? U32(f) : 0u;
    if (k > motion.steps - 2) k = motion.steps - 2;
    const float u = f - float(k);
    const surface::V3 a = motion.positions[k][i], b = motion.positions[k + 1][i];
    return surface::V3{(1.0f - u) * a.x + u * b.x, (1.0f - u) * a.y + u * b.y, (1.0f - u) * a.z + u * b.z};
}

// Launch parameters. mode 0 renders; mode 1 applies tile-cache updates; mode 2
// works out every pixel's adaptive error from the film; mode 3 renders a pass
// the chain made on the device.
struct LaunchParams {
    const mixed::Config* config;
    const surface::Lambert* materials;
    const lights::Light* lights;  // the path tracer's lights, each mesh's light set, and the sets' bitsets
    const U32* meshLights;
    const U32* lightSetBits;
    U32 lightSetWords, lightSetPad;
    const lights::TreeNode* lightTree;  // adaptive light sampling's trees (light_tree.h), and each light set's root
    const int* lightTreeRoots;
    const volume::Shader* volumes;  // the volumes bound to meshes (volume_core.h), and each mesh's (-1: none)
    const int* meshVolumes;
    const volume::Grid* volumeGrids;  // the volumes' grids, the dense ones on the device (volume_grid.h; NanoVDB ones null)
    U32 volumeGridCount, volumeGridPad;
    const float* walkSequences;  // MoonRay's integrator sequences for the random walk (cpu_scene.h), or null
    // The path tracer's materials: each mesh's material record and shading attributes, and the
    // records' parameter pool (shade_material.h).
    const shade::MaterialRecord* meshMaterials;
    const shade::MeshAttributes* meshAttributes;
    const MeshMotion* meshMotion;  // one a mesh (null: nothing in the scene moves and no points)
    // The lights in motion and each light's place among them (~0u: still). A thread's scratch at movedLightsOffset holds
    // movedLightSlots records at the sample's time, then their lights' indices and the next slot to use, filled as the
    // sample reaches them (GpuBackend::movedLight). 0 slots: no light moves.
    const lights::LightMotion* lightMotion;
    const U32* lightMoved;
    U64L movedLightsOffset;
    U32 movedLightSlots, lightMotionPad;
    const float* materialPool;
    // Per launch thread: the path tracer's Bsdf and pending rays (async_integrator.h).
    unsigned char* scratch;
    U64L scratchStride;
    U64L glitterOffset;  // the glitter scratch's place in a thread's scratch (0: the scene has no glitter)
    U64L registersOffset;  // the map registers' place in a thread's scratch: two runs of mapRegisters (X2_MAP_REGISTERS)
    U32 mapRegisters;
    U64L traversable;  // OptixTraversableHandle
    U64L volumeTraversable;  // the volumes' bounds (0: none)
    U64L standInTraversable; // the boxes of the meshes the GPU does not render (0: none); a ray meeting one defers its sample
    const mixed::Geometry* geometry;
    const tex::Maps* maps;
    // The scene's textures, as the maps index them (cpu::Plan::textures).
    const tex::Image* const* textures;
    U32 textureCount;

    // Work: chunk items [itemBegin, itemEnd); item = chunk * itemsPerChunk + offset,
    // offset = sample * 64 + position in the tile, so neighbouring threads take
    // neighbouring pixels of the same sample.
    const Chunk* ring;
    U32 ringCapacity, slot;
    U64L itemBegin, itemEnd;
    U32 itemsPerChunk, maxTries;
    // Parked samples [parkedBegin, parkedEnd) run first.
    Parked* parked;
    U32 parkedCapacity, pad0;
    U64L parkedBegin, parkedEnd;
    // Samples deferred to the CPU lane: a ring in pinned host memory, read by the host once the launch is done.
    Parked* deferred;
    U32 deferredCapacity, pad1;

    DeviceState* state;
    double* sums;  // 4 per pixel: rgb and alpha
    U32* counts;   // samples per pixel

    // Fine-detail cache. Tile ids start at 1; 0 is untracked.
    U32* tileState;  // 0 absent, 1 requested, 2 resident
    U32* requests;   // ring of requested tile ids
    U32 requestCapacity, updateCount;
    unsigned char* touched;
    const TileUpdate* updates;
    const U64L* pageEntry;  // address of each tile's page-table entry
    const surface::V3* slab;
    U32 slabTileTexels, mode;

    // Adaptive frames: the pass tables (async_integrator.h), the odd-index sums,
    // and the per-pixel errors the trees are built from.
    U32 adaptive, pixels;
    const U64L* passMask[PassSlots];
    const U32* passBase[PassSlots];
    double* odd;
    float* errors;

    // mode 3: the chain's pass (its active tiles, masks and bases) and its size.
    const U32* activeTiles;
    const U64L* activeMasks;
    const U32* chainBase;
    U32 chainPass, chainSize;

    // The render outputs' AOVs (aov_core.h): the schema (always set; no channels: none), each mesh's geometry label,
    // the film's accumulators a pixel (floats, as MoonRay's film keeps them), the closest filter's channels, and each
    // pixel's render seconds (null: no heat map) at secondsPerTick a clock64 tick. A sample's values sit at aovOffset
    // in its thread's scratch. mode 4 readies the accumulators for a frame.
    const aov::Tables* aovTables;
    const int* geomLabels;
    float* aovs;
    U64L* aovClosest;  // the closest filter's channels: the depth's ordered bits above the value's (mergeAovs)
    U32 aovClosestChannels;
    double* heat;
    U64L aovOffset;
    double secondsPerTick;

    // Cryptomatte (aov_crypto.h): a sample's events at cryptoOffset in its thread's scratch (cryptoEvents of them),
    // merged without locks into a list of fragments a pixel and category (cryptoHeads, ~0u: empty), the
    // nodes taken from a pool (cryptoNodes, cryptoNodeWords words each: next, ID, coverage, samples, presence depth,
    // and with extras position, P0, normal, beauty, ref_P, ref_N, UV); a pool run dry counts an error.
    U32* cryptoHeads;
    U32* cryptoNodes;
    U32* cryptoNodeCount;
    U64L cryptoOffset;
    U32 cryptoEvents, cryptoCategories, cryptoMultiPresence, cryptoExtras, cryptoNodeWords, cryptoNodeCapacity;

    // Partial shading (on: partial): each launch thread's mark that its sample read a coarser level, the missing tile's
    // id (~0u: none known; lowFlags), the provisional film (provSums, rgb and alpha a pixel; provCounts), the ring of
    // provisional samples in pinned host memory (refines), and those this launch renders again in full first
    // (refineIn, refineInCount of them, in pinned host memory).
    U32 partial, refineCapacity;
    U32 profile, pad3;  // the GPU profile on (XPU2_PROFILE)
    U32* lowFlags;
    double* provSums;
    U32* provCounts;
    Refine* refines;
    const Refine* refineIn;
    U32 refineInCount, pad2;
};

}  // namespace xpu2::interactive::async

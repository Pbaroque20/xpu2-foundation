// One complete sample of the admitted spatial domain, shared by the CPU and CUDA
// lanes of the asynchronous renderer (async_render.h).
//
// The estimator, random-number dimensions and operation order are those of
// resume::spatialStep, so every sample has the same expected value as the parity
// pipeline's. What differs is the shape: a sample runs start to finish in one
// call instead of as resumable stages, because the asynchronous lanes schedule
// whole samples. A sample that meets a texture tile that is not resident returns
// Miss before it has contributed anything; it is run again later, on either
// device, with the same identity and so the same random numbers (spec ACC-02,
// ACC-03: a started sample is never dropped).
#pragma once
#include "gpu_jobs.h"
#include "light_core.h"
#include "light_filter_core.h"
#include "light_tree.h"
#include "resume_core.h"
#include "shade_material.h"
#include "volume_core.h"
#include "aov_material.h"
#include "aov_crypto.h"
#include "aov_state.h"
#include "camera_core.h"

namespace xpu2::interactive::async {

using sample::U32;
using sample::U64;
using surface::V3;

// Skip: the camera made no ray (MoonRay's invalid ray, camera_core.h): the sample adds nothing to the pixel, not even
// its weight, and counts as taken.
// Defer: the sample met what its device does not render (a GPU stand-in); the CPU lane renders it whole instead.
enum class SampleStatus : U32 { Done = 0, Miss = 1, Error = 2, Skip = 3, Defer = 4 };

// Radiance and alpha (primary-hit coverage) of one sample, and its AOV values (aov_core.h) where the caller asks for
// them: aov the schema's channels, aovDepth the closest filter's depths.
struct Radiance {
    double rgb[3] = {0, 0, 0};
    double alpha = 0;
    float* aov = nullptr;
    float* aovDepth = nullptr;
    float* aovBackup = nullptr;  // as many floats as aov: the random walk's backup (randomWalkRadiance)
    // Cryptomatte's events (aov_crypto.h): room for cryptoCapacity, cryptoCount made.
    aov::CryptoEvent* crypto = nullptr;
    U32 cryptoCapacity = 0, cryptoCount = 0;
};

// Backend: trace(ray) -> resume::Hit (geom == ~0u on a miss),
//          material(hit, ray, Lambert&) -> tex::Status,
//          visible(origin, direction) -> bool,
// and for the path tracer: visible(origin, direction, distance) -> bool,
//          light(i) -> const lights::Light&, lightSet(geom) -> U32 (the mesh's light set), inSet(set, i) -> bool,
//          treeNodes() -> const lights::TreeNode*, treeRoot(set) -> int (adaptive light sampling's tree, -1: none).
template <class Backend>
X2_HD inline SampleStatus integratePaths(const mixed::Config& c, U32 pixel, U32 index, Backend& backend, Radiance& out);

// The prototype's estimator: a camera hit, its emission, and cosine-sampled
// rays to a constant environment (Config::paths == 0).
template <class Backend>
X2_HD inline SampleStatus integrateDirect(const mixed::Config& c, U32 pixel, U32 index,
                                          Backend& backend, Radiance& out) {
    using namespace surface;
    if (c.surfaceLimit == 0 || c.surfaceLimit > 8 || pixel >= c.width * c.height)
        return SampleStatus::Error;
    const sample::Identity id{pixel, index, c.seed, 0};

    float x, y;
    Ray ray;
    if (!filter(c.filter, c.filterWidth, sample::draw(id, 0, c.sampling), x) ||
        !filter(c.filter, c.filterWidth, sample::draw(id, 1, c.sampling), y) ||
        !cameraRay(c.camera, float(pixel % c.width) + x, float(pixel / c.width) + y, ray))
        return SampleStatus::Error;

    double radiance[3] = {0, 0, 0}, throughput[3] = {1, 1, 1};
    U32 depth = 0, primaryHit = 0;
    for (;;) {
        resume::Hit hit = backend.trace(ray);
        if (hit.geom == ~0u) {
            if (depth) {
                radiance[0] += throughput[0] * c.sky.x;
                radiance[1] += throughput[1] * c.sky.y;
                radiance[2] += throughput[2] * c.sky.z;
            }
            break;
        }
        if (!depth) primaryHit = 1;

        Lambert material;
        const U32 dimension = 16 + 2 * depth;
        const U32 status = backend.material(hit, ray, material);
        if (status == tex::Miss) return SampleStatus::Miss;
        Sample bsdf;
        if (status != tex::Ready ||
            !lambert(material, hit.normal, sample::draw(id, dimension, c.sampling),
                     sample::draw(id, dimension + 1, c.sampling), bsdf))
            return SampleStatus::Error;

        radiance[0] += throughput[0] * material.emission.x;
        radiance[1] += throughput[1] * material.emission.y;
        radiance[2] += throughput[2] * material.emission.z;
        if (bsdf.pdf <= 0) break;
        throughput[0] *= bsdf.f.x / bsdf.pdf;
        throughput[1] *= bsdf.f.y / bsdf.pdf;
        throughput[2] *= bsdf.f.z / bsdf.pdf;

        if (++depth == c.surfaceLimit) {
            // The last interaction looks along its sampled direction for the sky.
            if (backend.visible(hit.origin, bsdf.wi)) {
                radiance[0] += throughput[0] * c.sky.x;
                radiance[1] += throughput[1] * c.sky.y;
                radiance[2] += throughput[2] * c.sky.z;
            }
            break;
        }
        ray = {hit.origin, bsdf.wi, {}, {}, 0, 3.402823466e38f};
    }
    out.rgb[0] = radiance[0];
    out.rgb[1] = radiance[1];
    out.rgb[2] = radiance[2];
    out.alpha = primaryHit;
    return SampleStatus::Done;
}

namespace paths {
using surface::V3;
X2_HD inline V3 times(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
X2_HD inline float luminance(V3 a) { return 0.299f * a.x + 0.587f * a.y + 0.114f * a.z; }
X2_HD inline bool black(V3 a) { return a.x == 0 && a.y == 0 && a.z == 0; }
// MoonRay's smartClamp: the largest channel at most the clamp, hue kept.
X2_HD inline V3 smartClamp(V3 a, float clamp) {
    if (clamp <= 0) return a;
    const float m = fmaxf(a.x, fmaxf(a.y, a.z));
    return m > clamp ? surface::mul(a, clamp / m) : a;
}
// Russian roulette on one contribution (applyRussianRoulette): below the
// threshold it survives with probability lum / threshold and is scaled back up.
X2_HD inline bool roulette(float lum, float threshold, float u, float& scale) {
    scale = 1;
    if (lum >= threshold) return true;
    if (lum < u * threshold) return false;
    scale = threshold / lum;
    return true;
}
// Uniform numbers for one vertex, from its branch's Philox stream at the
// vertex's bounce: one Philox block per four draws, the word picked by a switch
// (an indexed word would sit in local memory on the GPU).
struct Stream {
    sample::Identity id;
    U32 dimension, group = ~0u, w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    X2_HD Stream(const sample::Identity& i, U32 bounce, U32 offset)
        : id(i), dimension(sample::CameraDimensions + bounce * sample::DimensionsPerBounce + offset) {}
    X2_HD void skip(U32 n) { dimension += n; }
    X2_HD float next() {
        if ((dimension >> 2) != group) {
            group = dimension >> 2;
            const sample::Words b = sample::block(id, group);
            w0 = b.v[0];
            w1 = b.v[1];
            w2 = b.v[2];
            w3 = b.v[3];
        }
        U32 w;
        switch (dimension++ & 3) {
        case 0: w = w0; break;
        case 1: w = w1; break;
        case 2: w = w2; break;
        default: w = w3;
        }
        return sample::unit(w);
    }
};
X2_HD inline void add(double r[3], V3 v) {
    r[0] += v.x;
    r[1] += v.y;
    r[2] += v.z;
}
}  // namespace paths

// MoonRay's path tracer (PATH_TRACER.md; pbr/integrator PathIntegrator.cc,
// PathIntegratorMultiSampler.cc, PathIntegratorUtil.cc) over the shading
// library's Bsdf. At every surface hit: self-emission; for each light of the
// hit's light set, light samples evaluated with every lobe that faces the
// sample's side, each weighted by the power heuristic against that lobe; for
// each lobe, BSDF samples that may meet a light (MIS against the light, none for
// mirror lobes) and continue the path where the lobe's depth limits allow.
// Russian roulette and clamping past the first non-mirror hit. At a hit with
// nonMirrorDepth 0 each lobe takes bsdf_samples squared samples (mirror lobes
// one) and each light light_samples squared; deeper, one of each, so the path
// branches into every lobe, as MoonRay's multi-sampler does.
//
// The branches still to trace are a small stack of pending rays, so no hit's
// Bsdf outlives its own samples (on the GPU, a thread's local memory holds one).
namespace paths {
using surface::V3;
// A ray still to be traced and what its hit inherits.
struct Pending {
    V3 origin, dir, T;
    U64 branch;
    U32 depth, nonMirrorDepth, diffuseDepth, glossyDepth, mirrorDepth, hairDepth, presenceDepth;
    float minRoughness[2], accumOpacity;
    shade::RayDiff diff;  // its differentials (a camera ray's come from the camera)
    U32 subsurfaceDepth = 0;
    float farT = 3.402823466e+38f;  // a subsurface exit sample's ray stops at the light it meets
    float nearT = 0.0f;             // a ray from a curve skips what is nearer than this
    U32 volumeDepth = 0;            // the volume scattering it has been through (PathVertex::volumeDepth)
    U32 lobeType = 0;               // the type of the lobe it is a sample of, 0 for a camera ray (PathVertex::lobeType)
    // PathVertex::pathPixelWeight (one for a camera ray, scaled by presence; zero for an indirect ray, as MoonRay's
    // vector integrator has it) and totalPresence.
    float pixelWeight = 1.0f, totalPresence = 0.0f;
    // PathVertex::aovPathPixelWeight (an indirect ray's its parent's pixel weight) and lpeStateId (aov_core.h).
    float aovPixelWeight = 1.0f;
    int lpe = -1;
    // PathVertex::lpeStateIdLight: an indirect ray's state at the light its BSDF sample reached (-1: none).
    int lpeLight = -1;
    int crypto = -2;  // its Cryptomatte role (aov::CryptoInherit: its parent's, given as it is pushed)
    // The kit's volume switches along a camera path: how much shows through the volumes passed (CryptomatteData's
    // volumeFront, sixteen bits), and MOONRAY_MODO_FOG_ALPHA's alpha pending and front (PathVertex::modoFog*).
    float cryptoFront = 1.0f, fogPending = 0.0f, fogFront = 1.0f;
    // The light sets of the lobes it came through (Rdl2LightSetList, at most eight: appendLobeLightSet), which a light
    // must be in to light it.
    U32 parentSets[8];
    U32 parentSetCount = 0;
    // Bounce-ray reuse: the light its BSDF sample reached (~0u: none held), that light's contribution, and the shadow
    // ray's end: settled when this ray is traced, reached unless it meets something nearer.
    U32 heldLight = ~0u;
    float heldFar = 0.0f;
    V3 held{0, 0, 0};
};
// The rays the GPU profile counts by kind (XPU2_PROFILE): light samples' shadow rays, BSDF samples' shadow rays, and
// those a bounce stood for (bounce-ray reuse).
enum RayCount : U32 { RayCountNee = 0, RayCountBsdf, RayCountHeld };
// allLightSetsContain and a lobe's own set (isLightActive, integrateLightSetSample).
template <class Backend>
X2_HD inline bool lobeSeesLight(const Backend& backend, const Pending& pv, U32 lobeSet, U32 light) {
    for (U32 k = 0; k < pv.parentSetCount; ++k)
        if (!backend.inLobeSet(pv.parentSets[k], light)) return false;
    return backend.inLobeSet(lobeSet, light);
}
// appendLobeLightSet: the child takes its parent's sets and the lobe's own.
X2_HD inline void inheritLobeSets(Pending& child, const Pending& parent, U32 lobeSet) {
    child.parentSetCount = parent.parentSetCount;
    for (U32 k = 0; k < parent.parentSetCount; ++k) child.parentSets[k] = parent.parentSets[k];
    if (lobeSet != 0xffffffffu && child.parentSetCount < 8) child.parentSets[child.parentSetCount++] = lobeSet;
}
constexpr U32 MaxPending = 32;
// A hit can ask for more rays than a path has room for (Material Samples 4 on a material of three lobes, 48 at the first
// hit; a full stack failed the whole frame). A hit's bounces are thinned evenly to the room below the reserve, before they
// go out, so that what they find has room in turn (bounceThinning); past the reserve each further child is kept by chance.
// Either way a child's share of light is divided by its chance: the same image on average, a little noisier there. While
// room is plenty nothing changes.
constexpr U32 PendingReserve = 16;
X2_HD inline float bounceThinning(U32 top, U32 planned) {
    const U32 room = top < PendingReserve ? PendingReserve - top : 1u;
    return planned > room ? float(room) / float(planned) : 1.0f;
}
X2_HD inline bool roomForChild(U32 top, float u, float& weight) {
    if (top >= MaxPending) return false;
    const float chance = float(MaxPending - top) / float(PendingReserve + 1);
    if (u >= chance) return false;
    weight = 1.0f / chance;
    return true;
}
// A ray's mask (rdl2 VisibilityType): CAMERA for a camera ray and its presence continuations (lobe type 0), else the
// mask of the lobe it samples (lobeTypeToRayMask).
X2_HD inline U32 rayMask(U32 lobeType) { return lobeType == 0 ? 1u : lights::lobeRayMask(lobeType); }
// A child's address in the sample's tree: a 64-bit mix of its parent's, the
// lobe and the sample (collisions within one sample's tree have probability
// about 2^-64 a pair; the radix tree of sample_stream.h cannot hold the
// multi-sampler's branching).
// A ray's direction footprint (Ray_getDirFootprint): log2 of the geometric mean of its direction differentials'
// lengths, which picks a textured light's mip level.
X2_HD inline float dirFootprint(V3 dDx, V3 dDy) {
    const float lx = sqrtf(surface::dot(dDx, dDx)), ly = sqrtf(surface::dot(dDy, dDy));
    return 0.5f / logf(2.0f) * logf(lx * ly);
}
X2_HD inline U64 childBranch(U64 parent, U32 a, U32 b) {
    U64 z = parent + 0x9e3779b97f4a7c15ull * (U64(a) * 0x10001u + U64(b) + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
}  // namespace paths


namespace paths {
// MOONRAY_MODO_TOON_TERMINATOR (modoShadowOrigin): a toon material's shadow ray toward a light near the terminator
// starts from the smooth surface (State::modoSmoothOffset), fully when the light is at or behind the face, fading out
// by the time it is 0.1 (in cosine) above it.
X2_HD inline V3 toonShadowOrigin(const shade::State& st, V3 P, V3 wi) {
    const V3 offset = st.modoSmoothOffset;
    if (shade::dot(offset, offset) == 0.0f || shade::dot(st.N, wi) <= 0.0f || shade::dot(offset, st.N) < 0.0f) return P;
    V3 Ng = st.Ng;
    if (shade::dot(st.N, Ng) < 0.0f) Ng = Ng * -1.0f;
    const float amount = fminf(fmaxf(1.0f - shade::dot(Ng, wi) / 0.1f, 0.0f), 1.0f);
    return P + offset * amount;
}
// A shadow ray toward a light (prepareOcclusionTestRays): its start raised to the light's min shadow distance, its
// end short of the light by its clear radius (not with a falloff, which fades occluded light instead) and capped
// by its max shadow distance.
X2_HD inline void lightShadowSpan(const lights::Light& L, float distance, float& near, float& far) {
    const float clear = L.clearRadiusFalloff > 0.0f ? 0.0f : L.clearRadius;
    near = fmaxf(near, L.minShadowDistance);
    far = fmaxf((distance - clear) * lights::HitEpsilonEnd, 2.0f * near);
    if (L.maxShadowDistance > 0.0f) far = fminf(far, L.maxShadowDistance);
}
// PathIntegrator::isRayOccluded, the volumes' and subsurface's shadow rays: the end, already short of the light, is
// shortened by the clear radius.
X2_HD inline void lightShadowSpanScalar(const lights::Light& L, float& near, float& far) {
    near = fmaxf(near, L.minShadowDistance);
    const float clear = L.clearRadiusFalloff > 0.0f ? 0.0f : L.clearRadius;
    far = fmaxf(far - clear, near);
    if (L.maxShadowDistance > 0.0f) far = fminf(far, L.maxShadowDistance);
}
// accumulateRayPresence: the presence of each surface a shadow ray passes on its way, gathered (or, below the
// presence quality's throughput, drawn as all or nothing) until it reaches presence_threshold or the ray's depth
// (the path's, counting on from it) reaches max_presence_depth. A texture not yet resident ends it (textureMissed).
template <class Backend>
X2_HD inline float rayPresence(const mixed::Config& c, Backend& backend, V3 org, V3 dir, float near, float far, U32 depth, bool stochastic,
                               Stream& s) {
    float total = 0.0f, t0 = near;
    for (;; ++depth) {
        const surface::Ray r{org, dir, V3{0, 0, 0}, V3{0, 0, 0}, t0, far, 2u};  // SHADOW
        const resume::Hit h = backend.trace(r);
        if (h.geom == ~0u) return total;
        if (!backend.castsShadow(h.geom, false)) {
            t0 = h.t + 1e-4f * fmaxf(h.t, 1.0f);
            if (t0 >= far) return total;
            --depth;
            continue;
        }
        float p = 1.0f;
        if (backend.presenceAt(h, r, p) != tex::Ready) return total;
        if (stochastic) {
            if (s.next() < p) return 1.0f;
        } else {
            total += (1.0f - total) * p;
        }
        if (!(depth < c.maxPresenceDepth && total < c.presenceThreshold)) return total;
        t0 = h.t + 1e-4f * fmaxf(h.t, 1.0f);
        if (t0 >= far) return total;
    }
}

// PathIntegrator::isRayOccluded (subsurface and volume shadow rays): with presence shadows, occluded past the presence
// threshold, else the light less the presence it passes.
template <class Backend>
X2_HD inline float scalarShadowKeep(const mixed::Config& c, Backend& backend, const lights::Light& L, V3 org, V3 dir, float near, float far,
                                    U32 depth, bool stochastic, Stream& s) {
    backend.shadowFor(L.index, backend.shadowReceiver);
    if (!L.presenceShadows) return backend.visible(org, dir, near, far) ? 1.0f : 0.0f;
    const float p = rayPresence(c, backend, org, dir, near, far, depth, stochastic, s);
    return p > c.presenceThreshold ? 0.0f : 1.0f - p;
}

// RayHandlers' calculateShadowFalloff: what an occluded shadow ray keeps of its light when it ends within the clear
// radius and its falloff (all of it within the radius, none beyond the falloff).
X2_HD inline float lightShadowKept(const lights::Light& L, float far) {
    if (L.clearRadiusFalloff == 0.0f || !(far < L.clearRadius + L.clearRadiusFalloff)) return 0.0f;
    if (far <= L.clearRadius) return 1.0f;
    const float t = (far - L.clearRadius) / L.clearRadiusFalloff;
    float w;
    switch (L.clearRadiusInterpolation) {
    case 0: w = t; break;
    case 1: w = t * t; break;
    case 2: w = 1.0f - (1.0f - t) * (1.0f - t); break;
    case 3: w = t * t * (3.0f - 2.0f * t); break;
    default: return 0.0f;
    }
    return 1.0f - w;
}

// The lights the camera sees (Scene::intersectVisibleLight on a miss, LightSet_intersect over the visible light set
// in PathIntegratorBundled on a hit): bounded lights, then distant ones, then environments, as MoonRay orders them;
// one of those the ray meets chosen uniformly, its radiance times their number. A miss meets them from the ray's
// origin (MoonRay's C++ lights); a hit from render space's origin, the camera, as far as the hit (the ISPC lights).
X2_HD inline U32 visiblePass(const lights::Light& l) { return l.type == lights::Env ? 2u : l.type == lights::Distant ? 1u : 0u; }
template <class Backend>
X2_HD inline V3 visibleLightRadiance(const mixed::Config& c, const sample::Identity& root, Backend& backend, V3 origin, V3 dir,
                                     float maxDistance, bool missed, float footprint, U32 segment, U32* hitLight = nullptr) {
    U32 hits = 0;
    for (U32 pass = 0; pass < 3; ++pass)
        for (U32 j = 0; j < c.lightCount; ++j) {
            const lights::Light& L = backend.light(j);
            lights::Hit lh;
            if (L.visibleInCamera && visiblePass(L) == pass && lights::intersect(L, origin, nullptr, dir, maxDistance, lh, missed, true)) ++hits;
        }
    if (!hits) return V3{0, 0, 0};
    sample::Identity id = root;
    id.branch = childBranch(0x5649534c49474854ull, segment, 0);
    Stream s(id, 0, 0);
    const float choice = s.next();
    const U32 chosen = U32(choice * float(hits)) < hits ? U32(choice * float(hits)) : hits - 1;
    for (U32 pass = 0, seen = 0; pass < 3; ++pass)
        for (U32 j = 0; j < c.lightCount; ++j) {
            const lights::Light& L = backend.light(j);
            lights::Hit lh;
            if (!L.visibleInCamera || visiblePass(L) != pass || !lights::intersect(L, origin, nullptr, dir, maxDistance, lh, missed, true)) continue;
            if (seen++ != chosen) continue;
            float pdf;
            const V3 Li = lights::eval(L, origin, dir, lh, pdf, footprint, true);
            if (hitLight) *hitLight = j;
            return shade::finite3(Li) ? Li * float(hits) : V3{0, 0, 0};
        }
    return V3{0, 0, 0};
}
}  // namespace paths

//----------------------------------------------------------------------------
// Volumes (PathIntegratorVolume.cc) over homogeneous BaseVolumes bound to closed meshes (volume_core.h).
namespace paths {
// collectVolumeIntervals: a ray's crossings past its start, in order, and the regions it starts in.
template <class Backend>
X2_CALL U32 volumeIntervals(Backend& backend, V3 org, V3 dir, float tnear, volume::Crossing* x, U32& regions, U32 mask) {
    U32 n = backend.volumeCrossings(org, dir, tnear, x, volume::MaxCrossings, mask);
    regions = volume::interiority(x, n);
    return n;
}
}  // namespace paths

// transmittance(): Beer's law over each homogeneous stretch of [tnear, tfar], the extinction scaled; nothing once it
// falls to the transmittance threshold, nor through a volume that reaches to infinity.
template <class Backend>
X2_CALL V3 volumeTransmittance(const mixed::Config& c, Backend& backend, paths::Stream& s, V3 org, V3 dir, float tnear, float tfar,
                               float scale, int depth = 1) {
    volume::Crossing x[volume::MaxCrossings];
    U32 regions;
    const U32 n = paths::volumeIntervals(backend, org, dir, tnear, x, regions, 2u);  // SHADOW
    if (backend.deferred()) return V3{1, 1, 1};  // a volume the GPU does not render: the CPU lane renders the sample (its data is not here)
    V3 tr{1, 1, 1};
    if (n == 0 && regions == 0) return tr;
    float tStart = tnear;
    auto segment = [&](float t0, float t1) X2_OUT_OF_LINE -> bool {
        if (!regions) return true;
        if (t1 >= volume::FarLimit) return false;
        float minFeatureSize;
        if (volume::stretchHomogeneous(backend.volumes(), regions, minFeatureSize)) {
            const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? s.next() : 0.0f;
            const V3 sigmaT = volume::evalSigmaT(backend.volumes(), regions, c.volumeOverlapMode, rnd, t1 - t0);
            tr = volume::times(tr, volume::expNeg(sigmaT, (t1 - t0) * scale));
        } else {
            // transmittanceSubinterval's ray marching: the optical thickness summed a shadow step at a time (each
            // volume's nearest voxel), nothing once it passes what the transmittance threshold leaves.
            const float tauThreshold = c.volumeTransmittanceThreshold == 0.0f ? 3.402823466e+38f
                                                                              : -logf(c.volumeTransmittanceThreshold / volume::lum(tr));
            const float u = s.next();
            const float stepSize = minFeatureSize * c.invVolumeShadowQuality * fmaxf(float(depth - 1), 1.0f);
            float t = t0 + stepSize * u;
            V3 tau{0, 0, 0};
            const volume::Grids grids = backend.volumeGrids();
            while (t < t1) {
                const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? s.next() : 0.0f;
                tau = volume::plus(tau, volume::evalSigmaTAt(backend.volumes(), grids, regions, c.volumeOverlapMode, rnd,
                                                             surface::add(org, surface::mul(dir, t)), -1.0f));
                if (volume::lum(volume::scaled(tau, stepSize * scale)) > tauThreshold) return false;
                t += stepSize;
            }
            tr = volume::times(tr, volume::expNeg(tau, stepSize * scale));
        }
        return volume::lum(tr) > c.volumeTransmittanceThreshold;
    };
    for (U32 i = 0; i < n; ++i) {
        if (x[i].t > tfar) break;
        if (x[i].t - tStart > volume::Epsilon) {
            if (!segment(tStart, x[i].t)) return V3{0, 0, 0};
            tStart = x[i].t;
        }
        volume::apply(regions, x[i]);
    }
    if (tfar - tStart > volume::Epsilon && !segment(tStart, tfar)) return V3{0, 0, 0};
    return tr;
}

// computeEmissiveVolumeIntegral: an emissive volume's light along a ray to tEnd (or the first surface a shadow ray
// meets), through the shadow-casting volumes on its way, to where the ray last leaves that volume: its homogeneous
// stretches by Beer's law (the emission taken at a stretch's start), its heterogeneous ones a step at a time.
template <class Backend>
X2_CALL V3 emissiveVolumeIntegral(const mixed::Config& c, Backend& backend, paths::Stream& s, V3 org, V3 dir, float tnear, float tEnd,
                                   U32 emissive, int depth) {
    {
        const surface::Ray hard{org, dir, V3{0, 0, 0}, V3{0, 0, 0}, tnear, tEnd, 2u, V3{0, 0, 0}, V3{0, 0, 0}};
        const resume::Hit h = backend.trace(hard);
        if (h.geom != ~0u && h.t < tEnd) tEnd = h.t;
    }
    volume::Crossing x[volume::MaxCrossings];
    U32 regions;
    U32 n = paths::volumeIntervals(backend, org, dir, tnear, x, regions, 0xffu);
    if (backend.deferred()) return V3{0, 0, 0};  // a volume the GPU does not render: the CPU lane renders the sample (its data is not here)
    bool visited = (regions >> emissive & 1u) != 0;
    for (U32 i = 0; i < n; ++i)
        if (x[i].volume == emissive) visited = true;
    if (!visited) return V3{0, 0, 0};
    for (int i = int(n) - 1; i >= 0; --i)
        if (x[i].volume == emissive) {
            if (x[i].flags & volume::Exit) {
                n = U32(i) + 1;
                tEnd = x[i].t;
            }
            break;
        }
    const volume::Shader* shaders = backend.volumes();
    const volume::Grids grids = backend.volumeGrids();
    U32 casters = 0;
    for (U32 v = 0; v < c.volumeCount && v < volume::MaxVolumes; ++v)
        if (shaders[v].visibility & 2u) casters |= 1u << v;
    V3 LVe{0, 0, 0}, tr{1, 1, 1};
    bool reached = false;
    float tStart = tnear;
    auto at = [&](float t) { return surface::add(org, surface::mul(dir, t)); };
    auto sub = [&](float t0, float t1) {
        if (!regions) return;
        const U32 trRegions = regions & casters;
        const bool emitting = (regions >> emissive & 1u) != 0;
        float minFeatureSize;
        if (volume::stretchHomogeneous(shaders, regions, minFeatureSize)) {
            const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? s.next() : 0.0f;
            const V3 sigmaT = trRegions ? volume::evalSigmaT(shaders, trRegions, c.volumeOverlapMode, rnd, t1 - t0) : V3{0, 0, 0};
            const V3 seg = volume::expNeg(sigmaT, t1 - t0);
            if (emitting) {
                const V3 e = volume::localAt(shaders[emissive], grids, at(t0), t1 - t0, volume::InterpolationPoint).emission;
                const V3 w{volume::isZero(sigmaT.x) ? 0.0f : (1.0f - seg.x) / sigmaT.x, volume::isZero(sigmaT.y) ? 0.0f : (1.0f - seg.y) / sigmaT.y,
                           volume::isZero(sigmaT.z) ? 0.0f : (1.0f - seg.z) / sigmaT.z};
                LVe = volume::plus(LVe, volume::times(volume::times(tr, e), w));
            }
            tr = volume::times(tr, seg);
            return;
        }
        const float u = s.next();
        const float stepSize = minFeatureSize * c.invVolumeQuality * fmaxf(float(depth - 1), 1.0f);
        float stepStart = t0, t = t0 + stepSize * u;
        auto stepAt = [&](float tt) {
            const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? s.next() : 0.0f;
            const V3 sigmaT = trRegions ? volume::evalSigmaTAt(shaders, grids, trRegions, c.volumeOverlapMode, rnd, at(tt), -1.0f) : V3{0, 0, 0};
            const float delta = tt - stepStart;
            if (emitting)
                LVe = volume::plus(LVe, volume::scaled(volume::times(tr, volume::localAt(shaders[emissive], grids, at(tt), -1.0f,
                                                                                         volume::InterpolationPoint).emission), delta));
            tr = volume::times(tr, volume::expNeg(sigmaT, delta));
        };
        while (t < t1) {
            if (volume::lum(tr) < c.volumeTransmittanceThreshold) {
                reached = true;
                return;
            }
            stepAt(t);
            stepStart = t;
            t += stepSize;
        }
        if (volume::lum(tr) < c.volumeTransmittanceThreshold) {
            reached = true;
            return;
        }
        if (t - t1 > volume::Epsilon) stepAt(t1);
    };
    for (U32 i = 0; i < n; ++i) {
        if (x[i].t > tEnd || reached) break;
        if (x[i].t - tStart > volume::Epsilon) {
            sub(tStart, x[i].t);
            tStart = x[i].t;
        }
        volume::apply(regions, x[i]);
    }
    if (tEnd - tStart > volume::Epsilon && !reached) sub(tStart, tEnd);
    return LVe;
}

// estimateInScatteringSourceTermDirect: a light sample from a point in a volume, unoccluded, through the volumes
// on its way (their extinction scaled), times the phase function.
template <class Backend>
X2_CALL V3 volumeSourceTerm(const mixed::Config& c, Backend& backend, paths::Stream& s, const lights::Light& L, V3 P, V3 rayDir, float g,
                            const float ul[3], float scale) {
    V3 wi;
    lights::Hit lh;
    float r[3] = {ul[0], ul[1], ul[2]};
    if (!lights::sample(L, P, nullptr, r, wi, lh)) return V3{0, 0, 0};
    float pdfL;
    const V3 Li = lights::eval(L, P, wi, lh, pdfL);
    if (shade::isSampleInvalid(Li, pdfL) || !volume::finitef(pdfL)) return V3{0, 0, 0};
    backend.shadowFor(L.index, ~0u);  // a volume's shadow rays leave no mesh
    float near = volume::HitEpsilonStart, far = lights::HitEpsilonEnd * lh.distance;
    paths::lightShadowSpanScalar(L, near, far);
    const float keep = paths::scalarShadowKeep(c, backend, L, P, wi, near, far, 0, false, s);
    if (keep <= 0.0f) return V3{0, 0, 0};
    const V3 tr = volumeTransmittance(c, backend, s, P, wi, 0.0f, far, scale);
    const float ph = volume::phaseEval(g, V3{-rayDir.x, -rayDir.y, -rayDir.z}, wi);
    return volume::scaled(volume::times(tr, Li), ph / pdfL * keep);
}

// computeRadianceVolume over [tnear, tfar] of a ray: its homogeneous stretches (decoupledRayMarching), their single
// scattering (integrateVolumeScatteringDirect: per light, equi-angular and distance samples with MIS) and, with a
// max volume depth above one, the multiple scattering approximation (approximateVolumeMultipleScattering). The
// path's throughput takes the transmittance; its volume depth goes up when it scattered.
// MOONRAY_MODO_CRYPTO_VOLUMES (modoRecordCryptoVolumes): each volume's share of what a camera ray's march covered, and
// where on average, and what still shows through.
struct VolumeCryptoShare {
    U32 volume;
    float coverage, depth;
};
struct VolumeCryptoShares {
    VolumeCryptoShare share[16];
    U32 count = 0;
    float shown = 1.0f;
    bool marched = false;
};
// The light path expressions through the volumes (volumeEventTransition, then a light or an emission event, from the
// path's state pv.lpe): the sample's AOVs.
struct VolumeLpe {
    const aov::Tables* t = nullptr;
    float* aovs = nullptr;
    float* depths = nullptr;
};
// A march along a ray through the volumes (decoupledRayMarching's steps), from its start each time it is run: each step
// to emit(step, index), which stops it by returning false. The CPU lane keeps the steps in a table; the CUDA lane keeps
// none (a VDB cloud's hundreds or thousands do not fit a thread) and finds a step it needs by marching again, the
// marching stream started the same, so the very same steps (ReplaySteps, GPU_VDB).
template <class Backend>
struct VolumeMarch {
    const mixed::Config& c;
    Backend& backend;
    const volume::Shader* shaders;
    volume::Grids grids;
    const volume::Crossing* x;
    U32 n, regionsAtStart;
    V3 org, dir;
    float tnear, tfar;
    U32 depth;
    bool cutouts, sampleEmission;
    paths::Stream start;  // the marching stream as a march begins
    U32 limit;     // a march's steps are fewer than this (the CPU lane's table, volume::MarchLimit)
    // What a march leaves: the transmittances, each volume's emission, the volumes met, the steps taken.
    V3 vt, vth, vta, vtm;
    V3 emitted[volume::MaxVolumes];
    U32 visited, count;
    bool reachedThreshold, stopped;

    template <class Emit>
    X2_HD void run(Emit& emit) {
        paths::Stream marching = start;
        U32 regions = regionsAtStart;
        vt = vth = vta = V3{1, 1, 1};
        vtm = V3{0, 0, 0};
        for (U32 v = 0; v < volume::MaxVolumes; ++v) emitted[v] = V3{0, 0, 0};
        visited = count = 0;
        reachedThreshold = stopped = false;
        auto update = [&](V3 tr, V3 trh) {
            vt = volume::times(vt, tr);
            vth = volume::times(vth, trh);
            const V3 oldMin = vtm;
            vtm = volume::plus(volume::times(trh, vtm), volume::times(V3{1.0f - trh.x, 1.0f - trh.y, 1.0f - trh.z}, vta));
            vta = volume::plus(volume::times(tr, vta), volume::times(V3{1.0f - tr.x, 1.0f - tr.y, 1.0f - tr.z}, oldMin));
        };
        auto take = [&](const volume::Step& s) {
            if (!emit(s, count)) stopped = true;
            ++count;
        };
        auto march = [&](float t0, float t1) {
            if (!regions || t1 >= volume::FarLimit || count + 1 >= limit || reachedThreshold || stopped) return;
            visited |= regions;
            float minFeatureSize;
            if (volume::stretchHomogeneous(shaders, regions, minFeatureSize)) {
                const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? marching.next() : 0.0f;
                V3 emissions[volume::MaxVolumes];
                U32 chosen = volume::MaxVolumes;
                const volume::Coefficients k = volume::evalShadersAt(shaders, grids, regions, c.volumeOverlapMode, rnd, org, t1 - t0, volume::NoGrid,
                                                                     sampleEmission ? emissions : nullptr, &chosen, cutouts);
                take(volume::Step{k.sigmaT, k.sigmaS, vt, k.g, t0, t1 - t0, k.lightSet, k.sigmaTh, k.sigmaSh, vth, chosen, k.surfaceThreshold});
                const V3 tr = volume::expNeg(k.sigmaT, t1 - t0), trh = volume::expNeg(k.sigmaTh, t1 - t0);
                if (sampleEmission) {
                    // emission * e^(-sigmaT t) over the stretch: (1 - tr) / sigmaT (per channel, none without extinction).
                    auto integral = [&](V3 e) {
                        const V3 w{volume::isZero(k.sigmaT.x) ? 0.0f : (1.0f - tr.x) / k.sigmaT.x, volume::isZero(k.sigmaT.y) ? 0.0f : (1.0f - tr.y) / k.sigmaT.y,
                                   volume::isZero(k.sigmaT.z) ? 0.0f : (1.0f - tr.z) / k.sigmaT.z};
                        return volume::times(volume::times(volume::times(vt, vth), e), w);
                    };
                    if (c.volumeOverlapMode == volume::OverlapSum) {
                        for (U32 v = 0; v < volume::MaxVolumes; ++v)
                            if (regions >> v & 1) emitted[v] = volume::plus(emitted[v], integral(emissions[v]));
                    } else if (chosen < volume::MaxVolumes) {
                        emitted[chosen] = volume::plus(emitted[chosen], integral(emissions[0]));
                    }
                }
                update(tr, trh);
                return;
            }
            // A heterogeneous stretch: steps of the least feature size over the volume quality (wider deeper in the
            // path), the first at a random place in the first step, each step's coefficients at its end.
            const float u = marching.next();
            const float stepSize = minFeatureSize * c.invVolumeQuality * fmaxf(float(int(depth) - 1), 1.0f);
            float stepStart = t0, t = t0 + stepSize * u;
            auto step = [&](float at) {
                const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? marching.next() : 0.0f;
                V3 emissions[volume::MaxVolumes];
                U32 chosen = volume::MaxVolumes;
                const volume::Coefficients k = volume::evalShadersAt(shaders, grids, regions, c.volumeOverlapMode, rnd, surface::add(org, surface::mul(dir, at)),
                                                                     -1.0f, volume::NoGrid, sampleEmission ? emissions : nullptr, &chosen, cutouts);
                const float delta = at - stepStart;
                take(volume::Step{k.sigmaT, k.sigmaS, vt, k.g, stepStart, delta, k.lightSet, k.sigmaTh, k.sigmaSh, vth, chosen, k.surfaceThreshold});
                const V3 te = volume::times(vt, vth);
                if (sampleEmission) {
                    if (c.volumeOverlapMode == volume::OverlapSum) {
                        for (U32 v = 0; v < volume::MaxVolumes; ++v)
                            if (regions >> v & 1) emitted[v] = volume::plus(emitted[v], volume::scaled(volume::times(te, emissions[v]), delta));
                    } else if (chosen < volume::MaxVolumes) {
                        emitted[chosen] = volume::plus(emitted[chosen], volume::scaled(volume::times(te, emissions[0]), delta));
                    }
                }
                update(volume::expNeg(k.sigmaT, delta), volume::expNeg(k.sigmaTh, delta));
            };
            while (t < t1) {
                if (volume::lum(vta) < c.volumeTransmittanceThreshold) {
                    reachedThreshold = true;
                    return;
                }
                step(t);
                if (stopped) return;
                stepStart = t;
                t += stepSize;
                if (count + 1 >= limit) return;
            }
            if (volume::lum(vta) < c.volumeTransmittanceThreshold) {
                reachedThreshold = true;
                return;
            }
            if (t - t1 > volume::Epsilon) step(t1);
        };
        float tStart = tnear;
        for (U32 i = 0; i < n; ++i) {
            if (x[i].t > tfar || reachedThreshold || stopped) break;
            if (x[i].t - tStart > volume::Epsilon) {
                march(tStart, x[i].t);
                tStart = x[i].t;
            }
            volume::apply(regions, x[i]);
        }
        if (tfar - tStart > volume::Epsilon && !reachedThreshold && !stopped) march(tStart, tfar);
    }
};
// A step's weight for distance sampling: its scattering, through what lies before it.
X2_HD inline float stepWeight(const volume::Step& s) {
    return volume::lum(volume::times(volume::scaled(s.sigmaS, s.delta), s.tr)) + volume::lum(volume::times(volume::scaled(s.sigmaSh, s.delta), s.trH));
}
// The steps of a march as the light sampling asks them: by distance sampling (tabulate and sampleDiscrete's choice, its
// pdf and the number remapped into the step) and by where along the ray (findInterval, with the step's pdfDiscrete).
// The CPU lane's from its table (TableSteps); the CUDA lane's by marching again (ReplaySteps), its distribution's cdf made
// as tabulate makes it: the weights summed in order, each partial sum no more than the total, the last the total; even
// steps where none weighs.
struct TableSteps {
    const volume::Step* steps;
    const volume::Distribution* dist;
    U32 count;
    X2_HD volume::Step pick(float r, float& pdf, float& remapped) const { return steps[volume::sampleDiscrete(*dist, r, pdf, remapped)]; }
    X2_HD volume::Step at(float t, float& pdf) const {
        U32 k = 0;
        while (k + 1 < count && steps[k + 1].tStart <= t) ++k;
        pdf = volume::pdfDiscrete(*dist, k);
        return steps[k];
    }
};
template <class March>
struct ReplaySteps {
    March& m;
    U32 count;
    float total;  // the weights' sum; one where none weighs
    bool weightless;
    X2_HD float cdf(float& partial, const volume::Step& s, U32 index) const {
        if (index + 1 >= count) return total;
        if (weightless) return float(index + 1) / float(count);
        partial += stepWeight(s);
        if (partial > total) partial = total;
        return partial;
    }
    X2_HD volume::Step pick(float r, float& pdf, float& remapped) const {
        const float rw = r * total;
        float partial = 0.0f, prev = 0.0f, found = total, before = 0.0f;
        volume::Step chosen{};
        auto emit = [&](const volume::Step& s, U32 i) {
            const float here = cdf(partial, s, i);
            chosen = s;
            if (here >= rw || i + 1 >= count) {
                found = here, before = prev;
                return false;
            }
            prev = here;
            return true;
        };
        m.run(emit);
        const float value = found - before;
        pdf = value / total;
        remapped = (rw - before) / value;
        return chosen;
    }
    X2_HD volume::Step at(float t, float& pdf) const {
        float partial = 0.0f, here = 0.0f, kept = 0.0f, keptPrev = 0.0f;
        volume::Step chosen{};
        auto emit = [&](const volume::Step& s, U32 i) {
            if (i > 0 && s.tStart > t) return false;
            const float before = here;
            here = cdf(partial, s, i);
            chosen = s, kept = here, keptPrev = i > 0 ? before : 0.0f;
            return i + 1 < count;
        };
        m.run(emit);
        pdf = (kept - keptPrev) / total;
        return chosen;
    }
};
template <class Backend>
X2_CALL V3 volumeRadiance(const mixed::Config& c, const sample::Identity& root, Backend& backend, paths::Pending& pv, V3 org, V3 dir,
                          float tnear, float tfar, V3* alphaTransmittance = nullptr, V3* minTransmittance = nullptr,
                          VolumeCryptoShares* cryptoShares = nullptr, const VolumeLpe* lpe = nullptr, float* surfaceT = nullptr) {
    using namespace paths;
    V3 radiance{0, 0, 0};
    auto lpeAdd = [&](V3 contribution, int state) {
        if (lpe) aov::light(*lpe->t, lpe->aovs, lpe->depths, contribution, nullptr, 0, state);
    };
    volume::Crossing x[volume::MaxCrossings];
    U32 regions;
    const U32 rayMask = paths::rayMask(pv.lobeType);
    const U32 n = volumeIntervals(backend, org, dir, tnear, x, regions, rayMask);
    if (backend.deferred()) return radiance;  // a volume the GPU does not render: the CPU lane renders the sample (its data is not here)
    if (n == 0 && regions == 0) return radiance;
    const volume::Shader* shaders = backend.volumes();
    const U64 volumeBranch = childBranch(pv.branch, 0xfffc, pv.volumeDepth);
    sample::Identity vid = root;
    vid.branch = childBranch(volumeBranch, 0, 0);
    // VolumeTransmittance: through the volumes (E), through the holdouts (H), the alpha's and its minimum; camera rays
    // hold cutouts out (decoupledRayMarching's depth == 0). A camera ray's volume emission (decoupledRayMarching with
    // perVolumeLVe): each volume's, through what lies before.
    const bool cutouts = pv.depth == 0, sampleEmission = pv.depth == 0;
    const volume::Grids grids = backend.volumeGrids();
    VolumeMarch<Backend> march{c, backend, shaders, grids, x, n, regions, org, dir, tnear, tfar, pv.depth, cutouts, sampleEmission,
                               Stream(vid, pv.depth, 0), volume::MarchLimit};
    // The march once, what is needed of its steps taken as they come: the volume's "surface" (where the accumulated
    // transmittance falls below one minus the step's threshold), each step's coverage of what showed through by its
    // chosen volume at its middle (Cryptomatte), the weights' sum, the first step's start and the last one's end. The CPU
    // lane keeps the steps too.
#if defined(__CUDACC__)
    volume::Step* steps = nullptr;
#else
    volume::MarchSpace& space = volume::marchSpace();
    volume::Step* steps = space.steps;
#endif
    float surfaceAccumulated = 1.0f, cryptoShown = 1.0f, weights = 0.0f, firstStart = 0.0f, lastEnd = 0.0f;
    U32 lastVolume = volume::MaxVolumes;
    bool surfaceFound = false;
    auto cryptoAdd = [&](U32 v, float coverage, float depth) {
        if (coverage <= 0.0f) return;
        VolumeCryptoShares& cs = *cryptoShares;
        for (U32 i = 0; i < cs.count; ++i)
            if (cs.share[i].volume == v) {
                cs.share[i].coverage += coverage;
                cs.share[i].depth += coverage * depth;
                return;
            }
        if (cs.count < 16) cs.share[cs.count++] = VolumeCryptoShare{v, coverage, coverage * depth};
    };
    auto first = [&](const volume::Step& s, U32 i) {
        if (steps) steps[i] = s;
        if (i == 0) firstStart = s.tStart;
        lastEnd = s.tStart + s.delta;
        lastVolume = s.volume;
        weights += stepWeight(s);
        if (surfaceT && !surfaceFound) {
            const float sigmaT = volume::lum(s.sigmaT);
            const float stepTr = expf(-sigmaT * s.delta);
            const float desired = 1.0f - s.surfaceThreshold;
            if (desired > surfaceAccumulated * stepTr) {
                *surfaceT = s.tStart + -logf(desired / surfaceAccumulated) / sigmaT;
                surfaceFound = true;
            } else {
                surfaceAccumulated *= stepTr;
            }
        }
        if (cryptoShares) {
            const float covered = cryptoShown * (1.0f - expf(-volume::lum(s.sigmaT) * s.delta));
            cryptoAdd(s.volume, covered, s.tStart + 0.5f * s.delta);
            cryptoShown -= covered;
        }
        return true;
    };
    march.run(first);
    const U32 count = march.count;
    if (count == 0) return radiance;
    V3 vt = march.vt, vth = march.vth, vta = march.vta, vtm = march.vtm;
    const bool reachedThreshold = march.reachedThreshold;
    const U32 visited = march.visited;
    // Past the transmittance threshold the volume is opaque, and its alpha the holdouts' minimum.
    if (reachedThreshold) {
        vt = V3{0, 0, 0};
        vta = vtm;
    }
    if (alphaTransmittance) *alphaTransmittance = vta;
    if (minTransmittance) *minTransmittance = vtm;
    // Each step's coverage, taken above; past the threshold the last volume covers the rest.
    if (cryptoShares) {
        cryptoShares->marched = true;
        if (reachedThreshold) {
            cryptoAdd(lastVolume, cryptoShown, lastEnd);
            cryptoShown = 0.0f;
        }
        cryptoShares->shown = fminf(fmaxf(cryptoShown, 0.0f), 1.0f);
    }
    const V3 T = pv.T;
    // The camera ray's volume emission, each volume's through what lay before it on the path.
    if (sampleEmission)
        for (U32 v = 0; v < volume::MaxVolumes; ++v) {
            if (!(visited >> v & 1)) continue;
            const V3 emission = volume::times(T, march.emitted[v]);
            if (volume::black(emission)) continue;
            radiance = volume::plus(radiance, emission);
            if (lpe) lpeAdd(emission, aov::lpeNext(*lpe->t, pv.lpe, aov::LpeEmission, shaders[v].lpeLabel));
        }
    // The path's state at a volume scattering event (volumeEventTransition).
    const int volumeState = lpe ? aov::lpeVolume(*lpe->t, pv.lpe) : -1;
    if (pv.volumeDepth < c.maxVolumeDepth) {
#if defined(__CUDACC__)
        const ReplaySteps<VolumeMarch<Backend>> lookup{march, count, weights != 0.0f ? weights : 1.0f, weights == 0.0f};
#else
        // The steps' weights for distance sampling: their scattering, through what lies before them.
        float* density = space.density;
        for (U32 i = 0; i < count; ++i) density[i] = stepWeight(steps[i]);
        volume::Distribution& dist = space.dist;
        volume::tabulate(dist, density, count);
        const TableSteps lookup{steps, &dist, count};
#endif
        const float rayEnd = lastEnd;
        // integrateVolumeScatteringDirect
        const bool highQuality = pv.nonMirrorDepth == 0;
        const U32 samplesPerLight = highQuality ? c.lightSamples : 1, scatterCount = highQuality ? c.volumeIlluminationSamples : 1;
        if (scatterCount > 0) {
            const float invN = 1.0f / float(samplesPerLight * scatterCount);
            for (U32 li = 0; li < c.lightCount; ++li) {
                const lights::Light& L = backend.light(li);
                const bool equiAngular = volume::bounded(L);
                V3 direct{0, 0, 0};  // LDirect: the light's single scattering, one LPE sum
                for (U32 ls = 0; ls < samplesPerLight; ++ls) {
                    for (U32 ts = 0; ts < scatterCount; ++ts) {
                        sample::Identity sid = root;
                        sid.branch = childBranch(volumeBranch, 1 + li, ls * scatterCount + ts);
                        Stream ss(sid, pv.depth, 0);
                        float D = 0.0f, thetaA = 0.0f, thetaB = 0.0f, offset = 0.0f;
                        if (equiAngular) {
                            // equiAngularVolumeScattering: about a pivot on the light (its sample reused at the point)
                            const float ue = ss.next();
                            const float uel[3] = {ss.next(), ss.next(), ss.next()};
                            const V3 pivot = volume::pivot(L, uel);
                            offset = surface::dot(surface::add(pivot, surface::mul(org, -1.0f)), dir);
                            const V3 foot = surface::add(org, surface::mul(dir, offset));
                            D = shade::length(surface::add(pivot, surface::mul(foot, -1.0f)));
                            thetaA = atan2f(firstStart - offset, D);
                            thetaB = atan2f(rayEnd - offset, D);
                            const float te = volume::sampleEquiAngular(ue, D, thetaA, thetaB);
                            const float pdfTe = volume::pdfEquiAngular(te, D, thetaA, thetaB);
                            if (pdfTe != 0.0f && volume::finitef(pdfTe)) {
                                const float tePrime = te + offset;
                                float pdfK;  // findInterval: the last step starting at or before tePrime, and its pdfDiscrete
                                const volume::Step vp = lookup.at(tePrime, pdfK);
                                if (!volume::black(vp.sigmaS) && backend.inSet(vp.lightSet, li)) {
                                    const V3 trScatter = volume::times(volume::times(vp.tr, vp.trH),
                                                                       volume::expNeg(volume::plus(vp.sigmaT, vp.sigmaTh), tePrime - vp.tStart));
                                    float mis = 1.0f;
                                    const float pdfTd = pdfK *
                                                        volume::pdfDistanceExponential(tePrime, volume::lum(vp.sigmaT), vp.tStart,
                                                                                       vp.tStart + vp.delta);
                                    if (volume::finitef(pdfTd)) mis = lights::powerHeuristic(pdfTe, pdfTd);
                                    const V3 Ls =
                                        volumeSourceTerm(c, backend, ss, L, surface::add(org, surface::mul(dir, tePrime)), dir, vp.g, uel, 1.0f);
                                    direct = volume::plus(direct, volume::scaled(volume::times(volume::times(T, vp.sigmaS), volume::times(trScatter, Ls)),
                                                                                 invN * mis / pdfTe));
                                }
                            }
                        }
                        // distanceVolumeScatteringDirect
                        const float ud = ss.next();
                        const float udl[3] = {ss.next(), ss.next(), ss.next()};
                        float pdfDensity, udRemapped;
                        const volume::Step vp = lookup.pick(ud, pdfDensity, udRemapped);
                        if (pdfDensity == 0.0f) continue;
                        if (volume::black(vp.sigmaS) || !backend.inSet(vp.lightSet, li)) continue;
                        const float sigmaT = volume::lum(vp.sigmaT), t0 = vp.tStart, t1 = vp.tStart + vp.delta;
                        const float td = volume::sampleDistanceExponential(udRemapped, sigmaT, t0, t1);
                        const float pdfTd = pdfDensity * volume::pdfDistanceExponential(td, sigmaT, t0, t1);
                        if (pdfTd == 0.0f || !volume::finitef(pdfTd)) continue;
                        const V3 trScatter = volume::times(volume::times(vp.tr, vp.trH), volume::expNeg(volume::plus(vp.sigmaT, vp.sigmaTh), td - t0));
                        float mis = 1.0f;
                        if (equiAngular) {
                            const float pdfTe = volume::pdfEquiAngular(td - offset, D, thetaA, thetaB);
                            if (volume::finitef(pdfTe)) mis = lights::powerHeuristic(pdfTd, pdfTe);
                        }
                        const V3 Ls = volumeSourceTerm(c, backend, ss, L, surface::add(org, surface::mul(dir, td)), dir, vp.g, udl, 1.0f);
                        direct = volume::plus(direct, volume::scaled(volume::times(volume::times(T, vp.sigmaS), volume::times(trScatter, Ls)),
                                                                     invN * mis / pdfTd));
                    }
                }
                radiance = volume::plus(radiance, direct);
                if (lpe) lpeAdd(direct, aov::lpeNext(*lpe->t, volumeState, aov::LpeLight, L.lpeLabel));
            }
        }
#if !defined(__CUDACC__)
        // Emissive volumes lighting the steps (computeRadianceEmissiveRegionsVolumes): equi-angular samples about a
        // place in each region its geometry lets light volumes, each through its emissive volume integral.
        {
            const volume::Regions er = backend.emissiveRegions();
            const bool hq = pv.nonMirrorDepth == 0;
            const U32 nls = hq ? c.lightSamples : 1, nss = hq ? c.volumeIlluminationSamples : 1;
            const float invN = 1.0f / float(nls * nss);
            const float rayStart = steps[0].tStart, rayStop = steps[count - 1].tStart + steps[count - 1].delta;
            for (U32 ri = 0; ri < er.count && nss > 0; ++ri) {
                const volume::Region& R = er.regions[ri];
                if (!(R.visibility & 0x100u)) continue;
                V3 regionLight{0, 0, 0};  // the region's in-scattering, one LPE sum
                for (U32 k = 0; k < nls * nss; ++k) {
                    sample::Identity eid = root;
                    eid.branch = childBranch(volumeBranch, 0xfe00u + ri, k);
                    Stream es(eid, pv.depth, 0);
                    const float u1 = es.next(), u2 = es.next(), u3 = es.next();
                    const V3 pivot = volume::regionPoint(R, er.pool, u1, u2, u3);
                    const float ue = es.next();
                    const float offset = surface::dot(surface::add(pivot, surface::mul(org, -1.0f)), dir);
                    const V3 foot = surface::add(org, surface::mul(dir, offset));
                    const float D = shade::length(surface::add(pivot, surface::mul(foot, -1.0f)));
                    const float thetaA = atan2f(rayStart - offset, D), thetaB = atan2f(rayStop - offset, D);
                    const float te = volume::sampleEquiAngular(ue, D, thetaA, thetaB);
                    const float pdfTe = volume::pdfEquiAngular(te, D, thetaA, thetaB);
                    if (pdfTe == 0.0f || !volume::finitef(pdfTe)) continue;
                    const float tePrime = te + offset;
                    U32 kk = 0;  // findInterval over the steps' starts and the march's end
                    while (kk + 1 < count && steps[kk + 1].tStart <= tePrime) ++kk;
                    const volume::Step& vp = steps[kk];
                    if (volume::black(vp.sigmaS)) continue;
                    const V3 trScatter = volume::times(volume::times(vp.tr, vp.trH),
                                                       volume::expNeg(volume::plus(vp.sigmaT, vp.sigmaTh), tePrime - vp.tStart));
                    const V3 scatterPoint = surface::add(org, surface::mul(dir, tePrime));
                    V3 wi = surface::add(pivot, surface::mul(scatterPoint, -1.0f));
                    wi = surface::mul(wi, 1.0f / shade::length(wi));
                    float tEnd;
                    const float pdfWi = volume::regionPdf(R, er.pool, scatterPoint, wi, tEnd);
                    if (pdfWi == 0.0f || !volume::finitef(pdfWi)) continue;
                    const V3 Le = emissiveVolumeIntegral(c, backend, es, scatterPoint, wi, 0.0f, tEnd, R.volume, int(pv.depth) + 1);
                    const float ph = volume::phaseEval(vp.g, V3{-dir.x, -dir.y, -dir.z}, wi);
                    regionLight = volume::plus(regionLight, volume::scaled(volume::times(volume::times(T, vp.sigmaS), volume::times(trScatter, Le)),
                                                                           invN * ph / (pdfTe * pdfWi)));
                }
                radiance = volume::plus(radiance, regionLight);
                if (lpe) lpeAdd(regionLight, aov::lpeNext(*lpe->t, volumeState, aov::LpeEmission, shaders[R.volume].lpeLabel));
            }
        }
#endif
        ++pv.volumeDepth;
        // approximateVolumeMultipleScattering (Wrenninge 2015): scatter events along the ray, then free paths from
        // them with the extinction, contribution and anisotropy shrinking each order.
        if (pv.volumeDepth < c.maxVolumeDepth && c.volumeIlluminationSamples > 0) {
            const U32 scatterSamples = c.volumeIlluminationSamples;
            const float invN = 1.0f / float(scatterSamples);
            for (U32 i = 0; i < scatterSamples; ++i) {
                sample::Identity mid = root;
                mid.branch = childBranch(volumeBranch, 0xffff, i);
                Stream ms(mid, pv.depth, 0);
                float pdfDensity, utRemapped;
                const volume::Step vp = lookup.pick(ms.next(), pdfDensity, utRemapped);
                float sigmaT = volume::lum(volume::plus(vp.sigmaT, vp.sigmaTh));
                const float t0 = vp.tStart, t1 = vp.tStart + vp.delta;
                const float td = volume::sampleDistanceExponential(utRemapped, sigmaT, t0, t1);
                const float pdfT = pdfDensity * volume::pdfDistanceExponential(td, sigmaT, t0, t1);
                if (pdfT == 0.0f || !volume::finitef(pdfT)) continue;
                const V3 trScatter = volume::times(volume::times(vp.tr, vp.trH), volume::expNeg(volume::plus(vp.sigmaT, vp.sigmaTh), td - t0));
                V3 scatterPoint = surface::add(org, surface::mul(dir, td));
                V3 throughput = volume::scaled(volume::times(volume::times(T, vp.sigmaS), trScatter), 1.0f / pdfT);
                const float up0 = ms.next(), up1 = ms.next();
                V3 scatterDir = volume::phaseSample(vp.g, V3{-dir.x, -dir.y, -dir.z}, up0, up1);
                int scatterState = volumeState;  // a volume event at the first scatter, another at each order's
                for (U32 depth = pv.volumeDepth; depth < c.maxVolumeDepth; ++depth) {
                    if (volume::lum(throughput) < c.volumeTransmittanceThreshold) break;
                    const float ai = powf(c.volumeAttenuationFactor, float(depth)), bi = powf(c.volumeContributionFactor, float(depth)),
                                ci = powf(c.volumePhaseAttenuationFactor, float(depth));
                    const float tFreePath = -logf(1.0f - ms.next()) / sigmaT;
                    volume::Crossing y[volume::MaxCrossings];
                    U32 here;
                    const U32 m = volumeIntervals(backend, scatterPoint, scatterDir, volume::Epsilon, y, here, rayMask);
                    if (backend.deferred()) break;  // a volume the GPU does not render: the CPU lane renders the sample (its data is not here)
                    for (U32 j = 0; j < m && y[j].t <= tFreePath; ++j) volume::apply(here, y[j]);
                    const V3 rayDir = scatterDir;
                    scatterPoint = surface::add(scatterPoint, surface::mul(scatterDir, tFreePath));
                    if (!here) break;
                    const float rnd = c.volumeOverlapMode == volume::OverlapRnd ? ms.next() : 0.0f;
                    const volume::Coefficients kk =
                        volume::evalShadersAt(shaders, grids, here, c.volumeOverlapMode, rnd, scatterPoint, -1.0f, volume::NoGrid, nullptr, nullptr);
                    const float g = kk.g * ci;
                    sigmaT = volume::lum(kk.sigmaT);
                    if (volume::isZero(sigmaT)) break;
                    throughput = volume::scaled(volume::times(throughput, kk.sigmaS), 1.0f / sigmaT);
                    if (lpe) scatterState = aov::lpeVolume(*lpe->t, scatterState);
                    U32 firstVolume = 0;  // the light set of the first volume here
                    while (!(here >> firstVolume & 1)) ++firstVolume;
                    const U32 lightSet = shaders[firstVolume].lightSet;
                    for (U32 li = 0; li < c.lightCount; ++li) {
                        const float usl[3] = {ms.next(), ms.next(), ms.next()};
                        if (!backend.inSet(lightSet, li)) continue;
                        const V3 Ls = volumeSourceTerm(c, backend, ms, backend.light(li), scatterPoint, rayDir, g, usl, ai);
                        const V3 contribution = volume::scaled(volume::times(throughput, Ls), invN * bi);
                        radiance = volume::plus(radiance, contribution);
                        if (lpe) lpeAdd(contribution, aov::lpeNext(*lpe->t, scatterState, aov::LpeLight, backend.light(li).lpeLabel));
                    }
#if !defined(__CUDACC__)
                    // The emissive volumes lighting this scatter (a direction toward each region that lights volumes,
                    // through its emissive volume integral, by the phase function).
                    {
                        const volume::Regions er = backend.emissiveRegions();
                        for (U32 ri = 0; ri < er.count; ++ri) {
                            const volume::Region& R = er.regions[ri];
                            if (!(R.visibility & 0x100u)) continue;
                            const float ue0 = ms.next(), ue1 = ms.next(), ue2 = ms.next();
                            float pdfWi, tEnd;
                            const V3 wi = volume::regionSample(R, er.pool, scatterPoint, ue0, ue1, ue2, pdfWi, tEnd);
                            if (pdfWi == 0.0f || !volume::finitef(pdfWi)) continue;
                            const V3 Le = emissiveVolumeIntegral(c, backend, ms, scatterPoint, wi, 0.0f, tEnd, R.volume, int(pv.depth) + 1);
                            const float ph = volume::phaseEval(g, V3{-rayDir.x, -rayDir.y, -rayDir.z}, wi);
                            const V3 contribution = volume::scaled(volume::times(throughput, Le), invN * bi * ph / pdfWi);
                            radiance = volume::plus(radiance, contribution);
                            if (lpe) lpeAdd(contribution, aov::lpeNext(*lpe->t, scatterState, aov::LpeEmission, shaders[R.volume].lpeLabel));
                        }
                    }
#endif
                    const float uq0 = ms.next(), uq1 = ms.next();
                    scatterDir = volume::phaseSample(g, V3{-rayDir.x, -rayDir.y, -rayDir.z}, uq0, uq1);
                }
            }
        }
    }
    pv.T = volume::times(pv.T, volume::times(vt, vth));
    return radiance;
}

//----------------------------------------------------------------------------
// Diffusion subsurface scattering (PathIntegratorSubsurface.cc, which MoonRay's vector mode calls).
namespace paths {
// A subsurface exit sample: where it is, the normal it is lit with, the slice's geometric normal,
// and the offset origins of rays leaving it (facing: the side 'above' is on).
struct ExitPoint {
    V3 P, N, Ng, facing, above, below;
};
X2_HD inline V3 exitOrigin(const ExitPoint& x, V3 wi) { return shade::dot(wi, x.facing) >= 0.0f ? x.above : x.below; }
// The exit's Lambertian lobe (white, cosine included, the shadow terminator fix of the slice's Ng).
X2_HD inline shade::Color exitLambert(const mixed::Config& c, const ExitPoint& x, V3 wo, V3 wi, float& pdf) {
    const shade::Slice slice{x.Ng, wo, shade::AllLobeTypes, c.shadowTerminatorFix, true, true};
    const float cosWi = fmaxf(shade::dot(x.N, wi), 0.0f);
    pdf = cosWi * shade::OneOverPi;
    return shade::rgb(shade::shadowTerminatorFix(slice, x.N, wi) * shade::OneOverPi * cosWi);
}
}  // namespace paths

// computeRadianceSubsurfaceSample: an exit sample's light samples and Lambertian samples (MIS, no
// clamping or roulette), its bounces pushed from the entry point as MoonRay's parent ray starts there.
// At the first hit (strat) a pixel's exits draw their light and Lambertian samples from one stratified set, as
// MoonRay's one sequence a pixel: Sobol index strat * split + s, keyed by the exit kind (stratAddress).
// The light path expressions at a subsurface's exits (computeRadianceSubsurfaceSample): the sample's AOVs, the
// state after the subsurface event (subsurfaceEventTransition, the exit's own Lambertian lobe ignored, MOONRAY-1957),
// and the sample's AOV backup for the random walk's scaling.
// The light filters' radiance at a point (defined below with the light samples).
template <class Backend>
X2_HD inline V3 filterRadiance(const Backend& backend, U32 geom, U32 light, const lights::Light& L, V3 p, V3 wi, float distance,
                               const paths::Pending& pv, const sample::Identity& id, V3 radiance);
struct SubsurfaceLpe {
    const aov::Tables* t = nullptr;
    float* aovs = nullptr;
    float* depths = nullptr;
    float* backup = nullptr;
    int state = -1;
};
template <class Backend>
X2_CALL V3 subsurfaceExitRadiance(const mixed::Config& c, const sample::Identity& root, Backend& backend, U32 lightSet,
                                       const paths::Pending& pv, U32 depthAdd, U32 subsurfaceDepth, const shade::State& entry,
                                       const resume::Hit& entryHit, const paths::ExitPoint& x, shade::Color pt, U32 split, U64 branch,
                                       bool doIndirect, paths::Pending* stack, U32& top, SampleStatus& status,
                                       bool stratified = false, U64 strat = 0, U64 stratAddress = 0, const SubsurfaceLpe* lpe = nullptr) {
    const bool lpeOn = lpe && lpe->aovs && lpe->state >= 0;
    using namespace paths;
    backend.shadowFor(~0u, entryHit.geom);  // the exits' shadow rays leave the entry's mesh
    V3 radiance{0, 0, 0};
    pt = pt * (1.0f / float(split));
    for (U32 i = 0; i < c.lightCount; ++i) {
        if (!backend.inSet(lightSet, i)) continue;
        const lights::Light& L = backend.light(i);
        if (!(L.visibilityMask & lights::VisibleDiffuseReflection)) continue;
        if (!lobeSeesLight(backend, pv, backend.exitLightSet, i)) continue;
        for (U32 s = 0; s < split; ++s) {
            sample::Identity id = root;
            id.branch = childBranch(branch, i, s);
            Stream ls(id, pv.depth, 0);
            float r[3] = {ls.next(), ls.next(), ls.next()};
            if (stratified) {
                const sample::Identity sid{root.pixel, strat * split + s, root.stream, childBranch(stratAddress, i, 0)};
                r[0] = sample::draw(sid, sample::CameraDimensions + 6, c.sampling);
                r[1] = sample::draw(sid, sample::CameraDimensions + 7, c.sampling);
            }
            V3 wi;
            lights::Hit lh;
            if (!lights::sample(L, x.P, &x.N, r, wi, lh)) continue;
            float pdfL;
            V3 Li = lights::eval(L, x.P, wi, lh, pdfL);
            if (c.lightFilters) {
                sample::Identity fid = id;
                fid.branch = childBranch(branch, i, s + 0xfff4u);
                Li = filterRadiance(backend, entryHit.geom, i, L, x.P, wi, lh.distance, pv, fid, Li);
            }
            if (shade::isSampleInvalid(Li, pdfL) || !(fabsf(pdfL) < 3.4e38f) || !shade::finite3(Li)) continue;  // isSampleInvalid: a pdf of 0 alone
            float pdf;
            const shade::Color f = exitLambert(c, x, entry.wo, wi, pdf);
            if (shade::isSampleInvalid(f, pdf) || shade::dot(x.N, wi) < shade::Epsilon) continue;
            const shade::Color t = pt * Li * f * (1.0f / pdfL) * lights::powerHeuristic(pdfL, pdf);
            float near = 0.0f, far = lh.distance * lights::HitEpsilonEnd;
            lightShadowSpanScalar(L, near, far);
            Stream ps(id, pv.depth, 0);
            const float keep = shade::finite3(t) ? scalarShadowKeep(c, backend, L, exitOrigin(x, wi), wi, near, far, pv.depth,
                                                                     shade::luminance(pv.T) < 1.0f - c.presenceQuality, ps) : 0.0f;
            shade::Color tr{1, 1, 1};
            if (keep > 0.0f) {
                shade::Color tv = t * keep;
                if (c.volumeCount) {
                    sample::Identity tid = id;
                    tid.branch = childBranch(id.branch, 0xfffb, 0);
                    Stream vs(tid, pv.depth, 0);
                    tr = volumeTransmittance(c, backend, vs, exitOrigin(x, wi), wi, near, far, 1.0f);
                    tv = tv * tr;
                }
                radiance += tv;
                // LPE: reached, to every expression as it arrives, the unoccluded ones too (MoonRay's subsurface
                // light samples), and to the visibility AOVs whatever the depth.
                if (lpeOn) {
                    const int state = aov::lpeNext(*lpe->t, lpe->state, aov::LpeLight, L.lpeLabel);
                    aov::light(*lpe->t, lpe->aovs, lpe->depths, tv, nullptr, 0, state);
                    aov::visibility(*lpe->t, lpe->aovs, keep * (tr.x + tr.y + tr.z) / 3.0f, 1.0f, state);
                }
            } else if (lpeOn && shade::finite3(t)) {
                // Occluded: to the unoccluded expressions alone, and a miss.
                const int state = aov::lpeNext(*lpe->t, lpe->state, aov::LpeLight, L.lpeLabel);
                aov::light(*lpe->t, lpe->aovs, lpe->depths, t, nullptr, aov::PrefixUnoccluded, state);
                aov::visibility(*lpe->t, lpe->aovs, 0.0f, 1.0f, state);
            }
        }
    }
    for (U32 s = 0; s < split; ++s) {
        sample::Identity id = root;
        id.branch = childBranch(branch, 0xffff, s);
        Stream bs(id, pv.depth, 0);
        float r1 = bs.next(), r2 = bs.next();
        const float choice = bs.next();
        if (stratified) {
            const sample::Identity sid{root.pixel, strat * split + s, root.stream, childBranch(stratAddress, 0xffff, 0)};
            r1 = sample::draw(sid, sample::CameraDimensions + 10, c.sampling);
            r2 = sample::draw(sid, sample::CameraDimensions + 11, c.sampling);
        }
        const V3 wi = shade::localToGlobal(shade::frame(x.N), shade::sampleLocalHemisphereCosine(r1, r2));
        float pdf;
        const shade::Color f = exitLambert(c, x, entry.wo, wi, pdf);
        if (shade::isSampleInvalid(f, pdf) || shade::dot(x.N, wi) < shade::Epsilon) continue;
        U32 hits = 0;
        for (U32 j = 0; j < c.lightCount; ++j) {
            lights::Hit lh;
            if (backend.inSet(lightSet, j) && (backend.light(j).visibilityMask & lights::VisibleDiffuseReflection) &&
                lobeSeesLight(backend, pv, backend.exitLightSet, j) &&
                lights::intersect(backend.light(j), x.P, &x.N, wi, 3.4028234663852886e+38f, lh))
                ++hits;
        }
        V3 Li{0, 0, 0};
        float pdfL = 0.0f, lightDistance = 0.0f;
        bool lightHit = false;
        U32 lightIndex = 0;
        if (hits) {
            const U32 chosen = U32(choice * float(hits)) < hits ? U32(choice * float(hits)) : hits - 1;
            for (U32 j = 0, seen = 0; j < c.lightCount; ++j) {
                lights::Hit lh;
                if (!backend.inSet(lightSet, j) || !(backend.light(j).visibilityMask & lights::VisibleDiffuseReflection) ||
                    !lobeSeesLight(backend, pv, backend.exitLightSet, j) ||
                    !lights::intersect(backend.light(j), x.P, &x.N, wi, 3.4028234663852886e+38f, lh))
                    continue;
                if (seen++ != chosen) continue;
                Li = lights::eval(backend.light(j), x.P, wi, lh, pdfL) * float(hits);
                if (c.lightFilters) {
                    sample::Identity fid = root;
                    fid.branch = childBranch(branch, j, 0xfff5u);
                    Li = filterRadiance(backend, entryHit.geom, j, backend.light(j), x.P, wi, lh.distance, pv, fid, Li);
                }
                lightHit = !shade::isSampleInvalid(Li, pdfL) && fabsf(pdfL) < 3.4e38f && shade::finite3(Li);
                lightDistance = lh.distance;
                lightIndex = j;
                break;
            }
        }
        const shade::Color throughput = pt * f * (1.0f / pdf);
        float keptShare = 1.0f;
        bool pushChild = doIndirect;
        if (pushChild && top + PendingReserve >= MaxPending) {
            sample::Identity kid = root;
            kid.branch = childBranch(branch, 0xfffd, s);
            Stream chance(kid, pv.depth, 0);
            pushChild = roomForChild(top, chance.next(), keptShare);
        }
        if (pushChild) {
            Pending child;
            child.pixelWeight = 0.0f;
            child.volumeDepth = pv.volumeDepth;
            child.origin = shade::dot(wi, entryHit.normal) >= 0.0f ? entryHit.origin : entryHit.below;
            child.dir = wi;
            child.T = throughput * keptShare;
            child.branch = childBranch(branch, 0xfffe, s);
            child.depth = pv.depth + 1;
            child.diffuseDepth = pv.diffuseDepth + depthAdd;
            child.nonMirrorDepth = pv.nonMirrorDepth + depthAdd;
            child.glossyDepth = pv.glossyDepth;
            child.mirrorDepth = pv.mirrorDepth;
            child.hairDepth = pv.hairDepth;
            child.presenceDepth = pv.presenceDepth;
            child.subsurfaceDepth = subsurfaceDepth;
            child.accumOpacity = pv.accumOpacity;
            child.lobeType = shade::Diffuse | shade::Reflection;  // the exit's Lambertian lobe
            inheritLobeSets(child, pv, backend.exitLightSet);
            child.lpe = lpeOn ? lpe->state : -1;
            child.aovPixelWeight = pv.pixelWeight;
            child.farT = lightHit ? lightDistance * lights::HitEpsilonEnd : 3.402823466e+38f;
            // computeMinRoughness of a Lambertian lobe (roughness one).
            child.minRoughness[0] = shade::clampf(c.roughnessClamping, pv.minRoughness[0], 1.0f);
            child.minRoughness[1] = shade::clampf(c.roughnessClamping, pv.minRoughness[1], 1.0f);
            // scatterAndScale with scale one: the parent (entry) ray's differentials and a Lambertian lobe's.
            child.diff = shade::RayDiff{};
            if (entry.diffValid) {
                V3 dDdx, dDdy;
                shade::localHemisphereCosineDifferentials(r1, r2, dDdx, dDdy);
                shade::squarifyRectangle(dDdx, dDdy);
                const shade::Frame F = shade::frame(x.N);
                dDdx = shade::localToGlobal(F, dDdx);
                dDdy = shade::localToGlobal(F, dDdy);
                const V3 shift = entry.P - child.origin;
                child.diff.dOx = entry.dPdx + shift;
                child.diff.dOy = entry.dPdy + shift;
                child.diff.dDx = shade::normalize(wi + dDdx) - wi;
                child.diff.dDy = shade::normalize(wi + dDdy) - wi;
                child.diff.valid = shade::finite3(child.diff.dDx) && shade::finite3(child.diff.dDy) ? 1u : 0u;
            }
            stack[top++] = child;
        }
        if (lightHit) {
            const shade::Color t = throughput * Li * lights::powerHeuristic(pdf, pdfL);
            float near = 0.0f, far = lightDistance * lights::HitEpsilonEnd;
            lightShadowSpanScalar(backend.light(lightIndex), near, far);
            Stream ps(id, pv.depth, 0);
            const float keep = shade::finite3(t) ? scalarShadowKeep(c, backend, backend.light(lightIndex), exitOrigin(x, wi), wi, near, far, pv.depth,
                                                                     shade::luminance(pv.T) < 1.0f - c.presenceQuality, ps) : 0.0f;
            const int lightState = lpeOn ? aov::lpeNext(*lpe->t, lpe->state, aov::LpeLight, backend.light(lightIndex).lpeLabel) : -1;
            if (keep <= 0.0f && lpeOn && shade::finite3(t)) {
                // Occluded: a miss, and the unoccluded expressions as if it had arrived.
                aov::visibility(*lpe->t, lpe->aovs, 0.0f, 1.0f, lightState);
                aov::light(*lpe->t, lpe->aovs, lpe->depths, t, nullptr, aov::PrefixUnoccluded, lightState);
            }
            if (keep > 0.0f) {
                shade::Color tv = t * keep;
                shade::Color tr{1, 1, 1};
                if (c.volumeCount) {
                    sample::Identity tid = id;
                    tid.branch = childBranch(id.branch, 0xfffb, 1);
                    Stream vs(tid, pv.depth, 0);
                    tr = volumeTransmittance(c, backend, vs, exitOrigin(x, wi), wi, near, far, 1.0f);
                    tv = tv * tr;
                }
                // LPE: the unoccluded expressions as it left the exit, the rest as it arrived.
                if (lpeOn) {
                    aov::visibility(*lpe->t, lpe->aovs, keep * (tr.x + tr.y + tr.z) / 3.0f, 1.0f, lightState);
                    aov::light(*lpe->t, lpe->aovs, lpe->depths, t, &tv, aov::PrefixUnoccluded, lightState);
                }
                radiance += tv;
            }
        }
    }
    return radiance;
}
// computeRadiancePathTraceSubsurface: MoonRay's random walk (VolumeSubsurface). Each sample enters along a
// cosine direction below the surface: first a zero-scatter step with the zero-scatter extinction; if it does not
// leave, up to 256 scattering steps, isotropic then mostly Dwivedi sampled, with distances by the throughput's
// channels. A walk that leaves the object (through the same material, the exit facing away when asked) is lit at
// its exit as a white Lambertian; the sum is scaled up for the samples that never left. MoonRay draws the walk's
// distances and its backward-or-forward choices from two copies of one sequence (distanceSamples and
// dwivediDirectionSamples share their SequenceIDIntegrator), so each scatter event's choice is the distance draw
// before the one that led to it; the walk here keeps that pairing. In the first three non-mirror bounces MoonRay's
// sequences are its integrator tables (IntegratorSample: consecutive points of a bc_1D sequence for the distances
// and choices, another for the channels, a ppd_2D sequence for the directions, each chosen and Cranley-Patterson
// rotated by the sample's identity); the walk's estimate depends on that ordering, so the walk here draws from the
// same tables, the sequences and rotations chosen at random, and independent numbers after them.
template <class Backend>
X2_CALL V3 randomWalkRadiance(const mixed::Config& c, const sample::Identity& root, Backend& backend, U32 lightSet,
                                   const paths::Pending& pv, U32 subsurfaceDepth, const shade::State& st, const resume::Hit& hit,
                                   const shade::Bsdf& bsdf, bool doIndirect, paths::Pending* stack, U32& top, SampleStatus& status,
                                   const SubsurfaceLpe* lpe = nullptr, shade::Color* ssAov = nullptr) {
    using namespace paths;
    using shade::Color;
    V3 radiance{0, 0, 0};
    Color ss{0, 0, 0};  // the exits' throughput, the subsurface's material AOV albedo
    const U32 firstChild = top;  // the bounces this walk pushes, scaled with it
    // The light and visibility AOVs as they stood (MOONRAY-3352): the walk's own share is scaled with its radiance.
    const aov::Tables* at = lpe && lpe->aovs && lpe->backup ? lpe->t : nullptr;
    if (at)
        for (U32 i = 0; i < at->count; ++i) {
            const aov::Entry& e = at->entries[i];
            if (e.type != aov::TypeLight && e.type != aov::TypeVisibility) continue;
            for (U32 k = 0; k < e.channels; ++k) lpe->backup[e.offset + k] = lpe->aovs[e.offset + k];
        }
    const shade::BssrdfRecord& b = bsdf.bssrdf;
    const Color albedo = b.rwAlbedo, sigmaT = b.sigmaT, sigmaS = sigmaT * albedo, v0 = b.dwivediV0;
    const U32 n = pv.nonMirrorDepth == 0 ? c.bssrdfSamples : 1u;
    const Color fresnel =
        b.fresnel != shade::NoIndex ? shade::fresnelEval(bsdf.fresnels, b.fresnel, fabsf(shade::dot(st.N, st.wo))) : shade::rgb(1);
    const U64 branch = childBranch(pv.branch, 0xfffc, 0);
    const U32 material = backend.materialId(hit.geom);
    const shade::Frame frame = shade::frame(st.N);
    const float wBackwardSampling = 0.9f;
    auto expC = [](Color a) { return Color{expf(a.x), expf(a.y), expf(a.z)}; };
    U32 evaluated = 0;
    for (U32 s = 0; s < n; ++s) {
        sample::Identity id = root;
        id.branch = childBranch(branch, 1, s);
        Stream rs(id, pv.depth, 0);
        Color pt = b.scale * pv.T * fresnel * (1.0f / float(n));
        if (shade::isBlack(pt)) continue;
        // The walk's three sequences (MoonRay's tables up to sample 1015, as partitioned1D/2D use them).
        const float* tables = pv.nonMirrorDepth < 3 ? backend.walkSequences() : nullptr;
        U32 distanceBase = 0, channelBase = 0, directionBase = 0, kDistance = 0, kChannel = 0, kDirection = 0;
        float distanceShift = 0.0f, channelShift = 0.0f, directionShift[2] = {0.0f, 0.0f};
        if (tables) {
            auto sequence = [&]() { return U32(fminf(rs.next() * 4096.0f, 4095.0f)); };
            distanceBase = sequence() * 1024u;
            distanceShift = rs.next();
            channelBase = sequence() * 1024u;
            channelShift = rs.next();
            directionBase = 4096u * 1024u + sequence() * 2048u;
            directionShift[0] = rs.next();
            directionShift[1] = rs.next();
        }
        auto rotate = [](float x, float shift) {
            const float r = x + shift;
            return r >= 1.0f ? r - 1.0f : r;
        };
        auto nextDistance = [&]() { return tables && kDistance < 1016 ? rotate(tables[distanceBase + kDistance++], distanceShift) : rs.next(); };
        auto nextChannel = [&]() { return tables && kChannel < 1016 ? rotate(tables[channelBase + kChannel++], channelShift) : rs.next(); };
        auto nextDirection = [&](float& a, float& b2) {
            if (tables && kDirection < 1016) {
                a = rotate(tables[directionBase + 2 * kDirection], directionShift[0]);
                b2 = rotate(tables[directionBase + 2 * kDirection + 1], directionShift[1]);
                ++kDirection;
            } else {
                a = rs.next();
                b2 = rs.next();
            }
        };
        float u0, u1;
        nextDirection(u0, u1);
        V3 dir = -shade::localToGlobal(frame, shade::sampleLocalHemisphereCosine(u0, u1));
        // Below the surface: the hit's far-side origin.
        V3 org = shade::dot(dir, hit.normal) >= 0.0f ? hit.origin : hit.below;
        bool reachSurface = false, hadScatterEvent = false, choseDwivedi = false;
        float previousDistanceU = 0.0f;  // the last distance draw (the choice at the next scatter event)
        resume::Hit exitHit;
        shade::State exitState;
        auto valid = [&](const resume::Hit& h, const surface::Ray& ray) {
            if (h.geom == ~0u) return false;
            exitState = shade::State{};
            exitState.incoming = shade::RayDiff{};
            if (!backend.surfaceState(h, ray, exitState)) return false;
            return !b.resolveSelfIntersections || !exitState.entering;
        };
        // Zero scatter.
        {
            const float uChannel = nextChannel(), uDistance = nextDistance();
            previousDistanceU = uDistance;
            const float recip = 1.0f / (pt.x + pt.y + pt.z);
            const int ch = uChannel < pt.x * recip ? 0 : uChannel < (pt.x + pt.y) * recip ? 1 : 2;
            const float zs = ch == 0 ? b.zeroScatterSigmaT.x : ch == 1 ? b.zeroScatterSigmaT.y : b.zeroScatterSigmaT.z;
            const float tfar = -logf(1.0f - uDistance) / zs;
            const surface::Ray ray{org, dir, V3{0, 0, 0}, V3{0, 0, 0}, 0.0f, tfar};
            const resume::Hit h = backend.traceMaterial(ray, material);
            if (valid(h, ray)) {
                const float t1 = h.t;
                const Color tr{expf(-b.zeroScatterSigmaT.x * t1), expf(-b.zeroScatterSigmaT.y * t1), expf(-b.zeroScatterSigmaT.z * t1)};
                const float exitProbability = (pt.x * tr.x + pt.y * tr.y + pt.z * tr.z) * recip;
                pt = pt * tr * (1.0f / exitProbability);
                reachSurface = true;
                exitHit = h;
                ++evaluated;
            }
        }
        if (!reachSurface) {
            for (U32 nScatter = 0; nScatter < 256; ++nScatter) {
                if (shade::isBlack(pt)) break;
                const float phaseSamplingWeight = nScatter > 1 ? 0.1f : 1.0f;
                // sampleChannel
                const float uChannel = nextChannel();
                const float norm = 1.0f / (pt.x + pt.y + pt.z);
                const Color channelWeights = norm * pt;
                const int ch = uChannel < channelWeights.x ? 0 : uChannel < channelWeights.x + channelWeights.y ? 1 : 2;
                // sampleDistanceMultiScatter
                const float uDistance = nextDistance(), backward = previousDistanceU;
                previousDistanceU = uDistance;
                float NdotDir = shade::dot(dir, frame.z);
                // Dwivedi distances: longer steps back toward the normal (the choice is the previous direction's).
                const Color sigmaTPrime = sigmaT * (shade::rgb(1) - Color{NdotDir / v0.x, NdotDir / v0.y, NdotDir / v0.z});
                const float sT = ch == 0 ? sigmaT.x : ch == 1 ? sigmaT.y : sigmaT.z;
                const float sTP = ch == 0 ? sigmaTPrime.x : ch == 1 ? sigmaTPrime.y : sigmaTPrime.z;
                const float distance = -logf(1.0f - uDistance) / (choseDwivedi ? sTP : sT);
                Color tr = expC(-1.0f * sigmaT * distance);
                const Color pdfDistance = sigmaT * tr, pdfDistanceDwivedi = sigmaTPrime * expC(-1.0f * sigmaTPrime * distance);
                const surface::Ray ray{org, dir, V3{0, 0, 0}, V3{0, 0, 0}, 0.0f, distance};
                const resume::Hit h = backend.traceMaterial(ray, material);
                if (valid(h, ray)) {
                    if (!hadScatterEvent) break;  // zero scattering was counted above
                    tr = expC(-1.0f * sigmaT * h.t);
                    reachSurface = true;
                    const Color trp = expC(-1.0f * sigmaTPrime * h.t);
                    const float pdfSurf = pt.x * norm * (phaseSamplingWeight * tr.x + (1.0f - phaseSamplingWeight) * trp.x) +
                                          pt.y * norm * (phaseSamplingWeight * tr.y + (1.0f - phaseSamplingWeight) * trp.y) +
                                          pt.z * norm * (phaseSamplingWeight * tr.z + (1.0f - phaseSamplingWeight) * trp.z);
                    pt = pt * tr * (1.0f / pdfSurf);
                    exitHit = h;
                    break;
                }
                // A scattering event.
                hadScatterEvent = true;
                org = org + distance * dir;
                pt = pt * tr * sigmaS * shade::OneOverFourPi;
                float d0, d1;
                nextDirection(d0, d1);
                // sampleDirection
                if (d0 < phaseSamplingWeight) {
                    choseDwivedi = false;
                    const float r = d0 / phaseSamplingWeight;
                    const V3 local = shade::sampleSphereUniform(r, d1);
                    dir = shade::localToGlobal(frame, local);
                    NdotDir = backward < wBackwardSampling ? local.z : -local.z;
                } else {
                    choseDwivedi = true;
                    const float r = (d0 - phaseSamplingWeight) / (1.0f - phaseSamplingWeight);
                    const float vc = ch == 0 ? v0.x : ch == 1 ? v0.y : v0.z;
                    NdotDir = vc - (vc + 1.0f) * powf((vc - 1.0f) / (vc + 1.0f), r);
                    const float sine = sqrtf(fmaxf(0.0f, 1.0f - NdotDir * NdotDir)), phi = shade::TwoPi * d1;
                    const V3 local{sine * cosf(phi), sine * sinf(phi), NdotDir};
                    dir = backward < wBackwardSampling ? shade::normalize(shade::localToGlobal(frame, local))
                                                       : -shade::normalize(shade::localToGlobal(frame, local));
                }
                // calculateOneSamplePDF
                const Color pdfDwivedi = shade::OneOverTwoPi * b.dwivediNormPdf / (v0 - shade::rgb(NdotDir));
                const float pdfPhase = shade::OneOverFourPi;
                float pdfSampling = 0.0f;
                const float pd[3] = {pdfDistance.x, pdfDistance.y, pdfDistance.z},
                            pdd[3] = {pdfDistanceDwivedi.x, pdfDistanceDwivedi.y, pdfDistanceDwivedi.z},
                            pdw[3] = {pdfDwivedi.x, pdfDwivedi.y, pdfDwivedi.z}, cw[3] = {channelWeights.x, channelWeights.y, channelWeights.z};
                for (int k = 0; k < 3; ++k)
                    pdfSampling += cw[k] * (phaseSamplingWeight * pdfPhase * pd[k] + (1.0f - phaseSamplingWeight) * pdw[k] * pdd[k]);
                pt = pt * (1.0f / pdfSampling);
            }
            if (hadScatterEvent) ++evaluated;
        }
        if (!reachSurface) continue;
        // The exit: lit as a white Lambertian facing out of the object.
        const V3 nOut = -exitState.N;
        const ExitPoint x{exitState.P, nOut, nOut, exitHit.normal, exitHit.origin, exitHit.below};
        ss = ss + pt;
        radiance += subsurfaceExitRadiance(c, root, backend, lightSet, pv, 1, subsurfaceDepth, st, hit, x, pt, 1, childBranch(branch, 2, s),
                                           doIndirect, stack, top, status, false, 0, 0, lpe);
        if (status != SampleStatus::Done) return radiance;
    }
    if (evaluated > 0 && evaluated < n) {
        const float scale = float(n) / float(evaluated);
        radiance = radiance * scale;
        ss = ss * scale;
        for (U32 k = firstChild; k < top; ++k) stack[k].T = stack[k].T * scale;
        if (at)
            for (U32 i = 0; i < at->count; ++i) {
                const aov::Entry& e = at->entries[i];
                if (e.type != aov::TypeLight && e.type != aov::TypeVisibility) continue;
                for (U32 k = 0; k < e.channels; ++k) {
                    const float before = lpe->backup[e.offset + k];
                    lpe->aovs[e.offset + k] = before + scale * (lpe->aovs[e.offset + k] - before);
                }
            }
    }
    if (ssAov) *ssAov = *ssAov + ss;
    return radiance;
}

// computeRadianceDiffusionSubsurface: local samples (probe rays along the entry's frame axes, MIS over the
// axes, area compensation) or, when the ray's footprint is wider than the profile, the entry point lit as
// Lambertian; then computeDiffusionForwardScattering's samples through the object.
template <class Backend>
X2_CALL V3 subsurfaceRadiance(const mixed::Config& c, const sample::Identity& root, Backend& backend, U32 lightSet,
                                   const paths::Pending& pv, U32 subsurfaceDepth, const shade::State& st, const resume::Hit& hit,
                                   V3 rayDir, const shade::Bsdf& bsdf, bool doIndirect, paths::Pending* stack, U32& top,
                                   SampleStatus& status, const SubsurfaceLpe* lpe = nullptr, shade::Color* ssAov = nullptr) {
    backend.exitLightSet = bsdf.bssrdf.lightSet;  // the exits' lobes inherit the Bssrdf's light set
    using namespace paths;
    V3 radiance{0, 0, 0};
    if (!c.enableSss || c.bssrdfSamples < 1) return radiance;
    const shade::BssrdfRecord& b = bsdf.bssrdf;
    if (b.type == shade::BssrdfRandomWalk)
        return randomWalkRadiance(c, root, backend, lightSet, pv, subsurfaceDepth, st, hit, bsdf, doIndirect, stack, top, status, lpe,
                                  ssAov);
    const float maxRadius = b.maxRadius, minRadius = maxRadius * 0.1f;
    const V3 P = st.P, N = st.N;
    const shade::Frame localF = shade::frame(N);
    const shade::Color scaleFresnelWo =
        (b.fresnel != shade::NoIndex ? shade::fresnelEval(bsdf.fresnels, b.fresnel, fabsf(shade::dot(N, st.wo))) : shade::rgb(1)) * b.scale;
    const U64 sssBranch = childBranch(pv.branch, 0xfffd, 0);
    // At the first non-mirror hit MoonRay draws a pixel's subsurface samples, and their exits' light and Lambertian
    // samples, from one sequence across the pixel's samples; here one stratified set, Sobol index index * split + i.
    const bool stratified = pv.nonMirrorDepth == 0;
    const U32 material = backend.materialId(hit.geom);
    const ExitPoint entryPoint{P, N, st.Ng, hit.normal, hit.origin, hit.below};
    // canSubstituteSubsurface
    const V3 dPdx = st.dPds * st.dsdx + st.dPdt * st.dtdx, dPdy = st.dPds * st.dsdy + st.dPdt * st.dtdy;
    const float footprint = (shade::dot(dPdx, dPdx) + shade::dot(dPdy, dPdy)) * 0.5f;
    U32 split;
    if (footprint / (maxRadius * maxRadius) > 0.85f) {
        const U32 lambertSplit = pv.nonMirrorDepth > 0 ? 1u : c.bsdfSamples;
        const shade::Color throughput = pv.T * shade::bssrdfDiffuseReflectance(b) * scaleFresnelWo;
        radiance += subsurfaceExitRadiance(c, root, backend, lightSet, pv, 1, subsurfaceDepth, st, hit, entryPoint, throughput, lambertSplit,
                                           childBranch(sssBranch, 1, 0), doIndirect, stack, top, status, stratified, root.index,
                                           childBranch(sssBranch, 1, 1), lpe);
        split = 1;
    } else {
        split = pv.nonMirrorDepth > 0 ? 2u : c.bssrdfSamples;
        if (split > 16) split = 16;
        const shade::Color ptSubsurface = pv.T * (1.0f / float(split));
        shade::Color measured{0, 0, 0};
        ExitPoint exits[16];
        shade::Color pts[16];
        U32 drawn[16];
        U32 n = 0;
        for (U32 i = 0; i < split; ++i) {
            sample::Identity id = root;
            id.branch = childBranch(sssBranch, 2, i);
            Stream ss(id, pv.depth, 0);
            float r1 = ss.next();
            float r2 = ss.next();
            if (stratified) {
                const sample::Identity sid{root.pixel, U64(root.index) * split + i, root.stream, childBranch(sssBranch, 2, 0xfff0)};
                r1 = sample::draw(sid, sample::CameraDimensions + 4, c.sampling);
                r2 = sample::draw(sid, sample::CameraDimensions + 5, c.sampling);
            }
            V3 directionProj;
            const int axis = shade::bssrdfSelectAxis(localF, r1, directionProj);
            V3 dPi;
            float r;
            const float pdf = shade::bssrdfSampleLocal(b, r1, r2, dPi, r);
            if (pdf == 0.0f) continue;
            const V3 PiTangent = P + shade::bssrdfOffsetLocalToGlobal(localF, axis, dPi);
            const float search = fmaxf(fminf(r, maxRadius), minRadius);
            const surface::Ray probe{PiTangent - directionProj * search, directionProj, V3{0, 0, 0}, V3{0, 0, 0}, 0.0f, 2.0f * search};
            const resume::Hit ph = backend.traceMaterial(probe, material);
            if (ph.geom == ~0u) continue;
            shade::State ps;
            ps.incoming = shade::RayDiff{};
            if (!backend.surfaceState(ph, probe, ps)) continue;
            if (-shade::dot(ps.N, directionProj) < shade::Epsilon) continue;
            const V3 dp = P - ps.P;
            r = shade::length(dp);
            if (r > maxRadius) continue;
            const shade::Color e = shade::bssrdfEval(b, r) * shade::bssrdfMISAxisWeight(localF, ps.N, dp, b);
            measured = measured + e;
            exits[n] = ExitPoint{ps.P, ps.N, ps.Ng, ph.normal, ph.origin, ph.below};
            pts[n] = ptSubsurface * e * scaleFresnelWo;
            drawn[n] = i;
            ++n;
        }
        measured = measured * (1.0f / float(split));
        const shade::Color compensation = shade::bssrdfAreaCompensation(measured, shade::bssrdfDiffuseReflectance(b));
        if (ssAov)
            for (U32 k = 0; k < n; ++k) *ssAov = *ssAov + pts[k] * compensation;
        for (U32 k = 0; k < n; ++k) {
            radiance += subsurfaceExitRadiance(c, root, backend, lightSet, pv, 1, subsurfaceDepth, st, hit, exits[k], pts[k] * compensation, 1,
                                               childBranch(sssBranch, 3, k), doIndirect, stack, top, status, stratified,
                                               U64(root.index) * split + drawn[k], childBranch(sssBranch, 3, 0xfff0), lpe);
            if (status != SampleStatus::Done) return radiance;
        }
    }
    if (status != SampleStatus::Done) return radiance;
    // computeDiffusionForwardScattering: directions in an 85-degree cap into the surface, the profile at the
    // distance travelled, a Jacobian; its exit samples one bounce deeper still.
    const float cosThetaMax = cosf(85.0f / 180.0f * shade::Pi);
    const shade::Color ptSubsurface = pv.T * (1.0f / float(split));
    for (U32 i = 0; i < split; ++i) {
        sample::Identity id = root;
        id.branch = childBranch(sssBranch, 4, i);
        Stream ss(id, pv.depth, 0);
        float r1 = ss.next(), r2 = ss.next();
        if (stratified) {
            const sample::Identity sid{root.pixel, U64(root.index) * split + i, root.stream, childBranch(sssBranch, 4, 0xfff0)};
            r1 = sample::draw(sid, sample::CameraDimensions + 8, c.sampling);
            r2 = sample::draw(sid, sample::CameraDimensions + 9, c.sampling);
        }
        V3 local;
        const float pdf = shade::bssrdfSampleGlobal(r1, r2, cosThetaMax, local);
        const V3 direction = -shade::localToGlobal(localF, local);
        const surface::Ray back{shade::dot(direction, hit.normal) >= 0.0f ? hit.origin : hit.below, direction, V3{0, 0, 0},
                                V3{0, 0, 0}, 0.0f, maxRadius};
        const resume::Hit bh = backend.traceMaterial(back, material);
        if (bh.geom == ~0u) continue;
        shade::State bs;
        bs.incoming = shade::RayDiff{};
        if (!backend.surfaceState(bh, back, bs)) continue;
        const V3 Ni = -bs.N;
        if (-shade::dot(Ni, N) < shade::Epsilon) continue;
        const float r = bh.t;
        const float cosTheta = fabsf(shade::dot(Ni, direction));
        const float jacobian = r * r / fmaxf(cosTheta, 0.05f);
        const shade::Color pt = ptSubsurface * shade::bssrdfEval(b, r) * (jacobian / pdf) * scaleFresnelWo;
        if (!shade::finite3(pt)) continue;
        if (ssAov) *ssAov = *ssAov + pt;
        const ExitPoint x{bs.P, Ni, -bs.Ng, bh.normal, bh.origin, bh.below};
        radiance += subsurfaceExitRadiance(c, root, backend, lightSet, pv, 2, subsurfaceDepth, st, hit, x, pt, 1, childBranch(sssBranch, 5, i),
                                           doIndirect, stack, top, status, stratified, U64(root.index) * split + i,
                                           childBranch(sssBranch, 5, 0xfff0), lpe);
        if (status != SampleStatus::Done) return radiance;
    }
    return radiance;
}

// A shading point's light filters (light_filter_core.h): what they read of the point, its path and the light, and their
// colour on the light's radiance (Light::eval's LightFilterList_eval, the filters' random values drawn from a stream
// of their own) or whether they let the light reach the point (canIlluminate's canIlluminateLightFilterList; an
// environment light's canIlluminate reads none).
template <class Backend>
X2_HD inline lights::FilterPoint filterPoint(const Backend& backend, const lights::Light& L, V3 p, const paths::Pending& pv) {
    lights::FilterPoint d;
    d.p = p;
    d.time = backend.time;
    d.lightPosition = L.position;
    d.lightDirection = L.direction;
    d.lightRadius = lights::filterLightRadius(L);
    d.lightRender2Local = lights::lightRender2Local(L);
    d.hasPath = true;
    d.diffuseDepth = pv.diffuseDepth;
    d.glossyDepth = pv.glossyDepth;
    d.mirrorDepth = pv.mirrorDepth;
    return d;
}
// The light filters' lookups on the backend: a cookie's map at (s, t), a VDB filter's density at a point.
template <class Backend>
struct FilterLookups {
    const Backend& backend;
    X2_HD V3 operator()(U32 program, U32 reg, float s, float t) const { return backend.filterMap(program, reg, s, t); }
    X2_HD float vdb(U32 grid, U32 mode, V3 p) const { return backend.filterVdb(grid, mode, p); }
};
template <class Backend>
X2_HD inline V3 filterRadiance(const Backend& backend, U32 geom, U32 light, const lights::Light& L, V3 p, V3 wi, float distance,
                               const paths::Pending& pv, const sample::Identity& id, V3 radiance) {
    const lights::FilterList* list = backend.filterList(geom, light);
    if (!list) return radiance;
    lights::FilterPoint d = filterPoint(backend, L, p, pv);
    d.wi = wi;
    d.distance = distance;
    if (list->needsSamples) {
        paths::Stream fs(id, pv.depth, 0);
        d.r2[0] = fs.next();
        d.r2[1] = fs.next();
        d.r3[0] = fs.next();
        d.r3[1] = fs.next();
        d.r3[2] = fs.next();
    }
    const V3 f = lights::filterListEval(backend.filters(), *list, d, FilterLookups<Backend>{backend});
    return V3{radiance.x * f.x, radiance.y * f.y, radiance.z * f.z};
}
template <class Backend>
X2_HD inline bool filterAllows(const Backend& backend, U32 geom, U32 light, const lights::Light& L, V3 p, float radius,
                               const paths::Pending& pv) {
    if (L.type == lights::Env) return true;
    const lights::FilterList* list = backend.filterList(geom, light);
    if (!list) return true;
    lights::FilterPoint d = filterPoint(backend, L, p, pv);
    d.radius = radius;
    return lights::filterListCanIlluminate(backend.filters(), *list, d);
}

// The volumes' and subsurface's integrators as the path tracer calls them (INDEXER.md, 5): these on the CPU and in a
// GPU program built whole; in a program built in modules the GPU backend's overloads (async_gpu_device.h), found by
// argument-dependent lookup, call the volume module's continuation callables.
template <class Backend>
X2_HD inline V3 volumeRadianceCall(const mixed::Config& c, const sample::Identity& root, Backend& backend, paths::Pending& pv, V3 org, V3 dir,
                                   float tnear, float tfar, V3* alphaTransmittance, V3* minTransmittance, VolumeCryptoShares* cryptoShares,
                                   const VolumeLpe* lpe, float* surfaceT) {
    return volumeRadiance(c, root, backend, pv, org, dir, tnear, tfar, alphaTransmittance, minTransmittance, cryptoShares, lpe, surfaceT);
}
template <class Backend>
X2_HD inline V3 subsurfaceRadianceCall(const mixed::Config& c, const sample::Identity& root, Backend& backend, U32 lightSet,
                                       const paths::Pending& pv, U32 subsurfaceDepth, const shade::State& st, const resume::Hit& hit, V3 rayDir,
                                       const shade::Bsdf& bsdf, bool doIndirect, paths::Pending* stack, U32& top, SampleStatus& status,
                                       const SubsurfaceLpe* lpe, shade::Color* ssAov) {
    return subsurfaceRadiance(c, root, backend, lightSet, pv, subsurfaceDepth, st, hit, rayDir, bsdf, doIndirect, stack, top, status, lpe, ssAov);
}
template <class Backend>
X2_HD inline V3 volumeTransmittanceCall(const mixed::Config& c, Backend& backend, paths::Stream& s, V3 org, V3 dir, float tnear, float tfar,
                                        float scale, int depth) {
    return volumeTransmittance(c, backend, s, org, dir, tnear, tfar, scale, depth);
}

template <class Backend>
X2_HD inline SampleStatus integratePaths(const mixed::Config& c, U32 pixel, U32 index, Backend& backend, Radiance& out) {
    using namespace paths;
    using shade::Color;
    if (pixel >= c.width * c.height || c.bsdfSamples == 0 || c.bsdfSamples > 64 || c.lightSamples == 0 || c.lightSamples > 64)
        return SampleStatus::Error;
    // A pixel sample map gives each pixel its own count: its samples past it are not taken (computeTotalNumSamples).
    const U32 pixelSpp = c.pixelSamples ? c.pixelSamples[pixel] : c.spp;
    if (c.pixelSamples && index >= pixelSpp) return SampleStatus::Skip;
    const sample::Identity root{pixel, index, c.seed, 0};
    float x, y;
    surface::Ray camera;
    if (!surface::filter(c.filter, c.filterWidth, sample::draw(root, 0, c.sampling), x) ||
        !surface::filter(c.filter, c.filterWidth, sample::draw(root, 1, c.sampling), y))
        return SampleStatus::Error;
    // The sample's time in the shutter (Sampler::getPrimarySample's, bent by the shutter bias): every ray of the
    // sample is traced at it.
    float time = 0.0f;
    if (c.motionBlur) {
        const float u = sample::draw(root, 4, c.sampling);
        if (c.shutterBias == 0.0f) {
            time = u;
        } else if (c.shutterBias > 0.0f) {
            time = fminf(powf(u, 1.0f / (1.0f + c.shutterBias)), 0x1.fffffep-1f);  // kMaxLessThanOne
        } else {
            time = 1.0f - powf(1.0f - u, 1.0f / (1.0f - c.shutterBias));
        }
    }
    backend.setTime(time);
    {
        // Camera::createRay with the sample's lens point (Sampler::fillPrimary draws one with depth of field alone).
        const float lensU = c.camera.dof ? sample::draw(root, 2, c.sampling) : 0.0f, lensV = c.camera.dof ? sample::draw(root, 3, c.sampling) : 0.0f;
        const surface::CameraResult made = surface::cameraSample(c.camera, float(pixel % c.width) + x, float(pixel / c.width) + y, lensU, lensV, time, camera);
        if (made == surface::CameraResult::None) return SampleStatus::Skip;
        if (made != surface::CameraResult::Made) return SampleStatus::Error;
    }

    // The pending rays and the hit's Bsdf live in the backend's per-thread scratch: on the GPU,
    // global memory, so the path tracer's state across a trace is a few registers, not kilobytes.
    Pending* stack = backend.pending();
    shade::Bsdf& bsdf = backend.bsdf();
    U32 top = 0;
    stack[top++] = Pending{camera.origin, camera.direction, {1, 1, 1}, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0}, 0, shade::RayDiff{}};
    // MoonRay's primary ray differential scale (initPrimaryRay): its scatterAndScale scales a diffuse lobe's by it.
    const float primaryScale = 1.0f / fminf(sqrtf(float(pixelSpp ? pixelSpp : 1)), 2.0f);
    // The camera ray's footprint (its differentials scaled, as initPrimaryRay's scaleDifferentials does) and the
    // camera's position, render space's origin.
    const float cameraFootprint = dirFootprint(surface::mul(surface::add(camera.dx, surface::mul(camera.direction, -1.0f)), primaryScale),
                                               surface::mul(surface::add(camera.dy, surface::mul(camera.direction, -1.0f)), primaryScale));
    const V3 cameraPosition = camera.origin;
    V3 radiance{0, 0, 0};
    float alpha = 0.0f;  // fillBundledRadiance: each camera-depth hit adds its path pixel weight
    // AOVs: the sample's values, and the light path expressions' state from the camera event.
    const aov::Tables& at = backend.aovs();
    float* aovs = at.count ? out.aov : nullptr;
    float* aovDepths = out.aovDepth;
    const bool lpeOn = aovs && (at.types & (aov::TypeLight | aov::TypeVisibility)) && at.lpeCamera >= 0;
    if (aovs) aov::init(at, aovs, aovDepths);
    stack[0].lpe = lpeOn ? at.lpeCamera : -1;
    // Cryptomatte: the camera ray's events, and each path's radiance (and a camera hit's alpha) credited to its event
    // as the next path is taken up; paths pushed meanwhile take their parent's role.
    const bool cryptoOn = at.crypto && out.crypto && out.cryptoCapacity;
    out.cryptoCount = 0;
    stack[0].crypto = cryptoOn ? aov::CryptoCamera : aov::CryptoNone;
    int cryptoOwner = aov::CryptoNone;
    U32 cryptoBase = 0;
    V3 cryptoMark = radiance;
    float cryptoAlphaMark = alpha;
    auto cryptoCredit = [&]() {
        if (!cryptoOn) return;
        for (U32 k = cryptoBase; k < top; ++k)
            if (stack[k].crypto == aov::CryptoInherit) stack[k].crypto = cryptoOwner >= 0 ? cryptoOwner : aov::CryptoNone;
        if (cryptoOwner >= 0) {
            aov::CryptoEvent& e = out.crypto[cryptoOwner];
            if (e.credit) {
                const V3 d = radiance - cryptoMark;
                e.beauty[0] += d.x * e.scale;
                e.beauty[1] += d.y * e.scale;
                e.beauty[2] += d.z * e.scale;
                if (e.radianceAlpha) e.beauty[3] += (alpha - cryptoAlphaMark) * e.scale;
            }
        }
        cryptoMark = radiance;
        cryptoAlphaMark = alpha;
    };
    // An event of the sample (CryptoNone when there is no room left).
    auto cryptoEvent = [&]() -> int {
        if (out.cryptoCount >= out.cryptoCapacity) return aov::CryptoNone;
        out.crypto[out.cryptoCount] = aov::CryptoEvent{};
        out.crypto[out.cryptoCount].scale = 1.0f;
        out.crypto[out.cryptoCount].presenceDepth = aov::CryptoNoDepth;
        return int(out.cryptoCount++);
    };
    // A light's or a lobe's label in the expressions.
    auto lobeLpeLabel = [&](const shade::Lobe& l) { return shade::labelLpeId(l.label); };
    while (top) {
        cryptoCredit();
        Pending pv = stack[--top];
        cryptoBase = top;
        cryptoOwner = pv.crypto;
        const bool fromCamera = pv.depth == 0 && pv.presenceDepth == 0;
        // The camera ray keeps its differentials, for the texture footprint of the first hit.
        const surface::Ray ray{pv.origin, pv.dir, fromCamera ? camera.dx : V3{0, 0, 0}, fromCamera ? camera.dy : V3{0, 0, 0},
                               fromCamera ? camera.nearT : pv.nearT, fromCamera ? camera.farT : pv.farT, rayMask(pv.lobeType),
                               fromCamera ? camera.ox : V3{0, 0, 0}, fromCamera ? camera.oy : V3{0, 0, 0}};
        const resume::Hit hit = backend.trace(ray);
        if (backend.deferred()) return SampleStatus::Defer;
        // Bounce-ray reuse: the light this ray's BSDF sample reached is reached unless the ray met something short of
        // the shadow ray's end; its light event as the BSDF sample's own (addDirectVisibleBsdfLobeSampleContributionBundled).
        if (pv.heldLight != ~0u) {
            const float kept = hit.geom != ~0u && hit.t < pv.heldFar ? lightShadowKept(backend.light(pv.heldLight), pv.heldFar) : 1.0f;
            if (kept > 0.0f) radiance += kept < 1.0f ? pv.held * kept : pv.held;
            if (lpeOn) {
                if (kept >= 1.0f) {
                    const Color through = pv.held;
                    aov::light(at, aovs, aovDepths, pv.held, &through, aov::PrefixUnoccluded, pv.lpeLight);
                } else {
                    aov::light(at, aovs, aovDepths, pv.held, nullptr, aov::PrefixUnoccluded, pv.lpeLight);
                }
            }
        }
        // The volumes the ray passes on its way (computeRadianceVolume): their light, and the throughput through them.
        const U32 volumeDepthBefore = pv.volumeDepth;
        V3 volumeAlphaT{1, 1, 1}, volumeMinT{0, 0, 0};
        const bool cryptoVolumes = (c.modoSwitches & 4u) && cryptoOn && pv.depth == 0 && pv.lobeType == 0;
        VolumeCryptoShares volumeShares;
        V3 volumeLight{0, 0, 0};
        float volumeSurfaceT = 3.402823466e38f;
        if (c.volumeCount) {
            const VolumeLpe volumeLpe{&at, aovs, aovDepths};
            volumeLight = volumeRadianceCall(c, root, backend, pv, ray.origin, ray.direction, ray.nearT, hit.geom == ~0u ? ray.farT : hit.t, &volumeAlphaT,
                                         &volumeMinT, cryptoVolumes ? &volumeShares : nullptr, lpeOn && pv.lpe >= 0 ? &volumeLpe : nullptr,
                                         pv.depth == 0 ? &volumeSurfaceT : nullptr);
            if (backend.deferred()) return SampleStatus::Defer;
            radiance += volumeLight;
        }
        // MOONRAY_MODO_CRYPTO_VOLUMES: the volumes as samples of their IDs, each its share of the coverage at its depth,
        // the volumes' light over their opacity as beauty; what lies behind covered as much as shows through.
        if (cryptoVolumes && volumeShares.marched) {
            const float weight = pv.pixelWeight * pv.cryptoFront, opacity = 1.0f - volumeShares.shown;
            if (weight > 0.0f && opacity > 0.0f) {
                const float k = 1.0f / fmaxf(weight * opacity, 0.01f);
                const V3 normal = surface::mul(shade::normalize(aov::renderVectorOf(c.worldToRender, ray.direction)), -1.0f);
                for (U32 i = 0; i < volumeShares.count; ++i) {
                    const volume::Shader& vs = backend.volumes()[volumeShares.share[i].volume];
                    if (!vs.cryptoValid) continue;
                    const int e = cryptoEvent();
                    if (e < 0) break;
                    aov::CryptoEvent& ev = out.crypto[e];
                    ev.id = vs.cryptoId;
                    ev.materialId = vs.cryptoMaterialId;
                    ev.assetId = vs.cryptoAssetId;
                    ev.weight = weight * volumeShares.share[i].coverage;
                    ev.beauty[0] = volumeLight.x * k;
                    ev.beauty[1] = volumeLight.y * k;
                    ev.beauty[2] = volumeLight.z * k;
                    ev.beauty[3] = 1.0f;
                    const V3 at = surface::add(ray.origin, surface::mul(ray.direction, volumeShares.share[i].depth / volumeShares.share[i].coverage));
                    ev.position = ev.refP = aov::renderPointOf(c.worldToRender, at);
                    ev.p0 = V3{0, 0, 0};
                    ev.normal = ev.refN = normal;
                    ev.uv[0] = ev.uv[1] = 0.0f;
                    ev.presenceDepth = pv.presenceDepth;
                    ev.increment = 1;
                    ev.scale = 1.0f;
                    ev.radianceAlpha = 0;
                    ev.credit = 0;
                    ev.used = 1;
                }
            }
            const float front = fminf(fmaxf(pv.cryptoFront * volumeShares.shown, 0.0f), 1.0f);
            pv.cryptoFront = float(unsigned(front * 65535.0f + 0.5f)) * (1.0f / 65535.0f);
        }
        // MOONRAY_MODO_FOG_ALPHA: the volumes this camera ray passed count once in its alpha, through those before.
        if ((c.modoSwitches & 2u) && pv.depth == 0) {
            const float shown = (volumeAlphaT.x + volumeAlphaT.y + volumeAlphaT.z) / 3.0f;
            pv.fogPending += pv.pixelWeight * pv.fogFront * (1.0f - shown);
            pv.fogFront *= shown;
        }
        // A hit's transparency (0 for what is there) lerped from the holdouts' minimum (computeRadianceRecurse's end).
        const float hitAlpha = 1.0f - (volumeMinT.x + volumeMinT.y + volumeMinT.z) / 3.0f;
        // A camera ray that passed volumes and hits nothing: their opacity as its alpha (reduceTransparency, the
        // average of the alpha transmittance).
        if (hit.geom == ~0u && pv.depth == 0)
            alpha += (c.modoSwitches & 2u) ? pv.fogPending : pv.pixelWeight * (1.0f - (volumeAlphaT.x + volumeAlphaT.y + volumeAlphaT.z) / 3.0f);
        // This ray's footprint, for the mip levels of textured lights (the camera's, or its differentials').
        const float footprint = fromCamera ? cameraFootprint : pv.diff.valid ? dirFootprint(pv.diff.dDx, pv.diff.dDy) : lights::NoFootprint;
        // The lights the camera sees, along a primary ray (presence continuations included).
        if (pv.depth == 0 && c.visibleLights) {
            const bool missed = hit.geom == ~0u;
            U32 hitLight = ~0u;
            const V3 Lv = pv.T * visibleLightRadiance(c, root, backend, missed ? ray.origin : cameraPosition, ray.direction,
                                                      missed ? 3.4028234663852886e+38f : hit.t, missed, footprint, pv.presenceDepth, &hitLight);
            radiance += Lv;
            // LPE: the camera saw a light in front of a surface (PathIntegratorBundled). One seen past every surface adds
            // nothing: MoonRay's miss handler (RayHandlers.cc, its case 1) declares the state anew inside its branch, so
            // the state it accumulates with stays -1. MOONRAY_MODO_CAMERA_LIGHT_LPE keeps it (tools/port_camera_light_lpe.py).
            if (lpeOn && hitLight != ~0u && (!missed || (c.modoSwitches & 8u)))
                aov::light(at, aovs, aovDepths, Lv, nullptr, 0, aov::lpeNext(at, pv.lpe, aov::LpeLight, backend.light(hitLight).lpeLabel));
        }
        if (hit.geom == ~0u) {
            // A camera ray that met a volume's surface and hits nothing: P, Wp and depth there (aovSetStateVarsVolumeOnly).
            if (pv.depth == 0 && aovs && (at.types & aov::TypeState) && volumeSurfaceT < 3.402823466e38f) {
                const V3 wp = surface::add(ray.origin, surface::mul(ray.direction, volumeSurfaceT));
                const V3 rp = aov::renderPointOf(c.worldToRender, wp);
                const float z = -volumeSurfaceT * aov::renderVectorOf(c.worldToRender, ray.direction).z;
                for (U32 ei = 0; ei < at.count; ++ei) {
                    const aov::Entry& e = at.entries[ei];
                    if (e.type != aov::TypeState) continue;
                    const float w = e.filter == aov::FilterAvg ? pv.pixelWeight : 1.0f;
                    float v[3] = {0, 0, 0};
                    if (e.state == aov::StateP) { v[0] = rp.x * w; v[1] = rp.y * w; v[2] = rp.z * w; }
                    else if (e.state == aov::StateWp) { v[0] = wp.x * w; v[1] = wp.y * w; v[2] = wp.z * w; }
                    else if (e.state == aov::StateDepth) v[0] = z * w;
                    else continue;
                    aov::put(e, aovs, aovDepths, v, aov::infinity());
                }
            }
            // An indirect ray that hits nothing: its volumes' light again, under the state of the light its BSDF
            // sample reached (MoonRay's miss handler, case 2).
            if (lpeOn && pv.depth > 0 && pv.lpeLight >= 0) aov::light(at, aovs, aovDepths, volumeLight, nullptr, 0, pv.lpeLight);
            // A camera ray that hits nothing: as many visibility misses as the light samples a hit would have taken.
            if (pv.depth == 0 && aovs && (at.types & aov::TypeVisibility))
                aov::visibility(at, aovs, 0.0f, float(c.lightSamples * c.lightCount), -1, true);  // lightSamples: light_samples squared
            // The background's extra AOV ('U:bg_white', aovAccumBackgroundExtraAovsBundled): white times the path's
            // throughput (its volumes' transmittance in it).
            if (lpeOn && at.lpeBackground >= 0) {
                const int s = aov::lpeNext(at, pv.lpe, aov::LpeExtra, at.lpeBackground);
                if (s >= 0) aov::light(at, aovs, aovDepths, pv.T, nullptr, 0, s);
            }
            continue;  // a miss adds nothing more: lights are reached through the BSDF samples
        }

        shade::State st;
        st.minRoughness[0] = pv.minRoughness[0];
        st.minRoughness[1] = pv.minRoughness[1];
        st.pathType = pv.diffuseDepth ? shade::PathIndirectDiffuse
                      : pv.glossyDepth ? shade::PathIndirectGlossy
                      : pv.mirrorDepth ? shade::PathIndirectMirror
                                       : shade::PathPrimary;
        st.causticPath = pv.diffuseDepth > 0;
        st.incoming = pv.diff;
        st.subsurfaceAllowed = pv.subsurfaceDepth < c.maxSubsurfacePerPath;
        st.rayLobe = pv.lobeType;
        st.textureDiffScale = c.textureDiffScale;
        st.pixelSamples = c.pixelSamples ? pixelSpp : 0;
        float presence = 1.0f;
        shade::reset(bsdf);
        const U32 status = backend.shade(hit, ray, st, bsdf, presence);
        if (status == tex::Miss) return SampleStatus::Miss;
        if (status != tex::Ready) return SampleStatus::Error;
        // A hit without an attribute its shading requires (ShadeBundleHandler's hasAllRequiredAttributes): the fatal
        // colour and the path's pixel weight, and nothing else; no AOV, no light path expression, no Cryptomatte.
        if (bsdf.fatal) {
            if (pv.depth == 0) alpha += pv.pixelWeight;
            radiance += pv.T * bsdf.selfEmission;
            continue;
        }
        // The hit's state for its AOVs (ShadeBundleHandler: state variables and primitive attributes at a camera-depth
        // hit, weighted by the path pixel weight it arrived with).
        // Camera::computeZDistance: a projective camera's camera space -z, any other's ray distance.
        const bool projective = c.camera.type == surface::CameraPerspective || c.camera.type == surface::CameraOrthographic;
        const aov::StateContext sc{c.worldToRender, cameraPosition, hit.t, projective,
                                   -volumeSurfaceT * aov::renderVectorOf(c.worldToRender, ray.direction).z, &c.camera, time, c.motionBlur,
                                   volumeSurfaceT, surface::add(ray.origin, surface::mul(ray.direction, volumeSurfaceT))};
        const float hitDepth = aov::cameraDepth(sc, st.P);  // the closest filter's depth
        if (aovs && pv.depth == 0 && (at.types & (aov::TypeState | aov::TypePrim)))
            for (U32 ei = 0; ei < at.count; ++ei) {
                const aov::Entry& e = at.entries[ei];
                const float w = e.filter == aov::FilterAvg ? pv.pixelWeight : 1.0f;
                float v[3] = {0, 0, 0};
                if (e.type == aov::TypeState) {
                    aov::stateValue(e.state, st, sc, w, v);
                } else if (e.type == aov::TypePrim) {
                    aov::primValueAt(st, at.primKeys[e.prim], v);
                    const float miss = e.filter == aov::FilterMin ? aov::infinity() : e.filter == aov::FilterMax ? -aov::infinity() : 0.0f;
                    for (U32 k = 0; k < 3; ++k) v[k] = v[k] == v[k] ? v[k] * w : miss;
                } else {
                    continue;
                }
                aov::put(e, aovs, aovDepths, v, hitDepth);
            }
        // Cryptomatte at a camera-depth hit (ShadeBundleHandler): a camera ray's hit counts its coverage and takes its
        // path's radiance as beauty; one reached through presence is settled below.
        const bool cryptoHere = cryptoOn && pv.depth == 0 && (pv.crypto == aov::CryptoCamera || pv.crypto == aov::CryptoThrough);
        const bool cryptoThrough = pv.crypto == aov::CryptoThrough;
        aov::CryptoEvent cryptoData{};
        if (cryptoHere) aov::cryptoHit(at, st, c.worldToRender, cryptoData);
        // A presence path's beauty alone (addBeautySampleVector): its radiance over its pixel weight, above 0.01 of it.
        auto cryptoBeauty = [&](float ppw) {
            const int b = cryptoEvent();
            if (b < 0) return aov::CryptoNone;
            aov::CryptoEvent& e = out.crypto[b];
            e.id = cryptoData.id;
            e.materialId = cryptoData.materialId;
            e.assetId = cryptoData.assetId;
            e.presenceDepth = pv.presenceDepth;
            e.scale = ppw == 0.0f ? 0.0f : 1.0f / ppw;
            e.credit = ppw > 0.01f ? 1u : 0u;
            return b;
        };
        // A presence surface's own sample (addSampleVector): its share of coverage, its presence as alpha.
        auto cryptoSample = [&](float weight, float a) {
            if (!(weight > 0.01f)) return;
            const int s = cryptoEvent();
            if (s < 0) return;
            aov::CryptoEvent& e = out.crypto[s];
            e = cryptoData;
            e.weight = weight * pv.cryptoFront;  // through the volumes before it (MOONRAY_MODO_CRYPTO_VOLUMES)
            e.beauty[0] = e.beauty[1] = e.beauty[2] = 0.0f;
            e.beauty[3] = a;
            e.presenceDepth = pv.presenceDepth;
            e.increment = 1;
            e.scale = 1.0f;
            e.credit = 0;
            e.used = 1;
        };
        // Presence (ShadeBundleHandler): below presence_quality's throughput a surface's presence is drawn as 0 or 1;
        // a presence short of one sends a continuation on through it with the rest of the throughput and pixel weight,
        // into empty space once the presence gathered so far reaches presence_threshold or the presence depth runs
        // out; the hit keeps its share, and one of no presence adds nothing itself.
        if (presence < 1.0f && shade::luminance(pv.T) < 1.0f - c.presenceQuality) {
            sample::Identity pid = root;
            pid.branch = childBranch(pv.branch, 0xfff9, pv.presenceDepth);
            Stream ps(pid, pv.depth, 0);
            presence = ps.next() < presence ? 1.0f : 0.0f;
        }
        // The material's extra AOVs (aovAccumExtraAovsBundled, not for a cut out): each one's value times the path's
        // throughput and the hit's presence, through its 'U:' event.
        if (lpeOn && bsdf.extraCount && !bsdf.earlyTermination)
            for (U32 i = 0; i < bsdf.extraCount; ++i) {
                if (bsdf.extraPost >> i & 1u) continue;
                const int s = aov::lpeNext(at, pv.lpe, aov::LpeExtra, bsdf.extraLabel[i]);
                if (s >= 0) aov::light(at, aovs, aovDepths, bsdf.extraValue[i] * pv.T * presence, nullptr, 0, s);
            }
        // With no room at all what lies behind goes unseen, rather than the whole frame failing.
        if (presence < 1.0f - lights::Epsilon && top < MaxPending) {
            const float totalPresence = (1.0f - pv.totalPresence) * presence;
            Pending through = pv;
            through.heldLight = ~0u;  // settled at this hit, where its shadow ray would have stopped (bounce-ray reuse)
            through.origin = hit.below;
            through.nearT = hit.epsilon;  // from a curve: on from the hit, past its epsilon
            // The differentials go on through it (transferred to the hit, a camera ray's scaled).
            through.diff = st.incoming;
            through.diff.valid = st.diffValid ? 1u : 0u;
            through.diff.dOx = st.dPdx + (st.P - hit.below);
            through.diff.dOy = st.dPdy + (st.P - hit.below);
            through.farT = (fromCamera ? camera.farT : pv.farT) - hit.t;
            if (totalPresence >= c.presenceThreshold || pv.presenceDepth >= c.maxPresenceDepth) {
                through.nearT = 3.402823466e+38f * 0.5f;
                through.farT = 3.402823466e+38f;
            }
            through.T = pv.T * (1.0f - presence);
            through.pixelWeight = pv.pixelWeight * (1.0f - presence);
            through.aovPixelWeight = pv.aovPixelWeight * (1.0f - presence);
            through.totalPresence = totalPresence;
            through.lpe = aov::lpeStraight(at, pv.lpe);  // presence is a straight event
            // Cryptomatte: a cut out's continuation starts afresh; another's is a presence path, and this surface
            // counts its share with its presence as alpha and its own radiance as beauty alone.
            through.crypto = !cryptoOn ? aov::CryptoNone : bsdf.earlyTermination ? aov::CryptoCamera : aov::CryptoThrough;
            if (cryptoHere && !bsdf.earlyTermination) {
                const float ppw = pv.pixelWeight * presence;
                cryptoSample(ppw, presence);
                pv.crypto = cryptoBeauty(ppw);
            }
            ++through.presenceDepth;
            through.volumeDepth = volumeDepthBefore;  // a presence continuation keeps its volume depth
            stack[top++] = through;
            pv.T = pv.T * presence;
            pv.pixelWeight *= presence;
            pv.aovPixelWeight *= presence;
        }
        // Material AOVs at this hit (aovAccumMaterialAovs), with the albedo of its BSDF samples where they are drawn.
        const int geomLabel = backend.geomLabel(hit.geom);
        float stateValues[3 * 14];
        float primValues[3 * 8];
        auto materialAovs = [&](const float* albedo, aov::Color ssAov) {
            if (!aovs || !(at.types & aov::TypeMaterial)) return;
            if (pv.depth != 0 && !at.materialSecondary) return;
            for (U32 s = 0; s < 14; ++s) aov::stateValue(aov::StateP + s, st, sc, 1.0f, stateValues + 3 * s);
            for (U32 k = 0; k < at.primCount && k < 8; ++k) aov::primValueAt(st, at.primKeys[k], primValues + 3 * k);
            unsigned char lobes[shade::MaxLobes];
            for (U32 k = 0; k < shade::MaxLobes; ++k) lobes[k] = 1;
            const aov::MaterialHit mh{&bsdf, &st, geomLabel, albedo, lobes, ssAov, c.worldToRender, pv.aovPixelWeight, pv.depth == 0, pv.lpe,
                                      stateValues, primValues};
            aov::materialAovs(at, aovs, aovDepths, mh, hitDepth);
        };
        // Cryptomatte at an opaque camera-depth hit: a camera ray's counts its coverage once with all its path's radiance;
        // one reached through presence counts its share (alpha one) and takes its radiance as beauty alone.
        if (cryptoHere && pv.crypto == (cryptoThrough ? aov::CryptoThrough : aov::CryptoCamera) && !bsdf.earlyTermination) {
            if (cryptoThrough) {
                cryptoSample(pv.pixelWeight, 1.0f);
                pv.crypto = cryptoBeauty(pv.pixelWeight);
            } else {
                const int e = cryptoEvent();
                if (e >= 0) {
                    out.crypto[e] = cryptoData;
                    out.crypto[e].weight = pv.cryptoFront;  // what shows through the volumes before it (one without them)
                    out.crypto[e].presenceDepth = aov::CryptoNoDepth;
                    out.crypto[e].increment = 1;
                    out.crypto[e].scale = 1.0f;
                    out.crypto[e].radianceAlpha = 1;
                    out.crypto[e].credit = 1;
                }
                pv.crypto = e;
            }
        }
        cryptoOwner = pv.crypto;
        // A shader's early termination (a cut out): nothing more from this hit.
        if (bsdf.earlyTermination) {
            materialAovs(nullptr, aov::Color{0, 0, 0});
            continue;
        }
        if (fabsf(presence) <= lights::Epsilon) continue;  // only the continuation counts
        if (pv.depth == 0) alpha += pv.pixelWeight * hitAlpha;
        if (cryptoOn && pv.crypto >= 0) out.crypto[pv.crypto].used = 1;  // the path's radiance reaches the film
        radiance += pv.T * bsdf.selfEmission;  // self-emission
        // LPE: an emission event, with the material's label.
        if (lpeOn && !shade::isBlack(bsdf.selfEmission))
            aov::light(at, aovs, aovDepths, pv.T * bsdf.selfEmission, nullptr, 0,
                       aov::lpeNext(at, pv.lpe, aov::LpeEmission, shade::labelLpeId(bsdf.materialLabels)));
        const bool subsurface = bsdf.bssrdf.type != shade::BssrdfNone;
        if (subsurface && hit.epsilon > 0.0f) return SampleStatus::Error;  // subsurface scattering on curves is not ported
        if (bsdf.topCount == 0 && !subsurface) {
            materialAovs(nullptr, aov::Color{0, 0, 0});
            continue;
        }
        // A hit with subsurface counts toward max_subsurface_per_path for everything after it.
        const U32 subsurfaceDepth = pv.subsurfaceDepth + (subsurface ? 1u : 0u);

        // indirectFlags: the lobe types whose depth limits still allow a bounce.
        U32 indirectFlags = 0;
        if (c.bsdfSamples > 0 && c.maxDepth > 0 && pv.depth < c.maxDepth) {
            if (pv.diffuseDepth < c.maxDiffuseDepth) indirectFlags |= shade::AllDiffuse;
            if (pv.glossyDepth < c.maxGlossyDepth) indirectFlags |= shade::AllGlossy;
            if (pv.mirrorDepth < c.maxMirrorDepth) indirectFlags |= shade::AllMirror;
            if (pv.hairDepth < c.maxHairDepth) indirectFlags |= shade::AllSurfaceSides;
        }
        const bool doIndirect = indirectFlags != 0;
        // The culling normal: none for a Bsdf with both sides or spherical lobes.
        V3 cull{0, 0, 0};
        const V3* cullN = nullptr;
        if (!subsurface && !bsdf.spherical && (bsdf.type & shade::AllSurfaceSides) != shade::AllSurfaceSides) {
            cull = (bsdf.type & shade::Reflection) ? st.Ng : -st.Ng;
            cullN = &cull;
        }
        const shade::Slice slice{st.Ng, st.wo, shade::AllLobeTypes, c.shadowTerminatorFix, true, st.entering};
        const bool split = pv.nonMirrorDepth == 0;
        const U32 nl = split ? c.lightSamples : 1;
        const bool clampHere = pv.nonMirrorDepth >= c.clampDepth;
        const bool rr = pv.nonMirrorDepth > 0 && c.rrThreshold > 0;
        const U32 lightSet = backend.lightSet(hit.geom);
        // Subsurface scattering (computeRadianceSubsurface): its exit samples' light, and their bounces pushed.
        aov::Color ssAov{0, 0, 0};
        if (subsurface) {
            SampleStatus status = SampleStatus::Done;
            SubsurfaceLpe sl;
            if (lpeOn) {
                sl.t = &at;
                sl.aovs = aovs;
                sl.depths = aovDepths;
                sl.backup = out.aovBackup;
                sl.state = aov::lpeSubsurface(at, pv.lpe, shade::labelLpeId(bsdf.bssrdf.label));
            }
            radiance += subsurfaceRadianceCall(c, root, backend, lightSet, pv, subsurfaceDepth, st, hit, ray.direction, bsdf, doIndirect,
                                           stack, top, status, lpeOn ? &sl : nullptr, aovs ? &ssAov : nullptr);
            if (status != SampleStatus::Done) return status;
        }
        // Sample counts per lobe (computeLobeSampleCount): mirrors take one.
        U32 lobeSamples[shade::MaxLobes];
        for (U32 k = 0; k < bsdf.topCount; ++k)
            lobeSamples[k] = (!split || (shade::topLobe(bsdf, k).type & shade::Mirror)) ? 1 : c.bsdfSamples;
        U32 plannedBounces = 0;
        for (U32 k = 0; k < bsdf.topCount; ++k) plannedBounces += lobeSamples[k];
        const float thinning = bounceThinning(top, plannedBounces);
        const V3 above = hit.origin, below = hit.below;
        // A curve's rays (PathIntegrator: the hit's epsilon hint): shadow rays skip its epsilon and reach at least twice
        // it; bounces skip the epsilon over the cosine to Ng.
        const float curveEpsilon = hit.epsilon;

#if !defined(__CUDACC__)
        // Emissive volumes (computeRadianceEmissiveRegionsBundled): per region, the lobes its geometry is visible to as
        // a BsdfOneSampler (picked by albedo), light samples (a place in its emission distribution) and BSDF samples,
        // MIS-weighted, each through the region's emissive volume integral.
        [&]() X2_OUT_OF_LINE {
            const volume::Regions er = backend.emissiveRegions();
            const U32 nL = split ? c.lightSamples : 1, nB = split ? c.bsdfSamples : 1;
            for (U32 ri = 0; ri < er.count; ++ri) {
                const volume::Region& R = er.regions[ri];
                U32 recs[shade::MaxLobes];
                float cdf[shade::MaxLobes];
                U32 nRec = 0;
                float norm = 0.0f;
                bool spherical = false;
                for (U32 k = 0; k < bsdf.topCount; ++k) {
                    const shade::Lobe& l = bsdf.records[bsdf.top[k]];
                    if (!(lights::lobeRayMask(l.type) & R.visibility)) continue;
                    recs[nRec] = bsdf.top[k];
                    cdf[nRec] = fmaxf(fabsf(shade::luminance(shade::lobeAlbedo(bsdf, bsdf.top[k], slice))), volume::Epsilon);
                    if (l.flags & shade::FlagSpherical) spherical = true;
                    norm += cdf[nRec++];
                }
                if (!nRec) continue;
                {
                    float acc = 0.0f;
                    for (U32 j = 0; j < nRec; ++j) {
                        cdf[j] = cdf[j] / norm + acc;
                        acc = cdf[j];
                    }
                }
                const int label = backend.volumes()[R.volume].lpeLabel;
                Color fs[shade::MaxLobes];
                U32 matched[shade::MaxLobes], nMatched = 0;
                // BsdfOneSampler::eval: the matching lobes' sum (each kept), the pdf weighted by the lobes' picks.
                auto oneEval = [&](V3 wi, float& pdf) {
                    const U32 flags = shade::surfaceFlags(slice, spherical, wi);
                    Color f{0, 0, 0};
                    pdf = 0.0f;
                    nMatched = 0;
                    float prev = 0.0f;
                    for (U32 j = 0; j < nRec; ++j) {
                        float p;
                        const Color col = shade::lobeEval(bsdf, recs[j], slice, wi, p);
                        if (shade::matchesFlags(bsdf.records[recs[j]].type, flags)) {
                            f = f + col;
                            fs[nMatched] = col;
                            matched[nMatched++] = recs[j];
                        }
                        pdf += p * (cdf[j] - prev);
                        prev = cdf[j];
                    }
                    return f;
                };
                auto emitted = [&](const V3& wi, float tEnd, Stream& es) {
                    const V3 from = curveEpsilon > 0.0f ? st.P : (shade::dot(wi, st.Ng) >= 0.0f ? above : below);
                    return emissiveVolumeIntegral(c, backend, es, from, wi, curveEpsilon > 0.0f ? curveEpsilon : 0.0f, tEnd, R.volume,
                                                  int(pv.depth) + 1);
                };
                auto add = [&](V3 contribution) {
                    for (U32 j = 0; j < nMatched; ++j) {
                        const Color lc = fs[j] * contribution;
                        radiance += lc;
                        if (lpeOn) {
                            const shade::Lobe& l = bsdf.records[matched[j]];
                            const int s1 = aov::lpeScatter(at, pv.lpe, l.type, shade::labelLpeId(l.label));
                            aov::light(at, aovs, aovDepths, lc, nullptr, 0, aov::lpeNext(at, s1, aov::LpeEmission, label));
                        }
                    }
                };
                for (U32 ls = 0; ls < nL; ++ls) {
                    sample::Identity eid = root;
                    eid.branch = childBranch(pv.branch, 0xfd00u + ri, ls);
                    Stream es(eid, pv.depth, 0);
                    const float u1 = es.next(), u2 = es.next(), u3 = es.next();
                    float pdfLight, tEnd;
                    const V3 wi = volume::regionSample(R, er.pool, st.P, u1, u2, u3, pdfLight, tEnd);
                    if (pdfLight == 0.0f) continue;
                    float pdfBsdf;
                    const Color f = oneEval(wi, pdfBsdf);
                    if (f.x == 0.0f && f.y == 0.0f && f.z == 0.0f) continue;
                    V3 contribution = emitted(wi, tEnd, es);
                    const float mis = nB ? lights::powerHeuristic(float(nL) * pdfLight, float(nB) * pdfBsdf) : 1.0f;
                    contribution = volume::scaled(volume::times(contribution, pv.T), mis / (pdfLight * float(nL)));
                    add(contribution);
                }
                for (U32 bs = 0; bs < nB; ++bs) {
                    sample::Identity eid = root;
                    eid.branch = childBranch(pv.branch, 0xfc00u + ri, bs);
                    Stream es(eid, pv.depth, 0);
                    const float uLobe = es.next(), u1 = es.next(), u2 = es.next();
                    // BsdfOneSampler::sample: a lobe by its pick, sampled; the others' matching values added.
                    U32 pick = 0;
                    while (pick + 1 < nRec && cdf[pick] <= uLobe) ++pick;
                    const float lobePdf = pick == 0 ? cdf[0] : cdf[pick] - cdf[pick - 1];
                    V3 wi;
                    float pdfBsdf;
                    Color f = shade::lobeSample(bsdf, recs[pick], slice, u1, u2, wi, pdfBsdf);
                    if (pdfBsdf == 0.0f) continue;
                    pdfBsdf *= lobePdf;
                    nMatched = 0;
                    fs[nMatched] = f;
                    matched[nMatched++] = recs[pick];
                    if (!(bsdf.records[recs[pick]].type & shade::Mirror)) {
                        const U32 flags = shade::surfaceFlags(slice, spherical, wi);
                        if (!shade::matchesFlags(bsdf.records[recs[pick]].type, flags)) f = Color{0, 0, 0};
                        float prev = 0.0f;
                        for (U32 j = 0; j < nRec; ++j) {
                            if (j != pick) {
                                float p;
                                const Color col = shade::lobeEval(bsdf, recs[j], slice, wi, p);
                                if (shade::matchesFlags(bsdf.records[recs[j]].type, flags)) {
                                    f = f + col;
                                    fs[nMatched] = col;
                                    matched[nMatched++] = recs[j];
                                }
                                pdfBsdf += p * (cdf[j] - prev);
                            }
                            prev = cdf[j];
                        }
                    }
                    if (pdfBsdf == 0.0f || (f.x == 0.0f && f.y == 0.0f && f.z == 0.0f)) continue;
                    float tEnd;
                    const float pdfLight = volume::regionPdf(R, er.pool, st.P, wi, tEnd);
                    if (pdfLight == 0.0f) continue;
                    V3 contribution = emitted(wi, tEnd, es);
                    const float mis = nL ? lights::powerHeuristic(float(nB) * pdfBsdf, float(nL) * pdfLight) : 1.0f;
                    contribution = volume::scaled(volume::times(contribution, pv.T), mis / (pdfBsdf * float(nB)));
                    add(contribution);
                }
            }
        }();
#endif
        // Light samples (drawLightSamples, integrateLightSetSample): each light of the hit's set; with adaptive light
        // sampling, the lights outside the set's tree and those the tree picks (chooseLightsToSample), a sample's pdf
        // times the probability its light was picked while the MIS weights keep the light's own (misPdf).
        const U64 lightBranch = childBranch(pv.branch, 0xfffe, 0);
        // Visibility AOV misses (accumVisibilityAovsOccluded): at camera depth through mirrors, a light sample that
        // fails (Light::sample, isSampleInvalid) or a light culled from the hit (computeActiveLights) counts as attempts
        // that missed, through every lobe whose state has a visibility AOV: MoonRay's vector loop means to stop at the
        // first, but its insert(addVisibility, lane, false) discards its result, so it never stops.
        const bool visibilityMisses = lpeOn && pv.nonMirrorDepth == 0 && (at.types & aov::TypeVisibility);
        auto visibilityMiss = [&](const lights::Light& L, float attempts) {
            for (U32 k = 0; k < bsdf.topCount; ++k) {
                const shade::Lobe& lobe = shade::topLobe(bsdf, k);
                const int state = aov::lpeNext(at, aov::lpeScatter(at, pv.lpe, lobe.type, lobeLpeLabel(lobe)), aov::LpeLight, L.lpeLabel);
                aov::visibility(at, aovs, 0.0f, attempts, state);
            }
        };
        // A camera-depth hit's samples count toward the visibility AOVs even when they carry nothing.
        const bool visibilityHits = lpeOn && pv.depth == 0 && (at.types & aov::TypeVisibility);
        auto lightSamples = [&](U32 i, float selection) {
            const lights::Light& L = backend.light(i);
            const U64 lightAddress = childBranch(lightBranch, i, 0);
            sample::Identity lid = root;
            lid.branch = lightAddress;
            Stream ls(lid, pv.depth, 0);
            for (U32 s = 0; s < nl; ++s) {
                float r[3] = {ls.next(), ls.next(), ls.next()};
                const float u = ls.next();
                if (split) {
                    // The hit's samples of a light are one stratified set (policy bit 4).
                    const sample::Identity strat{pixel, U64(index) * nl + s, c.seed, lightAddress};
                    r[0] = sample::draw(strat, sample::CameraDimensions + 2, c.sampling);
                    r[1] = sample::draw(strat, sample::CameraDimensions + 3, c.sampling);
                }
                V3 wi;
                lights::Hit lh;
                if (!lights::sample(L, st.P, cullN, r, wi, lh, footprint)) {
                    if (visibilityMisses) visibilityMiss(L, 1.0f);
                    continue;
                }
                float pdfL;
                V3 Li = lights::eval(L, st.P, wi, lh, pdfL, footprint);
                if (c.lightFilters) {
                    sample::Identity fid = lid;
                    fid.branch = childBranch(lightAddress, s, 0xfff2);
                    Li = filterRadiance(backend, hit.geom, i, L, st.P, wi, lh.distance, pv, fid, Li);
                }
                if (shade::isSampleInvalid(Li, pdfL) || !(fabsf(pdfL) < 3.4e38f) || !shade::finite3(Li)) {
                    if (visibilityMisses) visibilityMiss(L, 1.0f);
                    continue;
                }
                const Color factor = pv.T * Li * (1.0f / (float(nl) * pdfL * selection));
                const U32 flags = shade::surfaceFlags(slice, bsdf.spherical, wi);
                Color t{0, 0, 0};
                bool any = false;
                // LightSampleLPE: each lobe's share and its LPE state as if the light were reached.
                Color lobeT[shade::MaxLobes];
                int lobeLpe[shade::MaxLobes];
                for (U32 k = 0; k < bsdf.topCount; ++k) {
                    lobeLpe[k] = -1;
                    const shade::Lobe& lobe = shade::topLobe(bsdf, k);
                    if (!shade::matchesFlags(lobe.type, flags)) continue;
                    if (!(lights::lobeRayMask(lobe.type) & L.visibilityMask)) continue;
                    if (!lobeSeesLight(backend, pv, lobe.lightSet, i)) continue;
                    float pdf;
                    const Color f = shade::lobeEval(bsdf, bsdf.top[k], slice, wi, pdf);
                    if (shade::isSampleInvalid(f, pdf)) continue;
                    Color tk = factor * f * lights::powerHeuristic(float(nl) * pdfL, float(lobeSamples[k]) * pdf);
                    if (clampHere) tk = smartClamp(tk, c.clampValue);
                    t += tk;
                    any = true;
                    if (lpeOn) {
                        lobeT[k] = tk;
                        lobeLpe[k] = aov::lpeNext(at, aov::lpeScatter(at, pv.lpe, lobe.type, lobeLpeLabel(lobe)), aov::LpeLight, L.lpeLabel);
                    }
                }
                if (!any) continue;
                if (rr) {
                    float scale;
                    if (!roulette(shade::luminance(t), c.rrThreshold, u, scale)) continue;
                    t = t * scale;
                    if (lpeOn)
                        for (U32 k = 0; k < bsdf.topCount; ++k) lobeT[k] = lobeT[k] * scale;
                }
                if (!shade::finite3(t) || (shade::isExactlyZero(t) && !visibilityHits)) continue;
                const V3 shadowOrigin = toonShadowOrigin(st, curveEpsilon > 0.0f ? st.P : (shade::dot(wi, st.Ng) >= 0.0f ? above : below), wi);
                float shadowNear = curveEpsilon > 0.0f ? curveEpsilon : 0.0f, shadowFar;
                lightShadowSpan(L, lh.distance, shadowNear, shadowFar);
                float kept;
                backend.shadowFor(i, hit.geom);
                if (L.rayTerminator) {
                    kept = 1.0f;  // FORCE_NOT_OCCLUDED
                } else if (L.presenceShadows) {
                    // The presence shadows' queue: the light less the presence it passes.
                    sample::Identity sid = lid;
                    sid.branch = childBranch(lightAddress, s, 0xfff8);
                    Stream ps(sid, pv.depth, 0);
                    kept = 1.0f - rayPresence(c, backend, shadowOrigin, wi, shadowNear, shadowFar, pv.depth,
                                              shade::luminance(pv.T) < 1.0f - c.presenceQuality, ps);
                } else {
                    backend.countRay(RayCountNee);
                    kept = backend.visible(shadowOrigin, wi, shadowNear, shadowFar) ? 1.0f : lightShadowKept(L, shadowFar);
                }
                Color tr{1, 1, 1};
                if (kept > 0.0f) {
                    if (kept < 1.0f) t = t * kept;
                    if (c.volumeCount) {
                        // The volumes on the shadow ray's way (transmittance()).
                        sample::Identity tid = lid;
                        tid.branch = childBranch(lightAddress, s, 0xfffb);
                        Stream vs(tid, pv.depth, 0);
                        tr = volumeTransmittanceCall(c, backend, vs, shadowOrigin, wi, shadowNear, shadowFar, 1.0f, int(pv.depth) + 1);
                        t = t * tr;
                    }
                    radiance += t;
                }
                // LPE (the occlusion queue's accumLightAovs and accumVisibilityAovs): unoccluded, each lobe's share through
                // the volumes to its expressions, and to the unoccluded ones as it is; occluded, to the unoccluded ones only.
                // A camera-depth hit's sample counts toward the visibility AOVs once, through the first lobe whose state
                // has one.
                if (lpeOn) {
                    // Presence shadows always reach the light, less the presence they passed (presenceShadowsQueue).
                    const bool reached = L.presenceShadows || kept >= 1.0f;
                    const float pass = L.presenceShadows ? kept : 1.0f;
                    bool visibilityCounted = !(pv.depth == 0 && (at.types & aov::TypeVisibility));
                    for (U32 k = 0; k < bsdf.topCount; ++k) {
                        if (lobeLpe[k] < 0) continue;
                        if (reached) {
                            const Color through = lobeT[k] * tr * pass;
                            aov::light(at, aovs, aovDepths, lobeT[k], &through, aov::PrefixUnoccluded, lobeLpe[k]);
                        } else {
                            aov::light(at, aovs, aovDepths, lobeT[k], nullptr, aov::PrefixUnoccluded, lobeLpe[k]);
                        }
                        if (!visibilityCounted)
                            visibilityCounted = aov::visibility(at, aovs, reached ? pass * (tr.x + tr.y + tr.z) / 3.0f : 0.0f, 1.0f, lobeLpe[k]);
                    }
                }
            }
        };
        const int treeRoot = c.lightTree ? backend.treeRoot(lightSet) : -1;
        // computeActiveLights: the set's lights that can light the hit, the culling eased by the subsurface's reach;
        // the rest are not sampled (and count as the visibility AOVs' misses, a light's samples each).
        const float activeRadius = subsurface ? bsdf.bssrdf.maxRadius : 0.0f;
        for (U32 i = 0; i < c.lightCount; ++i) {
            if (!backend.inSet(lightSet, i)) continue;
            const lights::Light& L = backend.light(i);
            if (!lights::canIlluminate(L, st.P, cullN, activeRadius) ||
                (c.lightFilters && !filterAllows(backend, hit.geom, i, L, st.P, activeRadius, pv))) {
                if (visibilityMisses) visibilityMiss(L, float(c.lightSamples));
                continue;
            }
            if (treeRoot >= 0 && lights::inTree(L)) continue;
            lightSamples(i, 1.0f);
        }
        if (treeRoot >= 0) {
            // LightTree_sample about the hit's point and geometric normal, culling lights below it.
            sample::Identity tid = root;
            tid.branch = childBranch(lightBranch, 0xfffc, 0);
            Stream ts(tid, pv.depth, 0);
            lights::treeSample(backend.treeNodes(), treeRoot, c.lightTreeThreshold, st.P, st.Ng, [&]() { return ts.next(); },
                               [&](int light, float pdf) {
                                   // LightTree_chooseLight: a light culled from the hit is not in its set.
                                   if (lights::canIlluminate(backend.light(U32(light)), st.P, cullN, activeRadius) &&
                                       (!c.lightFilters || filterAllows(backend, hit.geom, U32(light), backend.light(U32(light)), st.P, activeRadius, pv)))
                                       lightSamples(U32(light), pdf);
                               });
        }

        // BSDF samples (drawBsdfSamples, integrateBsdfSample, addIndirectAndDirectVisibleContributions).
        float albedo[3 * shade::MaxLobes];
        for (U32 k = 0; k < 3 * shade::MaxLobes; ++k) albedo[k] = 0.0f;
        // The BSDF samples out of line (X2_OUT_OF_LINE); a full stack of pending paths ends the sample as before.
        [&]() X2_OUT_OF_LINE {
        for (U32 k = 0; k < bsdf.topCount; ++k) {
            const U32 record = bsdf.top[k];
            const shade::Lobe& lobe = bsdf.records[record];
            const U32 nk = lobeSamples[k];
            const bool mirror = (lobe.type & shade::Mirror) != 0, hair = (lobe.flags & shade::FlagHair) != 0;
            const U32 lobeMask = lights::lobeRayMask(lobe.type);
            const bool lobeIndirect = shade::matchesFlags(lobe.type, indirectFlags) || (hair && pv.hairDepth < c.maxHairDepth);
            const U64 lobeBranch = childBranch(pv.branch, k, 0xffff);
            // MOONRAY_MODO_TOON_ALBEDO (Aov.cc): a toon diffuse lobe's own albedo, not its samples' f / pdf.
            const bool toonAlbedo = (c.modoSwitches & 1u) && lobe.kind == shade::KindToon;
            if (toonAlbedo && nk) {
                const Color a = shade::lobeAlbedo(bsdf, record, slice);
                albedo[3 * k] = a.x;
                albedo[3 * k + 1] = a.y;
                albedo[3 * k + 2] = a.z;
            }
            for (U32 i = 0; i < nk; ++i) {
                const U64 sampleBranch = childBranch(lobeBranch, i, 0);
                sample::Identity bid = root;
                bid.branch = sampleBranch;
                Stream bs(bid, pv.depth, 0);
                float r1 = bs.next(), r2 = bs.next();
                const float choice = bs.next(), u = bs.next();
                if (split && !mirror) {
                    // The hit's samples of a lobe are one stratified set: Sobol indices index * nk + i.
                    const sample::Identity strat{pixel, U64(index) * nk + i, c.seed, lobeBranch};
                    r1 = sample::draw(strat, sample::CameraDimensions, c.sampling);
                    r2 = sample::draw(strat, sample::CameraDimensions + 1, c.sampling);
                }
                V3 wi;
                float pdf;
                const Color f = shade::lobeSample(bsdf, record, slice, r1, r2, wi, pdf);
                if (!shade::isSampleValid(f, pdf) || !shade::finite3(f) || !shade::finite3(wi)) continue;
                if (!shade::matchesFlags(lobe.type, shade::surfaceFlags(slice, bsdf.spherical, wi))) continue;
                // The material albedo AOV: the lobe's samples' f / (count pdf).
                if (!toonAlbedo) {
                    albedo[3 * k] += f.x / (float(nk) * pdf);
                    albedo[3 * k + 1] += f.y / (float(nk) * pdf);
                    albedo[3 * k + 2] += f.z / (float(nk) * pdf);
                }
                const Color pt = pv.T * f * (1.0f / (float(nk) * pdf));
                // The lights the ray meets: one chosen uniformly, its radiance times their number. Ray termination
                // lights count only where the lobe's bounces have ended (isLightActive's includeRayTerminationLights).
                const bool terminators = (lobe.type & ~indirectFlags & shade::AllLobes) != 0;
                U32 hits = 0;
                for (U32 j = 0; j < c.lightCount; ++j) {
                    lights::Hit lh;
                    if (backend.inSet(lightSet, j) && (lobeMask & backend.light(j).visibilityMask) &&
                        (terminators || !backend.light(j).rayTerminator) && lobeSeesLight(backend, pv, lobe.lightSet, j) &&
                        lights::intersect(backend.light(j), st.P, cullN, wi, 3.4028234663852886e+38f, lh))
                        ++hits;
                }
                Color direct{0, 0, 0};
                float lightDistance = 0;
                bool lightHit = false;
                U32 lightIndex = 0;
                if (hits) {
                    const U32 chosen = U32(choice * float(hits)) < hits ? U32(choice * float(hits)) : hits - 1;
                    for (U32 j = 0, seen = 0; j < c.lightCount; ++j) {
                        lights::Hit lh;
                        if (!backend.inSet(lightSet, j) || !(lobeMask & backend.light(j).visibilityMask) ||
                            !(terminators || !backend.light(j).rayTerminator) || !lobeSeesLight(backend, pv, lobe.lightSet, j) ||
                            !lights::intersect(backend.light(j), st.P, cullN, wi, 3.4028234663852886e+38f, lh))
                            continue;
                        if (seen++ != chosen) continue;
                        float pdfL;
                        V3 Li = lights::eval(backend.light(j), st.P, wi, lh, pdfL, footprint) * float(hits);
                        if (c.lightFilters) {
                            sample::Identity fid = bid;
                            fid.branch = childBranch(sampleBranch, 0xfff3, 0);
                            Li = filterRadiance(backend, hit.geom, j, backend.light(j), st.P, wi, lh.distance, pv, fid, Li);
                        }
                        if (!shade::isSampleInvalid(Li, pdfL) && fabsf(pdfL) < 3.4e38f && shade::finite3(Li)) {  // a negative pdf counts (MoonRay)
                            direct = mirror ? Li * pt : Li * pt * lights::powerHeuristic(float(nk) * pdf, float(nl) * pdfL);
                            lightDistance = lh.distance;
                            lightHit = true;
                            lightIndex = j;
                        }
                        break;
                    }
                }
                // A ray termination light reached with no bounce left lights a (non-hair) transmission lobe alone,
                // unoccluded (addDirectVisibleBsdfLobeSampleContribution).
                bool forceUnoccluded = false;
                if (lightHit && !doIndirect && backend.light(lightIndex).rayTerminator) {
                    if (!shade::matchesFlags(lobe.type, shade::AllTransmission) || hair) {
                        lightHit = false;
                        direct = Color{0, 0, 0};
                    } else {
                        forceUnoccluded = true;
                    }
                }
                if (!lightHit && !doIndirect) continue;  // an invalid sample (integrateBsdfSample)
                if (clampHere) direct = smartClamp(direct, c.clampValue);
                Color indirect = doIndirect ? pt : Color{0, 0, 0};
                if (rr) {
                    float scale;
                    if (!roulette(fmaxf(shade::luminance(direct), shade::luminance(indirect)), c.rrThreshold, u, scale)) continue;
                    direct = direct * scale;
                    indirect = indirect * scale;
                }
                const V3 origin = shade::dot(wi, st.Ng) >= 0.0f ? above : below;
                const V3 shadowFrom = toonShadowOrigin(st, origin, wi);
                float start = 0.0f;
                if (curveEpsilon > 0.0f) {
                    const float denom = fabsf(shade::dot(st.Ng, wi));
                    start = shade::isZero(denom) ? curveEpsilon : curveEpsilon / denom;
                }
                float shadowNear = start, shadowFar = 0.0f;
                float kept = 0.0f;
                // Whether the bounce goes on (pushed below): its indirect light, under the transparency threshold.
                float opacity = pv.accumOpacity;
                bool opaqueEnough = true;
                if (mirror && (lobe.type & shade::Transmission)) {
                    const float transparency = (f.x + f.y + f.z) / 3.0f;
                    opacity = pv.accumOpacity + (1 - transparency) * (1 - pv.accumOpacity);
                    opaqueEnough = !(opacity > c.transparencyThreshold);
                }
                const bool bounces = lobeIndirect && !shade::isExactlyZero(indirect) && shade::finite3(indirect) && opaqueEnough;
                // Bounce-ray reuse: the light reached is settled when the bounce is traced, which stands for its shadow ray.
                bool held = false;
                if (lightHit && !shade::isBlack(direct) && shade::finite3(direct)) {
                    const lights::Light& L = backend.light(lightIndex);
                    lightShadowSpan(L, lightDistance, shadowNear, shadowFar);
                    backend.shadowFor(lightIndex, hit.geom);
                    held = bounces && thinning >= 1.0f && top + PendingReserve < MaxPending && !forceUnoccluded && !L.presenceShadows && lobeMask && (c.shadowReuse & lobeMask) == lobeMask &&
                           !c.volumeCount && shadowNear == start && shadowFrom.x == origin.x && shadowFrom.y == origin.y && shadowFrom.z == origin.z;
                    if (held) {
                        backend.countRay(RayCountHeld);
                    } else if (forceUnoccluded) {
                        kept = 1.0f;
                    } else if (L.presenceShadows) {
                        sample::Identity sid = bid;
                        sid.branch = childBranch(sampleBranch, 0xfff8, 0);
                        Stream ps(sid, pv.depth, 0);
                        kept = 1.0f - rayPresence(c, backend, shadowFrom, wi, shadowNear, shadowFar, pv.depth,
                                                  shade::luminance(pv.T) < 1.0f - c.presenceQuality, ps);
                    } else {
                        backend.countRay(RayCountBsdf);
                        kept = backend.visible(shadowFrom, wi, shadowNear, shadowFar) ? 1.0f : lightShadowKept(L, shadowFar);
                    }
                }
                const Color directBefore = direct;
                Color tr{1, 1, 1};
                if (kept > 0.0f && !held) {
                    if (kept < 1.0f) direct = direct * kept;
                    if (c.volumeCount) {
                        sample::Identity tid = bid;
                        tid.branch = childBranch(sampleBranch, 0xfffb, 0);
                        Stream vs(tid, pv.depth, 0);
                        tr = volumeTransmittanceCall(c, backend, vs, shadowFrom, wi, shadowNear, shadowFar, 1.0f, int(pv.depth) + 1);
                        direct = direct * tr;
                    }
                    radiance += direct;
                }
                // LPE: the light the sample reached (addDirectVisibleBsdfLobeSampleContributionBundled; held: when the
                // bounce is traced).
                if (lpeOn && !held && lightHit && !shade::isBlack(directBefore) && shade::finite3(directBefore)) {
                    const int state = aov::lpeNext(at, aov::lpeScatter(at, pv.lpe, lobe.type, lobeLpeLabel(lobe)), aov::LpeLight,
                                                   backend.light(lightIndex).lpeLabel);
                    const bool presenceShadows = backend.light(lightIndex).presenceShadows;
                    const bool reached = presenceShadows || kept >= 1.0f;
                    if (reached) {
                        const Color through = directBefore * tr * (presenceShadows ? kept : 1.0f);
                        aov::light(at, aovs, aovDepths, directBefore, &through, aov::PrefixUnoccluded, state);
                    } else {
                        aov::light(at, aovs, aovDepths, directBefore, nullptr, aov::PrefixUnoccluded, state);
                    }
                }
                if (!bounces) continue;  // its indirect light, under the transparency threshold (mirror transmission lobes accumulate opacity)
                float keptShare = 1.0f;
                if (thinning < 1.0f || top + PendingReserve >= MaxPending) {
                    sample::Identity kid = bid;
                    kid.branch = childBranch(sampleBranch, 0xfff7, 0);
                    Stream chance(kid, pv.depth, 0);
                    if (thinning < 1.0f) {
                        if (chance.next() >= thinning) continue;
                        keptShare = 1.0f / thinning;
                    }
                    if (top + PendingReserve >= MaxPending) {
                        float tight = 1.0f;
                        if (!roomForChild(top, chance.next(), tight)) continue;
                        keptShare *= tight;
                    }
                }
                Pending child;
                child.pixelWeight = 0.0f;
                child.aovPixelWeight = pv.pixelWeight;
                child.lpe = lpeOn ? aov::lpeScatter(at, pv.lpe, lobe.type, lobeLpeLabel(lobe)) : -1;
                child.lpeLight = lpeOn && lightHit ? aov::lpeNext(at, child.lpe, aov::LpeLight, backend.light(lightIndex).lpeLabel) : -1;
                child.origin = origin;
                child.dir = wi;
                child.diff = shade::scatterDifferentials(bsdf, record, st, wi, r1, r2, origin, primaryScale);
                child.T = indirect * keptShare;
                child.branch = sampleBranch;
                child.depth = pv.depth + 1;
                child.diffuseDepth = pv.diffuseDepth + ((lobe.type & shade::Diffuse) ? 1 : 0);
                child.glossyDepth = pv.glossyDepth + ((lobe.type & shade::Glossy) ? 1 : 0);
                child.mirrorDepth = pv.mirrorDepth + (mirror ? 1 : 0);
                child.nonMirrorDepth = pv.nonMirrorDepth + (mirror ? 0 : 1);
                child.hairDepth = pv.hairDepth + (hair ? 1 : 0);
                child.presenceDepth = pv.presenceDepth;
                child.subsurfaceDepth = subsurfaceDepth;
                child.farT = 3.402823466e+38f;
                child.nearT = start;
                // Every bounce counts toward the volume depth (PathIntegratorMultiSampler: parent's + 1), so with
                // max_volume_depth one only camera rays gather volume scattering.
                child.volumeDepth = pv.volumeDepth + 1;
                child.accumOpacity = opacity;
                child.lobeType = lobe.type;
                inheritLobeSets(child, pv, lobe.lightSet);
                // computeMinRoughness: the path's roughness floor for later hair lobes.
                child.minRoughness[0] = child.minRoughness[1] = 0.0f;
                if (c.roughnessClamping > 0.0f) {
                    float ru, rv;
                    if (shade::lobeRoughness(bsdf, record, ru, rv)) {
                        child.minRoughness[0] = fminf(ru, rv) * c.roughnessClamping;
                        child.minRoughness[1] = fmaxf(ru, rv) * c.roughnessClamping;
                    }
                }
                child.minRoughness[0] = shade::clampf(child.minRoughness[0], pv.minRoughness[0], 1.0f);
                child.minRoughness[1] = shade::clampf(child.minRoughness[1], pv.minRoughness[1], 1.0f);
                if (held) {  // bounce-ray reuse: the light reached, settled when this bounce is traced
                    child.heldLight = lightIndex;
                    child.heldFar = shadowFar;
                    child.held = directBefore;
                }
                // The post scatter extra AOVs (aovAccumPostScatterExtraAovs): each one's value times the sample's
                // throughput, through its 'U:' event after the scattering.
                if (lpeOn && bsdf.extraPost)
                    for (U32 i = 0; i < bsdf.extraCount; ++i) {
                        if (!(bsdf.extraPost >> i & 1u)) continue;
                        const int s = aov::lpeNext(at, child.lpe, aov::LpeExtra, bsdf.extraLabel[i]);
                        if (s >= 0) aov::light(at, aovs, aovDepths, bsdf.extraValue[i] * child.T, nullptr, 0, s);
                    }
                stack[top++] = child;
            }
        }
        }();
        materialAovs(albedo, ssAov);
    }
    cryptoCredit();
    if (backend.textureMissed()) return SampleStatus::Miss;
    if (backend.deferred()) return SampleStatus::Defer;
    out.rgb[0] = radiance.x;
    out.rgb[1] = radiance.y;
    out.rgb[2] = radiance.z;
    out.alpha = alpha;
    return SampleStatus::Done;
}

// The GPU programs carry one integrator (XPU2_PATHS_ONLY): two in one kernel cost
// the path tracer registers.
template <class Backend>
X2_HD inline SampleStatus integrate(const mixed::Config& c, U32 pixel, U32 index, Backend& backend, Radiance& out) {
#if defined(XPU2_PATHS_ONLY)
    return integratePaths(c, pixel, index, backend, out);
#else
    return c.paths ? integratePaths(c, pixel, index, backend, out) : integrateDirect(c, pixel, index, backend, out);
#endif
}

// The unit of work (spec §7): one 8x8 tile of the image and a range of sample
// indices, not yet started. 16 bytes, so moving one across PCIe is cheap.
struct Chunk {
    U32 tile;   // tile index, row-major over the image's 8x8 tiles
    U32 first;  // first sample index
    U32 count;  // number of sample indices
    U32 cell;   // owning cell, for per-cell accounting
};
static_assert(sizeof(Chunk) == 16, "Chunks are 16 bytes");

// A provisional sample (partial shading, texture_core.h coarse()): one the GPU finished from a texture's next level up
// where a fine-detail tile was not yet resident. Its value stays apart from the film (the GPU's provisional film)
// until the sample is rendered again in full, by the CPU lane or by the GPU once its tile has landed, which takes it
// out. tile: the first missing tile (0: not known), for the host to know when the GPU may take it.
struct Refine {
    U32 pixel, index, pass, tile;
    float rgba[4];
};
static_assert(sizeof(Refine) == 32, "Provisional samples are 32 bytes");

constexpr U32 TileSize = 8, TilePixels = TileSize * TileSize;

// Adaptive frames (async_adaptive.h): a chunk's first field holds its pass and its
// sample offset within the pass; each active pixel's sample index is its own base
// (the samples it already has) plus that offset, so a pixel's indices stay
// contiguous, as MoonRay keeps them. Pass tables live in a ring of PassSlots.
constexpr U32 MaxPasses = 160, PassSlots = 4, PassShift = 16;
X2_HD inline U32 chunkPass(const Chunk& c) { return c.first >> PassShift; }
X2_HD inline U32 chunkOffset(const Chunk& c) { return c.first & ((1u << PassShift) - 1); }

// A rounded product and sum that the CUDA compiler may not fuse into one
// multiply-add, so the host (built without contraction) and the GPU work a value
// out alike.
X2_HD inline float mulRn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
X2_HD inline float addRn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

// One pixel's adaptive error from its sums, its odd-index sums and its sample
// count, on either device and alike on both (MoonRay's
// estimatePixelErrorInternal): the luma of the difference between the even and
// odd half-images, over the square root of the pixel's luma, plus the alpha
// difference. ErrorNotReady until both halves have a sample.
constexpr float ErrorNotReady = -1.f;
X2_HD inline float pixelError(const double total[4], const double odd[4], U32 samples) {
    const float n = float(samples), nOdd = float(samples / 2), nEven = n - nOdd;
    if (nOdd <= 0 || nEven <= 0) return ErrorNotReady;
    float diff[4], avg[3];
    for (int c = 0; c < 4; ++c) {
        const float t = float(total[c]), o = float(odd[c]);
        if (t != t || o != o) return 0.f;
        diff[c] = fabsf(addRn(mulRn(t, nOdd), -mulRn(o, n)) / mulRn(nOdd, nEven));
        if (c < 3) avg[c] = t / n;
    }
    auto luma = [](const float v[3]) {
        return addRn(addRn(mulRn(0.299f, v[0]), mulRn(0.587f, v[1])), mulRn(0.114f, v[2]));
    };
    const float lumDiff = luma(diff), lumAvg = luma(avg);
    if (lumAvg <= 0) return diff[3];
    return addRn(lumDiff / sqrtf(lumAvg), diff[3]);
}

// Pixel of the i-th position of a tile, or ~0u when it lies outside the image.
X2_HD inline U32 tilePixel(U32 tile, U32 position, U32 width, U32 height) {
    const U32 tilesX = (width + TileSize - 1) / TileSize;
    const U32 x = (tile % tilesX) * TileSize + position % TileSize;
    const U32 y = (tile / tilesX) * TileSize + position / TileSize;
    return x < width && y < height ? y * width + x : ~0u;
}
// The same, ~0u also outside the pixels rendered (the sub viewport: min x, min y, max x, max y, half open).
X2_HD inline U32 tilePixel(U32 tile, U32 position, U32 width, U32 height, const int* viewport) {
    const U32 p = tilePixel(tile, position, width, height);
    if (p == ~0u) return p;
    const int x = int(p % width), y = int(p / width);
    return x >= viewport[0] && y >= viewport[1] && x < viewport[2] && y < viewport[3] ? p : ~0u;
}

}  // namespace xpu2::interactive::async

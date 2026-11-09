// Private XPU2: participating media for the asynchronous path tracer, shared by the CPU and CUDA lanes. What MoonRay
// does along a ray through volumes bound to closed meshes (pbr/integrator PathIntegratorVolume.cc:
// collectVolumeIntervals, evalVolumeShaders, evalSigmaT, decoupledRayMarching's homogeneous case), its phase
// function (pbr/core VolumePhase.h), distance and equi-angular sampling (pbr/core Util.h, Distribution.cc's
// GuideDistribution1D) and BaseVolume's coefficients (moonray dso/volume/Base BaseVolume.cc) for volumes whose
// attributes are not bound (homogeneous: one value over each stretch of the ray, its ramps by the stretch's length).
// The integrator's use of them is in async_integrator.h. Derived from MoonRay Apache-2.0, Copyright 2023-2024
// DreamWorks Animation LLC. See MOONRAY_LICENSE.txt.
#pragma once
#include "light_core.h"
#include "shade_color.h"
#include "volume_grid.h"
#include "volume_emission.h"

namespace xpu2::interactive::volume {
using surface::V3;
using sample::U32;

constexpr U32 MaxVolumes = 32;    // a scene's volumes: one bit each in a ray's regions
constexpr U32 MaxCrossings = 32;  // crossings collected along one ray
// The steps a ray's march keeps (decoupledRayMarching's VolumeProperties): a homogeneous stretch is one, a
// heterogeneous one a step each feature size over the volume quality. The host keeps a march's steps in a buffer of
// its own (volumeSteps); the CUDA lane none (it marches again, ReplaySteps).
// A march's steps are fewer than this on both lanes (async_integrator.h's VolumeMarch): the CPU lane keeps them in a table
// (MarchSpace); the CUDA lane keeps none, marching again for a step it needs (ReplaySteps).
constexpr U32 MarchLimit = 1u << 14;
#if defined(__CUDACC__)
constexpr U32 MaxSteps = MaxCrossings + 2;
#else
constexpr U32 MaxSteps = MarchLimit;
#endif
// GPU_VDB: VDB volumes on the CUDA lane (gpuRenders): their grids uploaded whole (async_gpu.h's uploadGrids), sampled as
// the host samples them (volume_grid.h), marched with no table of steps (async_integrator.h's ReplaySteps). Off: their
// samples go to the CPU lane, as before.
constexpr bool GpuVdb = true;
// GPU_EMISSION: glowing volumes (BaseVolume emission, a VDB's emission grid) on the CUDA lane: their emissive regions
// uploaded with the grids (async_gpu.h's uploadRegions) and sampled as the host samples them (volume_emission.h). Off:
// a scene with one renders on the CPU lane alone, as before.
constexpr bool GpuEmission = true;
constexpr float Epsilon = 1.1920928955078125e-07f;  // scene_rdl2 sEpsilon
constexpr float HitEpsilonStart = 1e-5f;            // pbr sHitEpsilonStart
constexpr float FarLimit = 0.999f * lights::DistantDistance;  // sHitEpsilonEnd * sDistantLightDistance
enum OverlapMode : U32 { OverlapSum = 0, OverlapMax = 1, OverlapRnd = 2 };
// DepthFixed: a CutoutVolume's indirect volume, read with no ray volume depth (-1: no ramps).
enum ShaderFlags : U32 { UseAttenuationRamp = 1, UseAlbedoRamp = 2, MatchAlbedo = 4, InvertAttenuation = 8, DepthFixed = 16 };

// A volume: a BaseVolume with no bound attributes (homogeneous: its attenuation and albedo (diffuse) colours and
// their ramps over the depth a ray travels through it, its density ramp, the attenuation's intensity and factor, its
// emission), or a VdbVolume on a VdbGeometry (heterogeneous: its density grid as the geometry's interpolation samples
// it, its emission grid, times the shader's opacity gain, colour and incandescence gain; the geometry's world to grid
// world transform; the grid's feature size, cube root of a voxel's render-space volume). Both: the anisotropy, and
// the lights its assignment's light set lets light it.
// A BaseVolume with bound attributes (KindBaseMapped, heterogeneous): its extinction from a baked grid (densityGrid,
// through worldToGrid, its mesh's world to object) or, unbaked, from the attenuation colour (MatchAlbedo: the diffuse
// colour), intensity and factor its maps give at the render origin (MoonRay's empty state); the diffuse colour,
// emission colour and intensity and anisotropy from one map program (program, a register each bound one, mapped a bit
// each) at each sample point; feature size the baked grid's voxel, else one.
enum ShaderKind : U32 { KindBase = 0, KindVdb = 1, KindBaseMapped = 2 };
enum MappedAttribute : U32 {
    MapAttenuationColor = 0, MapAttenuationIntensity, MapAttenuationFactor, MapAlbedo, MapEmissionColor, MapEmissionIntensity,
    MapAnisotropy, MappedCount
};
struct Shader {
    V3 attenuationColor, albedoColor;
    float intensity, factor, g;
    U32 flags, lightSet;
    float attenuationMin, attenuationMax, albedoMin, albedoMax, densityMin, densityMax;
    shade::ColorRamp attenuationRamp, albedoRamp;
    shade::FloatRamp densityRamp;
    V3 emission;  // BaseVolume: emission_color times emission_intensity
    U32 kind;
    U32 densityGrid, emissionGrid, interpolation;
    float featureSize;
    float worldToGrid[12];  // rows of an affine transform, as the light filters' X34
    V3 gain, colorMult, incandescence;
    U32 visibility;  // its geometry's (MoonRay's visibility bits: 2 shadow, 0x100 and 0x200 phase)
    int lpeLabel;    // its shader's label in the light path expressions (-1 none)
    U32 program, mapped;
    U32 regs[MappedCount];
    V3 emissionColor;
    float emissionIntensity;
    V3 originAttenuation;  // unbaked: the attenuation colour (or the diffuse one, MatchAlbedo) at the render origin
    float originIntensity, originFactor;
    U32 cutout;  // a CutoutVolume: held out on camera rays (its indirect volume elsewhere)
    // MOONRAY_MODO_CRYPTO_VOLUMES: its geometry's Cryptomatte IDs (modoCryptoVolumesBuild), when it has the object's.
    float cryptoId, cryptoMaterialId, cryptoAssetId;
    U32 cryptoValid;
    // A VdbVolume: its baked opacity gain (bakedGrid, NoGrid without one); its velocity grid (VDBVelocity: the sample
    // advected back along it twice, by tShutterOpen + time * tShutterRange); a moving geometry's world to grid at the
    // shutter's close (lerped with worldToGrid by the ray's time); its grid to world at both ends (the maps' P).
    U32 bakedGrid, hasVelocity, velocityGrid, moving;
    float tShutterOpen, tShutterRange;
    float worldToGrid1[12], gridToWorld[12], gridToWorld1[12];
    float surfaceOpacityThreshold;  // VolumeShader's surface_opacity_threshold (the state AOVs' volume surface)
    // Its maps run on the host alone (gpuRenders): they read image textures (a tile not on the card cannot be waited
    // for in the middle of a march), or XPU2_GPU_VOLUME_MAPS_OFF. Otherwise the card runs them too (GpuVolumeMaps).
    U32 mapsOnHost;
};
// GPU_VOLUME_MAPS: mapped volumes on the CUDA lane: their map programs (procedural maps, constants, ramps; no image
// textures) run at each sample point by the map interpreter built into the volume module (async_gpu_device.h's deviceVolumeMaps), as the host
// runs them, so a glowing volume whose maps give its emission no longer leaves the whole scene to the CPU. Off: their
// samples go to the CPU lane, and a glowing one sends the scene there, as before.
constexpr bool GpuVolumeMaps = true;
// GPU_MARCH_CHECKPOINTS: the card's step lookups (async_integrator.h's ReplaySteps) resume the march from the first
// march's checkpoints instead of marching again from the ray's start: the same steps bit for bit, fewer of them.
X2_HD constexpr bool gpuMarchCheckpoints() { return true; }  // (a function: no module keeps it as a global)
// GPU_ONE_TRAVERSAL: a ray's volume crossings from one traversal of the volume bounds (every candidate kept), not a trace
// a crossing: the same crossings, bit for bit.
X2_HD constexpr bool gpuOneTraversal() { return true; }
// A mapped volume's bound values at a point (the unbound ones its constants): the map programs (cpu_scene.h's
// volumeMapped on the host, deviceVolumeMaps on the card), through Grids::maps.
struct Mapped {
    V3 attenuationColor, albedoColor, emissionColor;
    float attenuationIntensity, attenuationFactor, emissionIntensity, anisotropy;
};
// A map program's registers into a mapped volume's values (both lanes): each bound value times its map's colour
// (evalColor) or the colour's average (evalFloat), unless zero.
X2_HD inline void applyMaps(const Shader& s, const surface::V3* regs, Mapped& m) {
    auto color = [&](unsigned k, V3& v) {
        if (!(s.mapped >> k & 1) || (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f)) return;
        const V3 c = regs[s.regs[k]];
        v = V3{v.x * c.x, v.y * c.y, v.z * c.z};
    };
    auto scalar = [&](unsigned k, float& f) {
        if (!(s.mapped >> k & 1) || fabsf(f) <= 1e-6f) return;  // shade::isZero
        const V3 c = regs[s.regs[k]];
        f *= (c.x + c.y + c.z) / 3.0f;
    };
    color(MapAttenuationColor, m.attenuationColor);
    scalar(MapAttenuationIntensity, m.attenuationIntensity);
    scalar(MapAttenuationFactor, m.attenuationFactor);
    color(MapAlbedo, m.albedoColor);
    color(MapEmissionColor, m.emissionColor);
    scalar(MapEmissionIntensity, m.emissionIntensity);
    scalar(MapAnisotropy, m.anisotropy);
}
#if defined(__CUDACC__) && defined(XPU2_DEVICE_VOLUME_MAPS)
// The card's run of a mapped volume's maps at p (world), context its DeviceVolumeMaps (async_gpu_device.h, which sets
// XPU2_DEVICE_VOLUME_MAPS; a CUDA file without it, the adaptive kernels, has no volumes).
static __device__ __noinline__ void deviceVolumeMaps(const void* context, const Shader& s, V3 p, Mapped& m);
#endif
struct MapEval {
    void (*run)(const void* context, const Shader& s, V3 p, Mapped& values);
    const void* context;
};
// A ray's crossing of a volume's mesh: where, which volume, and whether it enters (1) or leaves (2) it.
struct Crossing {
    float t;
    U32 volume, flags;
};
enum CrossingFlags : U32 { Entry = 1, Exit = 2 };
// VolumeProperties of a stretch (homogeneous) or a step (heterogeneous): its coefficients and the transmittance from
// the ray's start to it.
struct Step {
    V3 sigmaT, sigmaS, tr;
    float g, tStart, delta;
    U32 lightSet;
    V3 sigmaTh, sigmaSh, trH;  // the holdouts' extinction and scattering, and their transmittance to the step
    U32 volume;                // the volume evalVolumeShaders chose (its volumeIdSampled)
    float surfaceThreshold;    // its surface opacity threshold (1 with no volume)
};
X2_HD inline bool homogeneous(const Shader& s) { return s.kind == KindBase; }
// Whether the GPU's programs render a volume (async_gpu.h, async_gpu_device.h): a homogeneous BaseVolume, a VDB volume
// whose grids are on the card, or a mapped one whose maps give its attenuation alone (baked into a dense grid the GPU
// holds, or taken at the render origin unbaked); glowing or not (GpuEmission: their emissive regions are on the card
// too). Mapped ones whose maps give their colour, emission or anisotropy run those maps on the card too (GpuVolumeMaps)
// unless the maps read image textures: those are the CPU lane's, their samples deferred to it.
// The maps the card would run at each point: those that give more than the attenuation (baked into a grid, or taken
// at the render origin, on the host).
X2_HD inline bool pointMaps(const Shader& s) {
    const U32 attenuation = (1u << MapAttenuationColor) | (1u << MapAttenuationIntensity) | (1u << MapAttenuationFactor);
    return (s.mapped & ~attenuation) != 0;
}
X2_HD inline bool gpuRenders(const Shader& s, const Grids& grids) {
    if (!GpuEmission && (s.emission.x != 0.0f || s.emission.y != 0.0f || s.emission.z != 0.0f)) return false;
    if (s.kind == KindBase) return true;
    // Point maps the card runs (GpuVolumeMaps), unless they read textures (mapsOnHost).
    const bool mapsHere = !pointMaps(s) || (GpuVolumeMaps && !s.mapsOnHost);
    if (s.kind == KindVdb) {
        // A VDB volume the GPU samples from its NanoVDB grids (volume_grid.h): one that is not mapped (the maps run on
        // the host), its grids on the card (null where uploadGrids has not sent them: then deferred, as before), its
        // emission grid too where it glows.
        if (!GpuVdb) return false;
        auto there = [&](U32 g) { return g == NoGrid || (g < grids.count && (grids.grids[g].data || grids.grids[g].dense)); };
        const bool maps = !s.mapped || (GpuVolumeMaps && mapsHere);
        return maps && (GpuEmission ? there(s.emissionGrid) : s.emissionGrid == NoGrid) && s.densityGrid != NoGrid &&
               there(s.densityGrid) && there(s.bakedGrid) && (!s.hasVelocity || there(s.velocityGrid));
    }
    if (s.kind != KindBaseMapped) return false;
    // Maps that give its colour, emission or anisotropy: the card runs them (GpuVolumeMaps) unless they read textures.
    if (!mapsHere) return false;
    if (!GpuEmission && s.emissionIntensity != 0.0f && (s.emissionColor.x != 0.0f || s.emissionColor.y != 0.0f || s.emissionColor.z != 0.0f))
        return false;
    return s.densityGrid == NoGrid || (s.densityGrid < grids.count && grids.grids[s.densityGrid].dense);
}
// A point in the volume's grid world (its geometry's own space).
X2_HD inline V3 gridPoint(const Shader& s, V3 p) {
    const float* m = s.worldToGrid;
    return V3{m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
              m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
}

X2_HD inline V3 affine34(const float* m, V3 p) {
    return V3{m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7], m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
}
// A VDB's grid point at the ray's time: a moving geometry's lerped (transformPoint of the lerped render to prim).
X2_HD inline V3 gridPointAt(const Shader& s, V3 p, float time) {
    const V3 a = affine34(s.worldToGrid, p);
    if (!s.moving) return a;
    const V3 b = affine34(s.worldToGrid1, p);
    return V3{a.x + (b.x - a.x) * time, a.y + (b.y - a.y) * time, a.z + (b.z - a.z) * time};
}
X2_HD inline V3 worldPointAt(const Shader& s, V3 g, float time) {
    const V3 a = affine34(s.gridToWorld, g);
    if (!s.moving) return a;
    const V3 b = affine34(s.gridToWorld1, g);
    return V3{a.x + (b.x - a.x) * time, a.y + (b.y - a.y) * time, a.z + (b.z - a.z) * time};
}
X2_HD inline V3 times(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
X2_HD inline V3 scaled(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
X2_HD inline V3 plus(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
X2_HD inline V3 cmax0(V3 a) { return {fmaxf(a.x, 0.0f), fmaxf(a.y, 0.0f), fmaxf(a.z, 0.0f)}; }
X2_HD inline V3 expNeg(V3 a, float s) { return {expf(-a.x * s), expf(-a.y * s), expf(-a.z * s)}; }
X2_HD inline float lum(V3 a) { return 0.299f * a.x + 0.587f * a.y + 0.114f * a.z; }
X2_HD inline bool black(V3 a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f; }
X2_HD inline bool isZero(float x) { return fabsf(x) < Epsilon; }

// BaseVolume's ramp helpers: the ramp at the depth's place between its two depths.
X2_HD inline float rampT(float minDepth, float maxDepth, float depth) {
    const float t = fmaxf(depth - minDepth, 0.0f), interp = maxDepth - minDepth;
    return interp == 0.0f ? 0.0f : t / interp;
}
X2_CALL V3 attenuationColorFrom(const Shader& s, V3 color, float depth) {
    const bool match = (s.flags & MatchAlbedo) != 0;
    const bool useRamp = (s.flags & (match ? UseAlbedoRamp : UseAttenuationRamp)) != 0;
    if (useRamp && depth >= 0.0f) {
        color = match ? shade::rampEval(s.albedoRamp, rampT(s.albedoMin, s.albedoMax, depth))
                      : shade::rampEval(s.attenuationRamp, rampT(s.attenuationMin, s.attenuationMax, depth));
    }
    return (s.flags & InvertAttenuation) ? V3{1.0f - color.x, 1.0f - color.y, 1.0f - color.z} : color;
}
X2_HD inline V3 attenuationColor(const Shader& s, float depth) {
    return attenuationColorFrom(s, (s.flags & MatchAlbedo) ? s.albedoColor : s.attenuationColor, depth);
}
X2_HD inline V3 albedoColor(const Shader& s, float depth) {
    if ((s.flags & UseAlbedoRamp) && depth >= 0.0f) return shade::rampEval(s.albedoRamp, rampT(s.albedoMin, s.albedoMax, depth));
    return s.albedoColor;
}
// getDensity of density one: its density ramp at the depth (none for -1).
X2_HD inline float density(const Shader& s, float depth) {
    if (depth == -1.0f) return 1.0f;
    return shade::rampEval(s.densityRamp, rampT(s.densityMin, s.densityMax, depth));
}
// extinct() and albedo() of density one.
X2_HD inline V3 extinction(const Shader& s, float depth) {
    if (s.flags & DepthFixed) depth = -1.0f;
    const float d = density(s, depth);
    return cmax0(scaled(times(scaled(V3{d, d, d}, s.intensity), attenuationColor(s, depth)), s.factor));
}
X2_CALL V3 albedo(const Shader& s, float depth) {
    if (s.flags & DepthFixed) depth = -1.0f;
    const float d = density(s, depth);
    return cmax0(times(V3{d, d, d}, albedoColor(s, depth)));
}

// A volume's extinction, albedo multiplier and emission at a point (evalVolumeCoefficients with the shader's extinct,
// albedo and emission): a BaseVolume's at the depth (homogeneous), a VdbVolume's density (by the interpolation given:
// the geometry's in a march, the nearest voxel for shadows, evalDensity) times its gain, its colour, and its
// emission grid's nearest voxel times its incandescence gain.
struct Local {
    V3 sigmaT, albedo, emission;
    float g;
};
X2_HD inline Local localAt(const Shader& s, const Grids& grids, V3 p, float depth, U32 interpolation) {
    if (s.kind == KindVdb) {
        V3 gp = gridPointAt(s, p, grids.time);
        if (s.hasVelocity) {
            // VDBVelocity::getEvalPosition: back along the velocity there, then along the velocity found there.
            const float dt = s.tShutterOpen + grids.time * s.tShutterRange;
            const V3 v0 = sampleVec3(grids, s.velocityGrid, gp, InterpolationBox);
            const V3 q{float(double(gp.x) - double(v0.x * dt)), float(double(gp.y) - double(v0.y * dt)), float(double(gp.z) - double(v0.z * dt))};
            const V3 v1 = sampleVec3(grids, s.velocityGrid, q, InterpolationBox);
            gp = V3{float(double(gp.x) - double(v1.x * dt)), float(double(gp.y) - double(v1.y * dt)), float(double(gp.z) - double(v1.z * dt))};
        }
        const float density = sampleFloat(grids, s.densityGrid, gp, interpolation);
        const V3 temperature = s.emissionGrid != NoGrid ? sampleVec3(grids, s.emissionGrid, gp, InterpolationPoint) : V3{0, 0, 0};
        Mapped m{s.gain, s.colorMult, s.incandescence, 1.0f, 1.0f, 1.0f, s.g};
        // The maps at the sample's render position (transformVolumeSamplePosition of the advected point).
#if defined(__CUDACC__) && defined(XPU2_DEVICE_VOLUME_MAPS)
        if (pointMaps(s) && grids.maps) deviceVolumeMaps(grids.maps, s, worldPointAt(s, gp, grids.time), m);
#elif !defined(__CUDACC__)
        if (s.mapped && grids.maps) {
            const MapEval* e = static_cast<const MapEval*>(grids.maps);
            e->run(e->context, s, worldPointAt(s, gp, grids.time), m);
        }
#endif
        // evalVolumeCoefficients: the density times sampleBakedDensity (the baked gain's nearest voxel, else the gain).
        const V3 gain = s.bakedGrid != NoGrid ? sampleVec3(grids, s.bakedGrid, gp, InterpolationPoint) : s.gain;
        return Local{cmax0(scaled(gain, density)), cmax0(m.albedoColor), cmax0(times(temperature, m.emissionColor)), fminf(fmaxf(m.anisotropy, -1.0f), 1.0f)};
    }
    if (s.kind == KindBaseMapped) {
        if (s.flags & DepthFixed) depth = -1.0f;
        Mapped m{s.attenuationColor, s.albedoColor, s.emissionColor, s.intensity, s.factor, s.emissionIntensity, s.g};
#if defined(__CUDACC__) && defined(XPU2_DEVICE_VOLUME_MAPS)
        if (pointMaps(s) && grids.maps) deviceVolumeMaps(grids.maps, s, p, m);
#elif !defined(__CUDACC__)
        if (s.mapped && grids.maps) {
            const MapEval* e = static_cast<const MapEval*>(grids.maps);
            e->run(e->context, s, p, m);
        }
#endif
        // evalDensity: the baked grid's nearest voxel; unbaked, extinct() of the empty state at the depth.
        const float d = density(s, depth);
        V3 sigmaT;
        if (s.densityGrid != NoGrid) sigmaT = sampleVec3(grids, s.densityGrid, gridPoint(s, p), InterpolationPoint);
        else sigmaT = cmax0(scaled(times(scaled(V3{d, d, d}, s.originIntensity), attenuationColorFrom(s, s.originAttenuation, depth)), s.originFactor));
        // albedo() of density one (getAlbedoColor: the map's, or the ramp's at the depth), emission() of density one.
        V3 color = m.albedoColor;
        if ((s.flags & UseAlbedoRamp) && depth >= 0.0f) color = shade::rampEval(s.albedoRamp, rampT(s.albedoMin, s.albedoMax, depth));
        const V3 emission = cmax0(scaled(m.emissionColor, m.emissionIntensity));
        return Local{sigmaT, cmax0(times(V3{d, d, d}, color)), emission, fminf(fmaxf(m.anisotropy, -1.0f), 1.0f)};
    }
    return Local{extinction(s, depth), albedo(s, depth), cmax0(s.emission), s.g};
}
// evalVolumeShaders over the volumes a stretch is in (a bit each), as the overlap mode combines them: extinction,
// scattering, the anisotropy (scattering-weighted) and the light set of the chosen volume (the thickest for SUM).
struct Coefficients {
    V3 sigmaT, sigmaS;
    float g;
    U32 lightSet;
    V3 sigmaTh, sigmaSh;  // camera rays' holdouts (cutouts on): a CutoutVolume's extinction and scattering
    float surfaceThreshold;  // the chosen volume's surface opacity threshold (1 with none)
};
// At a point of a heterogeneous stretch (pointEval): each volume's extinction and albedo there; its emission into
// emissions (SUM: each volume's; otherwise the chosen one's, weighted, in emissions[0]), and the chosen volume.
X2_CALL Coefficients evalShadersAt(const Shader* shaders, const Grids& grids, U32 regions, U32 mode, float rnd, V3 p, float depth,
                                        U32 interpolationOverride, V3* emissions, U32* chosenVolume, bool cutouts = false) {
    Coefficients c{{0, 0, 0}, {0, 0, 0}, 0.0f, 0, {0, 0, 0}, {0, 0, 0}, 1.0f};
    V3 anisotropy{0, 0, 0}, sigmaTSum{0, 0, 0};
    float maxSigmaTLum = -3.402823466e+38f;
    U32 first = MaxVolumes, chosen = MaxVolumes;
    Local locals[MaxVolumes];
    for (U32 v = 0; v < MaxVolumes; ++v) {
        if (!(regions >> v & 1)) continue;
        const Shader& s = shaders[v];
        locals[v] = localAt(s, grids, p, depth, interpolationOverride != NoGrid ? interpolationOverride : s.interpolation);
        const V3 sigmaTLocal = locals[v].sigmaT;
        const V3 scatter = times(locals[v].albedo, sigmaTLocal);
        if (first == MaxVolumes) first = v;
        const float l = lum(sigmaTLocal);
        const bool cut = cutouts && s.cutout;
        if (mode == OverlapSum) {
            if (!cut) {
                c.sigmaT = plus(c.sigmaT, sigmaTLocal);
                c.sigmaS = plus(c.sigmaS, scatter);
                anisotropy = plus(anisotropy, scaled(scatter, locals[v].g));
                if (emissions) emissions[v] = locals[v].emission;
            } else {
                c.sigmaTh = plus(c.sigmaTh, sigmaTLocal);
                c.sigmaSh = plus(c.sigmaSh, scatter);
                if (emissions) emissions[v] = V3{0, 0, 0};
            }
            if (l > maxSigmaTLum) {
                maxSigmaTLum = l;
                c.lightSet = s.lightSet;
                chosen = v;
            }
        } else if (mode == OverlapMax) {
            if (l > maxSigmaTLum) {
                maxSigmaTLum = l;
                c.sigmaT = cut ? V3{0, 0, 0} : sigmaTLocal;
                c.sigmaS = cut ? V3{0, 0, 0} : scatter;
                c.sigmaTh = cut ? sigmaTLocal : V3{0, 0, 0};
                c.sigmaSh = cut ? scatter : V3{0, 0, 0};
                anisotropy = cut ? V3{0, 0, 0} : scaled(scatter, locals[v].g);
                c.lightSet = s.lightSet;
                chosen = v;
                if (emissions) emissions[0] = cut ? V3{0, 0, 0} : locals[v].emission;
            }
        } else {
            sigmaTSum = plus(sigmaTSum, sigmaTLocal);
        }
    }
    if (mode == OverlapRnd && first != MaxVolumes) {
        const float total = lum(sigmaTSum);
        U32 pick = first;
        float weight = 1.0f;
        if (total != 0.0f) {
            V3 running{0, 0, 0};
            pick = MaxVolumes;
            U32 last = first;
            for (U32 v = 0; v < MaxVolumes; ++v) {
                if (!(regions >> v & 1)) continue;
                running = plus(running, locals[v].sigmaT);
                last = v;
                if (rnd < lum(running) / total) {
                    pick = v;
                    break;
                }
            }
            if (pick == MaxVolumes) pick = last;
            weight = total / lum(locals[pick].sigmaT);
        }
        const V3 scatter = times(locals[pick].albedo, locals[pick].sigmaT);
        const bool cut = cutouts && shaders[pick].cutout;
        c.sigmaT = cut ? V3{0, 0, 0} : scaled(locals[pick].sigmaT, weight);
        c.sigmaS = cut ? V3{0, 0, 0} : scaled(scatter, weight);
        c.sigmaTh = cut ? scaled(locals[pick].sigmaT, weight) : V3{0, 0, 0};
        c.sigmaSh = cut ? scaled(scatter, weight) : V3{0, 0, 0};
        anisotropy = cut ? V3{0, 0, 0} : scaled(scaled(scatter, locals[pick].g), weight);
        c.lightSet = shaders[pick].lightSet;
        chosen = pick;
        if (emissions) emissions[0] = cut ? V3{0, 0, 0} : scaled(locals[pick].emission, weight);
    }
    if (chosenVolume) *chosenVolume = chosen;
    if (chosen < MaxVolumes) c.surfaceThreshold = shaders[chosen].surfaceOpacityThreshold;
    c.g = lum(anisotropy);
    const float normalization = lum(c.sigmaS);
    if (normalization > 0.0f) c.g /= normalization;
    return c;
}
// evalSigmaT at a point (shadow rays): each volume's density by the nearest voxel (evalDensity), combined.
X2_HD inline V3 evalSigmaTAt(const Shader* shaders, const Grids& grids, U32 regions, U32 mode, float rnd, V3 p, float depth) {
    return evalShadersAt(shaders, grids, regions, mode, rnd, p, depth, InterpolationPoint, nullptr, nullptr).sigmaT;
}
// Whether every volume of a stretch is homogeneous, and the least feature size of those that are not.
X2_HD inline bool stretchHomogeneous(const Shader* shaders, U32 regions, float& minFeatureSize) {
    bool all = true;
    minFeatureSize = 3.402823466e+38f;
    for (U32 v = 0; v < MaxVolumes; ++v) {
        if (!(regions >> v & 1)) continue;
        if (!homogeneous(shaders[v])) {
            all = false;
            minFeatureSize = fminf(minFeatureSize, shaders[v].featureSize);
        }
    }
    return all;
}
X2_HD inline Coefficients evalShaders(const Shader* shaders, U32 regions, U32 mode, float rnd, float depth) {
    Coefficients c{{0, 0, 0}, {0, 0, 0}, 0.0f, 0, {0, 0, 0}, {0, 0, 0}, 1.0f};
    V3 anisotropy{0, 0, 0}, sigmaTSum{0, 0, 0};
    float maxSigmaTLum = -3.402823466e+38f;
    U32 first = MaxVolumes;
    // RND: the volumes' running sums of extinction (luminance), then the one rnd picks.
    for (U32 v = 0; v < MaxVolumes; ++v) {
        if (!(regions >> v & 1)) continue;
        const Shader& s = shaders[v];
        const V3 sigmaTLocal = extinction(s, depth);
        const V3 scatter = times(albedo(s, depth), sigmaTLocal);
        if (first == MaxVolumes) first = v;
        const float l = lum(sigmaTLocal);
        if (mode == OverlapSum) {
            c.sigmaT = plus(c.sigmaT, sigmaTLocal);
            c.sigmaS = plus(c.sigmaS, scatter);
            anisotropy = plus(anisotropy, scaled(scatter, s.g));
            if (l > maxSigmaTLum) {
                maxSigmaTLum = l;
                c.lightSet = s.lightSet;
            }
        } else if (mode == OverlapMax) {
            if (l > maxSigmaTLum) {
                maxSigmaTLum = l;
                c.sigmaT = sigmaTLocal;
                c.sigmaS = scatter;
                anisotropy = scaled(scatter, s.g);
                c.lightSet = s.lightSet;
            }
        } else {
            sigmaTSum = plus(sigmaTSum, sigmaTLocal);
        }
    }
    if (mode == OverlapRnd && first != MaxVolumes) {
        const float total = lum(sigmaTSum);
        U32 chosen = first;
        float weight = 1.0f;
        if (total != 0.0f) {
            V3 running{0, 0, 0};
            chosen = MaxVolumes;
            U32 last = first;
            for (U32 v = 0; v < MaxVolumes; ++v) {
                if (!(regions >> v & 1)) continue;
                running = plus(running, extinction(shaders[v], depth));
                last = v;
                if (rnd < lum(running) / total) {
                    chosen = v;
                    break;
                }
            }
            if (chosen == MaxVolumes) chosen = last;  // past the last bound (rounding)
            weight = total / lum(extinction(shaders[chosen], depth));
        }
        const Shader& s = shaders[chosen];
        const V3 sigmaTLocal = extinction(s, depth);
        const V3 scatter = times(albedo(s, depth), sigmaTLocal);
        c.sigmaT = scaled(sigmaTLocal, weight);
        c.sigmaS = scaled(scatter, weight);
        anisotropy = scaled(scaled(scatter, s.g), weight);
        c.lightSet = s.lightSet;
    }
    c.g = lum(anisotropy);
    const float normalization = lum(c.sigmaS);
    if (normalization > 0.0f) c.g /= normalization;
    return c;
}
// evalSigmaT: the extinction alone (transmittance).
X2_HD inline V3 evalSigmaT(const Shader* shaders, U32 regions, U32 mode, float rnd, float depth) {
    return evalShaders(shaders, regions, mode, rnd, depth).sigmaT;
}

// collectVolumeIntervals' interiority: from the far end of the ray (outside every volume) back to its start, each
// volume's crossings counted so only its outermost entry and exit stay; the regions the start is in are returned.
X2_HD inline U32 interiority(Crossing* x, U32& n) {
    int count[MaxVolumes];
    for (U32 v = 0; v < MaxVolumes; ++v) count[v] = 0;
    U32 regions = 0;
    bool redundant = false;
    for (int i = int(n) - 1; i >= 0; --i) {
        const U32 v = x[i].volume;
        if (x[i].flags & Entry) {
            --count[v];
            if (count[v] == 0) regions &= ~(1u << v);
            else if (count[v] > 0) {
                x[i].flags &= ~Entry;
                redundant = true;
            }
        }
        if (x[i].flags & Exit) {
            if (count[v] == 0) regions |= 1u << v;
            else if (count[v] > 0) {
                x[i].flags &= ~Exit;
                redundant = true;
            }
            ++count[v];
        }
    }
    if (redundant) {
        U32 m = 0;
        for (U32 i = 0; i < n; ++i)
            if (x[i].flags) x[m++] = x[i];
        n = m;
    }
    return regions;
}
X2_HD inline void apply(U32& regions, const Crossing& x) {
    if (x.flags & Entry) regions |= 1u << x.volume;
    if (x.flags & Exit) regions &= ~(1u << x.volume);
}

// VolumePhase: Henyey-Greenstein (isotropic below |g| 1e-3).
X2_HD inline float phaseEval(float g, V3 wo, V3 wi) {
    if (fabsf(g) <= 1e-3f) return lights::OneOverFourPi;
    const float cosTheta = surface::dot(wo, wi), denom = 1 + g * g + 2 * g * cosTheta;
    return isZero(denom) ? 0.0f : lights::OneOverFourPi * (1 - g * g) / (denom * sqrtf(denom));
}
X2_HD inline V3 phaseSample(float g, V3 wo, float u1, float u2) {
    if (fabsf(g) <= 1e-3f) return shade::sampleSphereUniform(u1, u2);
    u1 = fmaxf(u1, Epsilon);  // clamp(u1, sEpsilon)
    const float sqrTerm = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
    const float cosTheta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const V3 local = shade::localSpherical(cosTheta, sinTheta, lights::TwoPi * u2);
    return shade::localToGlobal(shade::frame(V3{-wo.x, -wo.y, -wo.z}), local);
}

// Distance sampling proportional to exp(-sigma t) over [a, b], and equi-angular sampling about a pivot.
X2_HD inline float sampleDistanceExponential(float u, float sigma, float a, float b) {
    return a - logf(1.0f - u * (1.0f - expf(sigma * (a - b)))) / sigma;
}
X2_HD inline float pdfDistanceExponential(float t, float sigma, float a, float b) {
    return sigma / (expf((t - a) * sigma) - expf((t - b) * sigma));
}
X2_HD inline float sampleEquiAngular(float u, float D, float thetaA, float thetaB) { return D * tanf((1 - u) * thetaA + u * thetaB); }
X2_HD inline float pdfEquiAngular(float t, float D, float thetaA, float thetaB) { return D / ((thetaB - thetaA) * (D * D + t * t)); }
X2_HD inline bool finitef(float x) { return x - x == 0.0f; }

// GuideDistribution1D over the steps (discrete): weights to a cdf (all zero: uniform), and sampleDiscrete.
struct Distribution {
    float cdf[MaxSteps];
    float total;
    U32 n;
};
// A march's workspace: its steps, their weights and their distribution (each thread's own on the host, where a march
// may take thousands of steps).
struct MarchSpace {
    Step steps[MaxSteps];
    float density[MaxSteps];
    Distribution dist;
};
#if !defined(__CUDACC__)
inline MarchSpace& marchSpace() {
    static thread_local MarchSpace space;
    return space;
}
#endif
X2_HD inline void tabulate(Distribution& d, const float* w, U32 n) {
    d.n = n;
    float sum = 0.0f;
    for (U32 i = 0; i < n; ++i) sum += w[i];
    if (sum != 0.0f) {
        float partial = 0.0f;
        for (U32 i = 0; i + 1 < n; ++i) {
            partial += w[i];
            if (partial > sum) partial = sum;
            d.cdf[i] = partial;
        }
    } else {
        for (U32 i = 0; i + 1 < n; ++i) d.cdf[i] = float(i + 1) / float(n);
        sum = 1.0f;
    }
    d.cdf[n - 1] = sum;
    d.total = sum;
}
X2_HD inline U32 sampleDiscrete(const Distribution& d, float r, float& pdf, float& remapped) {
    const float rw = r * d.total;
    U32 i = 0;
    while (i + 1 < d.n && d.cdf[i] < rw) ++i;
    const float prev = i > 0 ? d.cdf[i - 1] : 0.0f, value = d.cdf[i] - prev;
    pdf = value / d.total;
    remapped = (rw - prev) / value;
    return i;
}
X2_HD inline float pdfDiscrete(const Distribution& d, U32 i) {
    const float prev = i > 0 ? d.cdf[i - 1] : 0.0f;
    return (d.cdf[i] - prev) / d.total;
}

// A light's equi-angular pivot (getEquiAngularPivot): a point on a rect or a disk (from its texture's distribution
// when it has one), a spot's lens, a cylinder, a sphere's centre, a mesh's tree root's centre (local, as MoonRay's).
X2_HD inline bool bounded(const lights::Light& l) { return l.type != lights::Env && l.type != lights::Distant; }
X2_HD inline V3 pivot(const lights::Light& l, const float r[3]) {
    V3 local{0, 0, 0};
    if (l.type == lights::Mesh) return l.mesh->nodes[0].center;
    if (l.type == lights::Cylinder) {
        float u = r[0], v = r[1];
        if (l.image) lights::imageSample(*l.image, r[0], r[1], 0.0f, l.filter, u, v);
        return surface::add(l.position, lights::rows(l.local2Render, lights::cylinderUv2Local(l, u, v)));
    }
    if (l.type == lights::Portal && l.image) {
        float u, v;
        lights::imageSample(*l.image, r[0], r[1], 0.0f, l.filter, u, v);
        local = V3{(u - l.uvOffset[0]) / l.uvScale[0], (v - l.uvOffset[1]) / l.uvScale[1], 0.0f};
        return surface::add(l.position, lights::rows(l.local2Render, local));
    }
    if ((l.type == lights::Rect || l.type == lights::Disk) && l.image) {
        float u, v;
        lights::imageSample(*l.image, r[0], r[1], 0.0f, l.filter, u, v);
        if (l.type == lights::Disk && (u * 2.0f - 1.0f) * (u * 2.0f - 1.0f) + (v * 2.0f - 1.0f) * (v * 2.0f - 1.0f) >= 1.0f) return l.position;
        local = V3{(u - l.uvOffset[0]) / l.uvScale[0], (v - l.uvOffset[1]) / l.uvScale[1], 0.0f};
        return surface::add(l.position, lights::rows(l.local2Render, local));
    }
    if (l.type == lights::Rect || l.type == lights::Portal) {
        local = V3{(0.5f - r[0]) * 2.0f * l.halfWidth, (0.5f - r[1]) * 2.0f * l.halfHeight, 0.0f};
    } else if (l.type == lights::Disk || l.type == lights::Spot) {
        const float rad = sqrtf(r[0]), angle = lights::TwoPi * r[1];
        local = V3{rad * cosf(angle), rad * sinf(angle), 0.0f};
        if (l.type == lights::Spot) local = V3{local.x * l.lensRadius, local.y * l.lensRadiusY, 0.0f};
    } else {
        return l.position;
    }
    return surface::add(l.position, lights::rows(l.local2Render, local));
}
}  // namespace xpu2::interactive::volume

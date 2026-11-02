// What every module of the GPU program shares (async_gpu.cu, async_material.cu): the launch parameters, named as
// OptiX links them, and the textures' device side (a miss asks for the tile).
#pragma once
#include <optix.h>
#include <optix_device.h>
#include "async_gpu_abi.h"
#include "points_core.h"
using namespace xpu2::interactive;
using namespace xpu2::interactive::async;

extern "C" {
__constant__ LaunchParams params{};
}

// Records each missing fine-detail tile once (the first sample to miss it asks
// for it), and marks tracked tiles that are read. With partial shading the lookup
// reads a coarser level instead, and the launch thread's mark keeps the tile.
struct RequestSink {
    U32* mark = nullptr;  // partial shading: the launch thread's mark (null: off)
    mutable unsigned missed = 0;
    __device__ void operator()(unsigned tile) const {
        missed = tile;
        if (!tile) return;
        if (atomicCAS(&params.tileState[tile], 0u, 1u) == 0u) {
            const U64L at = atomicAdd(&params.state->requestTail, 1ull);
            params.requests[at % params.requestCapacity] = tile;
        }
    }
    __device__ void touch(unsigned tile) const {
        if (!params.touched[tile]) params.touched[tile] = 1;
    }
    __device__ bool coarse() const { return mark != nullptr; }
    // A texel's parent a level up (tex::parentTexel), out of line (each of the filter's fetches calls it on a miss).
    // Read, it marks the launch thread's sample provisional with the tile it missed.
    __device__ __noinline__ unsigned parent(const tex::Image& image, unsigned level, unsigned x, unsigned y, surface::V3& out) const {
        const unsigned status = tex::parentTexel(image, level, x, y, out);
        if (status == tex::Ready) *mark = missed ? missed : ~0u;
        return status;
    }
};

// The scene's textures for the maps (shade_mapnodes.h): a miss asks for the tile.
struct DeviceTextures {
    const tex::Image* const* images;
    U32 count;
    shade::Color* registers;  // the launch thread's map registers (X2_MAP_REGISTERS)
    U32 registerCount;
    U32* mark;  // partial shading: the launch thread's mark (null: off), taken where the thread is known
    __device__ __noinline__ U32 lookup(U32 texture, tex::Binding b, const tex::Footprint& f, surface::V3& out) const {
        if (texture >= count || !images[texture]) return tex::Invalid;
        b.image = images[texture];
        b.swapCross = b.mode != 3 ? 1u : 0u;  // as XPU's vector shading looks up (displacement: in order)
        RequestSink sink{mark};
        return tex::lookup(b, f, out, sink);
    }
};

// The material callables' places in the binding table (async_material.cu, async_gpu.h Program::create).
enum MaterialCall : unsigned {
    CallShade = 0,
    CallPresence = 1,
    CallMaps = shade::MapsCallable,
    CallLightSample = lights::LightSampleCallable,
    CallLightIntersect = lights::LightIntersectCallable,
    CallLightEval = lights::LightEvalCallable,
    CallLobeEval = shade::LobeEvalCallable,
    CallLobeSample = shade::LobeSampleCallable,
    CallLobeAlbedo = shade::LobeAlbedoCallable,
    // The volume module's continuation callables (async_volumes.cu).
    CallVolumeRadiance = 9,
    CallSubsurfaceRadiance = 10,
    CallVolumeTransmittance = 11,
    // The DwaBase module's (async_dwa.cu), called by the material module.
    CallDwaResolve = shade::DwaResolveCallable,
    CallDwaLobes = shade::DwaLobesCallable,
    // The extras module's (async_extras.cu), called by the path tracer.
    CallCamera = surface::CameraCallable,
    CallAovLight = aov::AovLightCallable,
    CallAovVisibility = aov::AovVisibilityCallable,
    CallMaterialAovs = aov::MaterialAovsCallable
};

// The path tracer's backend on the GPU: its rays, hits, materials, lights and volumes (async_integrator.h's Backend),
// for every module that runs the integrators.
static __device__ float3 f3(surface::V3 v) { return make_float3(v.x, v.y, v.z); }

// A curve hit worked out again from its span (async_cpu.h's curveHit, the same arithmetic).
static __device__ bool curveHit(const curves::Set& c, const surface::Ray& ray, resume::Hit& out) {
    curves::Hit h;
    if (!curves::intersectSpan(c, out.prim, ray.origin, ray.direction, ray.nearT, ray.farT, h)) return false;
    out.t = h.t;
    out.u = h.u;
    out.v = h.v;
    out.point = shade::V3{ray.origin.x + ray.direction.x * h.t, ray.origin.y + ray.direction.y * h.t, ray.origin.z + ray.direction.z * h.t};
    out.origin = out.below = out.point;
    const bool roundCubic = c.round && c.basis == curves::BasisBspline;
    out.normal = roundCubic ? h.Ng : shade::normalize(-ray.direction);
    const U32 first = curves::spanVertex(c, out.prim);
    float radius;
    if (c.basis == curves::BasisLinear) {
        radius = curves::point(c, first).w * (1.0f - h.u) + curves::point(c, first + 1).w * h.u;
    } else {
        float w[4];
        curves::cubicWeights(c.basis, h.u, w);
        radius = w[0] * curves::point(c, first).w + w[1] * curves::point(c, first + 1).w + w[2] * curves::point(c, first + 2).w +
                 w[3] * curves::point(c, first + 3).w;
    }
    out.epsilon = 1.5f * radius;
    return shade::finite3(out.point) && out.epsilon > 0.0f;
}

struct GpuBackend {
    // The sample's time in the shutter (async_integrator.h sets it): every ray is traced at it.
    float time = 0.0f;
    // The GPU profile (XPU2_PROFILE): this thread's clock ticks in each stage, added to the launch's as it leaves.
    unsigned long long prof[ProfileStages] = {};
    __device__ long long tick() const { return params.profile ? clock64() : 0; }
    __device__ void tock(U32 stage, long long from) {
        if (params.profile) prof[stage] += (unsigned long long)(clock64() - from);
    }
    __device__ void countRay(U32 kind) {
        if (params.profile) ++prof[ProfileCountNee + kind];
    }
    // Each sample starts here (async_integrator.h): its time, and no texture miss or deferral yet.
    __device__ void setTime(float t) {
        time = t;
        missed = false;
        deferFlag = false;
        if (params.movedLightSlots) {  // no light in motion worked out yet at this time
            U32* tags = movedLightTags();
            for (U32 k = 0; k < params.movedLightSlots; ++k) tags[k] = ~0u;
        }
    }
    __device__ lights::Light* movedLights() const {
        return reinterpret_cast<lights::Light*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride + params.movedLightsOffset);
    }
    __device__ U32* movedLightTags() const { return reinterpret_cast<U32*>(movedLights() + params.movedLightSlots); }
    // A light in motion at the sample's time, worked out when first reached, in slots used in turn: the path tracer holds
    // one light at a time, so a slot is reused only after the light in it has been let go.
    __device__ const lights::Light& movedLight(U32 i) const {
        lights::Light* records = movedLights();
        U32* tags = movedLightTags();
        for (U32 k = 0; k < params.movedLightSlots; ++k)
            if (tags[k] == i) return records[k];
        U32& next = tags[params.movedLightSlots];
        const U32 at = next < params.movedLightSlots ? next : 0u;
        next = at + 1u;
        tags[at] = i;
        records[at] = lights::lightAtTime(params.lights[i], params.lightMotion[params.lightMoved[i]], time);
        return records[at];
    }
    __device__ surface::V3 vertexAt(const MeshMotion& motion, U32 i) const { return motionVertex(motion, i, time); }
    // Mesh geom's points (radii and centres), or null when it holds none.
    __device__ const float* pointRadii(U32 geom) const { return params.meshMotion ? params.meshMotion[geom].pointRadius : nullptr; }
    // A point's hit worked out again from its sphere (CpuBackend::trace, the same arithmetic).
    __device__ bool pointHit(const surface::Ray& ray, resume::Hit& out) const {
        const MeshMotion& motion = params.meshMotion[out.geom];
        const surface::V3 centre = motion.steps > 1 ? vertexAt(motion, out.prim) : params.geometry[out.geom].vertices[out.prim];
        float t;
        surface::V3 Ng;
        if (!points::intersect(centre, motion.pointRadius[out.prim], ray.origin, ray.direction, ray.nearT, ray.farT, t, Ng)) return false;
        out.t = t;
        out.u = out.v = 0.0f;
        out.point = surface::add(ray.origin, surface::mul(ray.direction, t));
        surface::V3 n;
        if (!surface::normalize(Ng, n)) return false;
        out.normal = surface::dot(n, ray.direction) > 0.0f ? surface::mul(n, -1.0f) : n;
        const float eps = 1e-5f * fmaxf(1.0f, fmaxf(fabsf(out.point.x), fmaxf(fabsf(out.point.y), fabsf(out.point.z))));
        out.origin = surface::add(out.point, surface::mul(out.normal, eps));
        out.below = surface::add(out.point, surface::mul(out.normal, -eps));
        return true;
    }
    // A mesh's shading attributes at the sample's time: its own while it is still, else a copy in local holding the time
    // steps about that time (CpuBackend::motion, curveAttributes).
    __device__ const shade::MeshAttributes& attributesAt(U32 geom, shade::MeshAttributes& local) const {
        const shade::MeshAttributes& a = params.meshAttributes[geom];
        if (!params.meshMotion || params.meshMotion[geom].steps < 2) return a;
        const MeshMotion& m = params.meshMotion[geom];
        local = a;
        if (m.kind == MotionCurves) {
            local.curve.time = time;
            local.time = time;
            return local;
        }
        if (m.kind == MotionPolygon) {
            local.positions0 = m.positions[0];
            local.positions1 = m.positions[1];
            local.normals1 = m.normals1;
            local.time = time;
            return local;
        }
        const float f = time * float(m.steps - 1);
        const U32 k = umin(U32(fmaxf(f, 0.0f)), m.steps - 2);
        const float between = f - float(k);
        if (m.normals) {
            local.normals = m.normals[k];
            local.normals1 = m.normals[k + 1];
        }
        if (m.dpds) {
            local.dpds = m.dpds[k];
            local.dpds1 = m.dpds[k + 1];
            local.dpdt = m.dpdt[k];
            local.dpdt1 = m.dpdt[k + 1];
            local.sampleTime = between;
        }
        if (m.kind == MotionSubd) {
            // OpenSubdivMesh::postIntersect: the motion vector and polyvertices from the first two steps.
            local.positions0 = m.positions[0];
            local.positions1 = m.positions[1];
            local.time = time;
            return local;
        }
        local.positions0 = m.positions[k];
        local.positions1 = m.positions[k + 1];
        local.time = between;
        local.motionScale = float(m.steps - 1);
        return local;
    }
    // Partial shading: the launch thread's mark that its sample read a coarser texel (null: off).
    __device__ U32* threadMark() const { return params.partial ? params.lowFlags + optixGetLaunchIndex().x : nullptr; }
    // The sample met a stand-in or a volume the GPU does not render: the CPU lane renders it instead.
    bool deferFlag = false;
    __device__ bool deferred() const { return deferFlag; }
    __device__ bool standInBetween(surface::V3 o, surface::V3 d, float near, float far) {
        if (!(near < far)) return false;
        unsigned instance = ~0u, primitive = 0;
        optixTrace(params.standInTraversable, f3(o), f3(d), near, far, time, 0xffu,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 1, 0, instance, primitive);
        return instance != ~0u;
    }
    // Shadow linking is not on the GPU yet (async_gpu.h leaves a scene with it to the CPU).
    U32 shadowReceiver = ~0u;
    __device__ void shadowFor(U32, U32 receiver) { shadowReceiver = receiver; }
    __device__ bool castsShadow(U32, bool) const { return true; }
    // Lobe light sets are not on the GPU yet (async_gpu.h leaves a scene with them to the CPU).
    U32 exitLightSet = ~0u;
    __device__ bool inLobeSet(U32, U32) const { return true; }
    // Light filters are not on the GPU yet (async_gpu.h leaves a scene with them to the CPU).
    __device__ const lights::Filter* filters() const { return nullptr; }
    __device__ surface::V3 filterMap(U32, U32, float, float) const { return surface::V3{1, 1, 1}; }
    __device__ float filterVdb(U32, U32, surface::V3) const { return 1.0f; }
    __device__ const lights::FilterList* filterList(U32, U32) const { return nullptr; }
    __device__ __noinline__ resume::Hit trace(const surface::Ray& ray) {
        const long long from = tick();
        if (params.profile) ++prof[ProfileCountTrace];
        const resume::Hit h = traceSurfaces(ray);
        // A stand-in before the hit (or anywhere along a ray that hits nothing) defers the sample.
        if (params.standInTraversable &&
            standInBetween(ray.origin, ray.direction, ray.nearT, h.geom != ~0u && h.t > ray.nearT ? h.t : ray.farT))
            deferFlag = true;
        tock(ProfileTrace, from);
        return h;
    }
    // Mesh geom's triangle's vertices where they are: a moving mesh's at the sample's time, a shared instance's placed by
    // its transform (shade::placed).
    __device__ void triangleAt(U32 geom, const mixed::Geometry& m, const surface::Triangle& t, surface::V3& a, surface::V3& b,
                               surface::V3& c) const {
        if (params.meshMotion && params.meshMotion[geom].steps > 1) {
            const MeshMotion& motion = params.meshMotion[geom];
            a = vertexAt(motion, t.a);
            b = vertexAt(motion, t.b);
            c = vertexAt(motion, t.c);
        } else {
            a = m.vertices[t.a];
            b = m.vertices[t.b];
            c = m.vertices[t.c];
        }
        if (params.meshAttributes && params.meshAttributes[geom].instance) {
            const shade::MeshAttributes& x = params.meshAttributes[geom];
            a = shade::placedPoint(x, a);
            b = shade::placedPoint(x, b);
            c = shade::placedPoint(x, c);
        }
    }
    __device__ resume::Hit traceSurfaces(const surface::Ray& ray) {
        unsigned instance = ~0u, primitive = 0;
        optixTrace(params.traversable, f3(ray.origin), f3(ray.direction), ray.nearT, ray.farT, time, ray.mask & 0xffu,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, instance, primitive);
        resume::Hit h;
        if (instance == ~0u || instance >= params.config->meshes) return h;
        h.geom = instance;
        h.prim = primitive;
        if (params.meshAttributes && params.meshAttributes[instance].curve.basis) {
            curves::Set c = params.meshAttributes[instance].curve;
            c.time = time;  // read only by curves in motion
            if (!curveHit(c, ray, h)) return resume::Hit{};
            return h;
        }
        if (pointRadii(instance)) return pointHit(ray, h) ? h : resume::Hit{};
        const auto m = params.geometry[instance];
        if (primitive >= m.triangleCount) {
            h.normal = {};
            return h;
        }
        const auto t = m.triangles[primitive];
        surface::V3 a, b, c;
        triangleAt(instance, m, t, a, b, c);
        if (!resume::geometryHit(h, ray, a, b, c)) h.normal = {};
        return h;
    }
    __device__ unsigned material(const resume::Hit& h, const surface::Ray& ray, surface::Lambert& out) {
        const auto& c = *params.config;
        if (h.geom >= c.meshes) return tex::Invalid;
        out = params.materials[h.geom];
        if (!c.textures) return tex::Ready;
        if (!params.maps) return tex::Invalid;
        const auto m = params.geometry[h.geom];
        const auto t = m.triangles[h.prim];
        tex::Footprint f;
        surface::V3 a, b, cc;
        triangleAt(h.geom, m, t, a, b, cc);
        if (!tex::footprint(a, b, cc, ray, h.u, h.v, c.spp, f, c.spatial != 0))
            return tex::Invalid;
        const auto maps = params.maps[h.geom];
        surface::V3* bases[2] = {&out.albedo, &out.emission};
        const tex::Binding bindings[2] = {maps.diffuse, maps.emission};
        RequestSink sink{threadMark()};
        for (int i = 0; i < 2; ++i) {
            const auto& b = bindings[i];
            auto& v = *bases[i];
            if (!b.image || (v.x == 0 && v.y == 0 && v.z == 0)) continue;
            surface::V3 color;
            const auto status = tex::lookup(b, f, color, sink);
            if (status != tex::Ready) return status;
            v = {v.x * color.x, v.y * color.y, v.z * color.z};
        }
        return tex::Ready;
    }
    __device__ U32 materialId(U32 geom) const { return params.meshMaterials[geom].offset; }
    __device__ resume::Hit traceMaterial(surface::Ray ray, U32 material) {
        for (int k = 0; k < 64; ++k) {
            const resume::Hit h = trace(ray);
            if (h.geom == ~0u || materialId(h.geom) == material) return h;
            ray.nearT = h.t + fmaxf(1e-6f, h.t * 1e-6f);
            if (!(ray.nearT < ray.farT)) break;
        }
        return resume::Hit{};
    }
    __device__ bool surfaceState(const resume::Hit& h, const surface::Ray& ray, shade::State& st) {
        const auto& c = *params.config;
        if (h.geom >= c.meshes) return false;
        shade::MeshAttributes local;
        const shade::MeshAttributes& attributes = attributesAt(h.geom, local);
        if (attributes.curve.basis) {
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return false;
            st.geom = h.geom;
            shade::curveDifferentials(attributes, ray, st.pixelSamples ? st.pixelSamples : c.spp, st);
            return true;
        }
        if (pointRadii(h.geom)) {
            if (!points::state(attributes, h.prim, h.normal, h.point, ray.direction, st)) return false;
            st.geom = h.geom;
            return true;
        }
        const auto m = params.geometry[h.geom];
        const auto t = m.triangles[h.prim];
        surface::V3 a, b, cc;
        triangleAt(h.geom, m, t, a, b, cc);
        if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, a, b, cc, h.u, h.v, h.point, ray.direction, st))
            return false;
        st.geom = h.geom;
        shade::hitDifferentials(attributes, h.prim, t.a, t.b, t.c, a, b, cc, ray, st.pixelSamples ? st.pixelSamples : c.spp,
                                st);
        return true;
    }
    // A shadow ray's hit: its material's presence alone (presence shadows); a texture not yet resident marks the
    // sample to be done again (textureMissed).
    bool missed = false;
    __device__ bool textureMissed() const { return missed; }
    __device__ unsigned presenceAt(const resume::Hit& h, const surface::Ray& ray, float& presence) {
        const auto& c = *params.config;
        presence = 1.0f;
        if (h.geom >= c.meshes) return tex::Ready;
        shade::State st;
        shade::MeshAttributes local;
        const shade::MeshAttributes& attributes = attributesAt(h.geom, local);
        if (attributes.curve.basis) {
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return tex::Ready;
        } else if (pointRadii(h.geom)) {
            if (!points::state(attributes, h.prim, h.normal, h.point, ray.direction, st)) return tex::Ready;
        } else {
            const auto m = params.geometry[h.geom];
            const auto t = m.triangles[h.prim];
            surface::V3 a, b, cc;
            triangleAt(h.geom, m, t, a, b, cc);
            if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, a, b, cc, h.u, h.v, h.point, ray.direction, st))
                return tex::Ready;
        }
        st.geom = h.geom;
        st.worldToRender = c.worldToRender;
        st.worldToObject = params.meshAttributes[h.geom].worldToObject;
        const DeviceTextures textures{params.textures, params.textureCount, threadRegisters(), params.mapRegisters, threadMark()};
        const unsigned status = materialPresence(params.meshMaterials[h.geom], st, textures, presence);
        if (status == tex::Miss) missed = true;
        return status;
    }
    // The path tracer's hit: its shading state, and its material's Bsdf (shade_material.h).
    __device__ unsigned shade(const resume::Hit& h, const surface::Ray& ray, shade::State& st, shade::Bsdf& bsdf, float& presence) {
        const auto& c = *params.config;
        if (h.geom >= c.meshes) return tex::Invalid;
        shade::MeshAttributes local;
        const shade::MeshAttributes& attributes = attributesAt(h.geom, local);
        if (attributes.curve.basis) {
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return tex::Invalid;
            st.geom = h.geom;
            st.worldToRender = c.worldToRender;
            st.worldToObject = params.meshAttributes[h.geom].worldToObject;
            shade::curveDifferentials(attributes, ray, st.pixelSamples ? st.pixelSamples : c.spp, st);
            const DeviceTextures textures{params.textures, params.textureCount, threadRegisters(), params.mapRegisters, threadMark()};
            return shadeMaterial(params.meshMaterials[h.geom], st, textures, bsdf, presence);
        }
        if (pointRadii(h.geom)) {
            if (!points::state(attributes, h.prim, h.normal, h.point, ray.direction, st)) return tex::Invalid;
            st.geom = h.geom;
            st.worldToRender = c.worldToRender;
            st.worldToObject = params.meshAttributes[h.geom].worldToObject;
            const DeviceTextures textures{params.textures, params.textureCount, threadRegisters(), params.mapRegisters, threadMark()};
            return shadeMaterial(params.meshMaterials[h.geom], st, textures, bsdf, presence);
        }
        const auto m = params.geometry[h.geom];
        const auto t = m.triangles[h.prim];
        surface::V3 a, b, cc;
        triangleAt(h.geom, m, t, a, b, cc);
        if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, a, b, cc, h.u, h.v, h.point, ray.direction, st))
            return tex::Invalid;
        st.geom = h.geom;
        st.worldToRender = c.worldToRender;
        st.worldToObject = params.meshAttributes[h.geom].worldToObject;
        const shade::MaterialRecord& record = params.meshMaterials[h.geom];
        shade::hitDifferentials(attributes, h.prim, t.a, t.b, t.c, a, b, cc, ray, st.pixelSamples ? st.pixelSamples : c.spp,
                                st);
        const DeviceTextures textures{params.textures, params.textureCount, threadRegisters(), params.mapRegisters, threadMark()};
        return shadeMaterial(record, st, textures, bsdf, presence);
    }
    // A material's Bsdf and its presence: the material module's callables (async_material.cu), or here when the
    // program is built whole (no XPU2_MATERIAL_MODULE).
    __device__ unsigned shadeMaterial(const shade::MaterialRecord& m, const shade::State& st, const DeviceTextures& textures,
                                      shade::Bsdf& bsdf, float& presence) {
        const long long from = tick();
#if XPU2_MATERIAL_MODULE
        const unsigned status = optixDirectCall<unsigned, const shade::MaterialRecord*, const float*, const shade::State*, const DeviceTextures*,
                                                shade::Bsdf*, float*, shade::DwaParams*, shade::GlitterScratch*>(
            CallShade, &m, params.materialPool, &st, &textures, &bsdf, &presence, dwaSlots(), glitter());
#else
        const unsigned status = shade::shadeMaterial(m, params.materialPool, st, textures, bsdf, presence, dwaSlots(), glitter());
#endif
        tock(ProfileShade, from);
        return status;
    }
    __device__ unsigned materialPresence(const shade::MaterialRecord& m, const shade::State& st, const DeviceTextures& textures,
                                         float& presence) {
        const long long from = tick();
#if XPU2_MATERIAL_MODULE
        const unsigned status = optixDirectCall<unsigned, const shade::MaterialRecord*, const float*, const shade::State*, const DeviceTextures*,
                                                float*>(CallPresence, &m, params.materialPool, &st, &textures, &presence);
#else
        const unsigned status = shade::materialPresence(m, params.materialPool, st, textures, presence);
#endif
        tock(ProfilePresence, from);
        return status;
    }
    __device__ bool visible(surface::V3 o, surface::V3 d) { return visible(o, d, __int_as_float(0x7f800000)); }
    // Unoccluded up to distance (a shadow ray toward a light).
    __device__ bool visible(surface::V3 o, surface::V3 d, float distance) { return visible(o, d, 0.0f, distance); }
    // Unoccluded between near and far (a shadow ray from a curve skips its epsilon).
    __device__ bool visible(surface::V3 o, surface::V3 d, float near, float distance) {
        const long long from = tick();
        if (params.profile) ++prof[ProfileCountShadow];
        unsigned occluded = 1, unused = 0;
        optixTrace(params.traversable, f3(o), f3(d), near, distance, time, 2u,  // SHADOW
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                       OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                   0, 1, 1, occluded, unused);
        // Unoccluded by what the GPU renders: a stand-in on the way could occlude it, so the CPU decides.
        if (occluded == 0 && params.standInTraversable && standInBetween(o, d, near, distance)) deferFlag = true;
        tock(ProfileShadow, from);
        return occluded == 0;
    }
    __device__ shade::Bsdf& bsdf() {
        return *reinterpret_cast<shade::Bsdf*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride);
    }
    __device__ paths::Pending* pending() {
        return reinterpret_cast<paths::Pending*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride +
                                                 ((sizeof(shade::Bsdf) + 15) & ~size_t(15)));
    }
    __device__ shade::Color* threadRegisters() {
        return reinterpret_cast<shade::Color*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride + params.registersOffset);
    }
    __device__ shade::DwaParams* dwaSlots() {
        return reinterpret_cast<shade::DwaParams*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride +
                                                   ((sizeof(shade::Bsdf) + 15) & ~size_t(15)) +
                                                   ((sizeof(paths::Pending) * paths::MaxPending + 15) & ~size_t(15)));
    }
    __device__ shade::GlitterScratch* glitter() {
        if (!params.glitterOffset) return nullptr;
        return reinterpret_cast<shade::GlitterScratch*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride +
                                                        params.glitterOffset);
    }
    __device__ const lights::Light& light(U32 i) const {
        if (params.lightMoved && params.lightMoved[i] != ~0u) return movedLight(i);
        return params.lights[i];
    }
    // The volumes, and a ray's crossings of their meshes past tnear, nearest first (the instances of visibility 2),
    // each entering or leaving by the triangle's geometric normal; repeats at one distance dropped.
    __device__ const volume::Shader* volumes() const { return params.volumes; }
    // The dense grids (a mapped BaseVolume's baked extinction); VDB volumes are the CPU's.
    __device__ volume::Grids volumeGrids() const { return volume::Grids{params.volumeGrids, params.volumeGridCount, nullptr, time}; }
    __device__ volume::Regions emissiveRegions() const { return volume::Regions{}; }  // so are emissive volumes
    __device__ const float* walkSequences() const { return params.walkSequences; }
    __device__ U32 volumeCrossings(surface::V3 o, surface::V3 d, float tnear, volume::Crossing* x, U32 max, U32 mask) {
        if (!params.volumeTraversable) return 0;
        const long long from = tick();
        U32 n = 0;
        float t = tnear;
        for (U32 k = 0; k < 4 * max && n < max; ++k) {
            unsigned instance = ~0u, primitive = 0;
            optixTrace(params.volumeTraversable, f3(o), f3(d), t, __int_as_float(0x7f800000), time, mask & 0xffu, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
                       instance, primitive);
            if (instance == ~0u || instance >= params.config->meshes) break;
            const auto m = params.geometry[instance];
            if (primitive >= m.triangleCount) break;
            const auto tri = m.triangles[primitive];
            // A placed mesh's triangle (an instance: a moved one, or shared data) where it is, as a surface's (triangleAt).
            surface::V3 a, b, c;
            triangleAt(instance, m, tri, a, b, c);
            const surface::Ray ray{o, d, {}, {}, t, __int_as_float(0x7f800000)};
            resume::Hit h;
            if (!resume::geometryHit(h, ray, a, b, c)) break;
            const surface::V3 ng = surface::cross(surface::add(b, surface::mul(a, -1.0f)), surface::add(c, surface::mul(a, -1.0f)));
            const U32 v = U32(params.meshVolumes[instance]);
            // A volume the GPU does not render (VDB, mapped colour or emission, emissive): the CPU lane renders the sample.
            if (!volume::gpuRenders(params.volumes[v], volumeGrids())) deferFlag = true;
            const U32 flags = surface::dot(d, ng) < 0.0f ? volume::Entry : volume::Exit;
            if (!(n > 0 && x[n - 1].t == h.t && x[n - 1].volume == v && x[n - 1].flags == flags)) x[n++] = volume::Crossing{h.t, v, flags};
            t = h.t + fmaxf(1e-6f, h.t * 1e-6f);
        }
        tock(ProfileVolumes, from);
        return n;
    }
    __device__ U32 lightSet(U32 geom) const { return params.meshLights[geom]; }
    __device__ const lights::TreeNode* treeNodes() const { return params.lightTree; }
    __device__ int treeRoot(U32 set) const { return params.lightTreeRoots ? params.lightTreeRoots[set] : -1; }
    __device__ bool inSet(U32 set, U32 i) const { return (params.lightSetBits[size_t(set) * params.lightSetWords + (i >> 5)] >> (i & 31) & 1u) != 0; }
    __device__ const aov::Tables& aovs() const { return *params.aovTables; }
    __device__ int geomLabel(U32 geom) const { return params.geomLabels ? params.geomLabels[geom] : -1; }
};

#if XPU2_VOLUME_MODULE
// The volumes' and subsurface's integrators in a program built in modules (INDEXER.md, 5): the volume module's
// continuation callables (async_volumes.cu), found by argument-dependent lookup from the path tracer's calls
// (async_integrator.h's volumeRadianceCall, subsurfaceRadianceCall, volumeTransmittanceCall).
__device__ inline surface::V3 volumeRadianceCall(const mixed::Config& c, const sample::Identity& root, GpuBackend& backend, paths::Pending& pv,
                                                 surface::V3 org, surface::V3 dir, float tnear, float tfar, surface::V3* alphaTransmittance,
                                                 surface::V3* minTransmittance, VolumeCryptoShares* cryptoShares, const VolumeLpe* lpe,
                                                 float* surfaceT) {
    surface::V3 out{0, 0, 0};
    optixContinuationCall<void, const mixed::Config*, const sample::Identity*, GpuBackend*, paths::Pending*, surface::V3, surface::V3, float,
                          float, surface::V3*, surface::V3*, VolumeCryptoShares*, const VolumeLpe*, float*, surface::V3*>(
        CallVolumeRadiance, &c, &root, &backend, &pv, org, dir, tnear, tfar, alphaTransmittance, minTransmittance, cryptoShares, lpe, surfaceT, &out);
    return out;
}
__device__ inline surface::V3 subsurfaceRadianceCall(const mixed::Config& c, const sample::Identity& root, GpuBackend& backend, U32 lightSet,
                                                     const paths::Pending& pv, U32 subsurfaceDepth, const shade::State& st, const resume::Hit& hit,
                                                     surface::V3 rayDir, const shade::Bsdf& bsdf, bool doIndirect, paths::Pending* stack,
                                                     U32& top, SampleStatus& status, const SubsurfaceLpe* lpe, shade::Color* ssAov) {
    surface::V3 out{0, 0, 0};
    optixContinuationCall<void, const mixed::Config*, const sample::Identity*, GpuBackend*, U32, const paths::Pending*, U32, const shade::State*,
                          const resume::Hit*, surface::V3, const shade::Bsdf*, bool, paths::Pending*, U32*, SampleStatus*, const SubsurfaceLpe*,
                          shade::Color*, surface::V3*>(CallSubsurfaceRadiance, &c, &root, &backend, lightSet, &pv, subsurfaceDepth, &st, &hit, rayDir,
                                                       &bsdf, doIndirect, stack, &top, &status, lpe, ssAov, &out);
    return out;
}
__device__ inline surface::V3 volumeTransmittanceCall(const mixed::Config& c, GpuBackend& backend, paths::Stream& s, surface::V3 org,
                                                      surface::V3 dir, float tnear, float tfar, float scale, int depth) {
    surface::V3 out{1, 1, 1};
    optixContinuationCall<void, const mixed::Config*, GpuBackend*, paths::Stream*, surface::V3, surface::V3, float, float, float, int, surface::V3*>(
        CallVolumeTransmittance, &c, &backend, &s, org, dir, tnear, tfar, scale, depth, &out);
    return out;
}
#endif


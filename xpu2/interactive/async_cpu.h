// The CPU lane of the asynchronous renderer: an Embree scene of native triangles,
// the CPU backend of the shared integrator, the host film, and workers that
// claim their own work from the grid by the cellular rules (async_grid.h).
//
// Native triangles with Embree's robust flag replace the parity pipeline's
// double-precision user-geometry test. Traversal differences between Embree
// and OptiX are expected and measured statistically (spec CPU-12, VER-3); the
// parity pipeline (deadline_pipeline.h) keeps the exact shared test.
#pragma once
#include "async_grid.h"
#include "points_core.h"
#include "cpu_jobs.h"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace xpu2::interactive::async {

// Curves on the CPU: an Embree user geometry whose spans run MoonRay XPU's GPU programs (curve_core.h), so the CPU
// finds what the GPU finds.
inline void curveBounds(const RTCBoundsFunctionArguments* args) {
    curves::Set c = *static_cast<const curves::Set*>(args->geometryUserPtr);
    if (c.stepCount > 1) c.time = float(args->timeStep) / float(c.stepCount - 1);
    float lower[3], upper[3];
    curves::spanBounds(c, args->primID, lower, upper);
    args->bounds_o->lower_x = lower[0];
    args->bounds_o->lower_y = lower[1];
    args->bounds_o->lower_z = lower[2];
    args->bounds_o->upper_x = upper[0];
    args->bounds_o->upper_y = upper[1];
    args->bounds_o->upper_z = upper[2];
}
inline void curveIntersect(const RTCIntersectFunctionNArguments* args) {
    if (args->N != 1 || !args->valid[0]) return;
    auto* rh = reinterpret_cast<RTCRayHit*>(args->rayhit);
    curves::Set c = *static_cast<const curves::Set*>(args->geometryUserPtr);
    c.time = rh->ray.time;
    curves::Hit h;
    if (!curves::intersectSpan(c, args->primID, {rh->ray.org_x, rh->ray.org_y, rh->ray.org_z}, {rh->ray.dir_x, rh->ray.dir_y, rh->ray.dir_z},
                               rh->ray.tnear, rh->ray.tfar, h))
        return;
    rh->ray.tfar = h.t;
    rh->hit.u = h.u;
    rh->hit.v = h.v;
    rh->hit.Ng_x = h.Ng.x;
    rh->hit.Ng_y = h.Ng.y;
    rh->hit.Ng_z = h.Ng.z;
    rh->hit.primID = args->primID;
    rh->hit.geomID = args->geomID;
    rh->hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
}
// A user geometry's occluding hit through the occlusion test's argument filter (shadow linking's linkFilter), as
// Embree runs it on its own geometries' hits: rejected, no hit.
inline bool occlusionAccepted(const RTCOccludedFunctionNArguments* args, float t, surface::V3 Ng, float u, float v) {
    auto* ray = reinterpret_cast<RTCRay*>(args->ray);
    RTCHit hit{};
    hit.Ng_x = Ng.x;
    hit.Ng_y = Ng.y;
    hit.Ng_z = Ng.z;
    hit.u = u;
    hit.v = v;
    hit.primID = args->primID;
    hit.geomID = args->geomID;
    hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    const float far = ray->tfar;
    ray->tfar = t;
    int valid = -1;
    RTCFilterFunctionNArguments f;
    f.valid = &valid;
    f.geometryUserPtr = args->geometryUserPtr;
    f.context = args->context;
    f.ray = reinterpret_cast<RTCRayN*>(args->ray);
    f.hit = reinterpret_cast<RTCHitN*>(&hit);
    f.N = 1;
    rtcInvokeOccludedFilterFromGeometry(args, &f);
    if (valid == 0) {
        ray->tfar = far;
        return false;
    }
    return true;
}
inline void curveOccluded(const RTCOccludedFunctionNArguments* args) {
    if (args->N != 1 || !args->valid[0]) return;
    auto* ray = reinterpret_cast<RTCRay*>(args->ray);
    curves::Set c = *static_cast<const curves::Set*>(args->geometryUserPtr);
    c.time = ray->time;
    curves::Hit h;
    if (curves::intersectSpan(c, args->primID, {ray->org_x, ray->org_y, ray->org_z}, {ray->dir_x, ray->dir_y, ray->dir_z}, ray->tnear,
                              ray->tfar, h) &&
        occlusionAccepted(args, h.t, h.Ng, h.u, h.v))
        ray->tfar = -INFINITY;
}
// A curve hit worked out again from its span, as the GPU lane does: its distance, coordinates and normal, the
// point rays leave from, and the epsilon they skip (LineSegments and CubicSpline: 1.5 times the radius there).
inline bool curveHit(const curves::Set& c, const surface::Ray& ray, resume::Hit& out) {
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
        radius = curves::shadingPoint(c, first, c.time).w * (1.0f - h.u) + curves::shadingPoint(c, first + 1, c.time).w * h.u;
    } else {
        float w[4];
        curves::cubicWeights(c.basis, h.u, w);
        radius = w[0] * curves::shadingPoint(c, first, c.time).w + w[1] * curves::shadingPoint(c, first + 1, c.time).w +
                 w[2] * curves::shadingPoint(c, first + 2, c.time).w + w[3] * curves::shadingPoint(c, first + 3, c.time).w;
    }
    out.epsilon = 1.5f * radius;
    return shade::finite3(out.point) && out.epsilon > 0.0f;
}

// MoonRay's backFaceCullingFilter for single-sided meshes: a hit the ray meets along the geometric normal is no hit.
inline void backFaceCullingFilter(const RTCFilterFunctionNArguments* args) {
    int* valid = args->valid;
    for (unsigned i = 0; i < args->N; ++i) {
        if (!valid[i]) continue;
        const float d = RTCRayN_dir_x(args->ray, args->N, i) * RTCHitN_Ng_x(args->hit, args->N, i) +
                        RTCRayN_dir_y(args->ray, args->N, i) * RTCHitN_Ng_y(args->hit, args->N, i) +
                        RTCRayN_dir_z(args->ray, args->N, i) * RTCHitN_Ng_z(args->hit, args->N, i);
        if (d >= 0.0f) valid[i] = 0;
    }
}

// Points (points_core.h) as user geometry: each one's box over its time steps, and MoonRay's sphere test at the ray's
// time.
inline void pointBounds(const RTCBoundsFunctionArguments* args) {
    const auto& m = *static_cast<const cpu::MeshData*>(args->geometryUserPtr);
    const unsigned k = args->timeStep;
    const surface::V3 p = k ? m.steps[k - 1][args->primID] : m.positions[args->primID];
    const float r = m.pointRadius[args->primID];
    args->bounds_o->lower_x = p.x - r;
    args->bounds_o->lower_y = p.y - r;
    args->bounds_o->lower_z = p.z - r;
    args->bounds_o->upper_x = p.x + r;
    args->bounds_o->upper_y = p.y + r;
    args->bounds_o->upper_z = p.z + r;
}
inline void pointIntersect(const RTCIntersectFunctionNArguments* args) {
    if (args->N != 1 || !args->valid[0]) return;
    const auto& m = *static_cast<const cpu::MeshData*>(args->geometryUserPtr);
    auto* rh = reinterpret_cast<RTCRayHit*>(args->rayhit);
    float t;
    surface::V3 Ng;
    if (!points::intersect(cpu::positionAt(m, args->primID, rh->ray.time), m.pointRadius[args->primID], {rh->ray.org_x, rh->ray.org_y, rh->ray.org_z},
                           {rh->ray.dir_x, rh->ray.dir_y, rh->ray.dir_z}, rh->ray.tnear, rh->ray.tfar, t, Ng))
        return;
    rh->ray.tfar = t;
    rh->hit.u = rh->hit.v = 0.0f;
    rh->hit.Ng_x = Ng.x;
    rh->hit.Ng_y = Ng.y;
    rh->hit.Ng_z = Ng.z;
    rh->hit.primID = args->primID;
    rh->hit.geomID = args->geomID;
    rh->hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
}
inline void pointOccluded(const RTCOccludedFunctionNArguments* args) {
    if (args->N != 1 || !args->valid[0]) return;
    const auto& m = *static_cast<const cpu::MeshData*>(args->geometryUserPtr);
    auto* ray = reinterpret_cast<RTCRay*>(args->ray);
    float t;
    surface::V3 Ng;
    if (points::intersect(cpu::positionAt(m, args->primID, ray->time), m.pointRadius[args->primID], {ray->org_x, ray->org_y, ray->org_z},
                          {ray->dir_x, ray->dir_y, ray->dir_z}, ray->tnear, ray->tfar, t, Ng) &&
        occlusionAccepted(args, t, Ng, 0.0f, 0.0f))
        ray->tfar = -INFINITY;
}

// The CPU's Embree scenes' flags: robust, compact (its trees in less memory, about a third, for a few percent of the
// tracing's speed; XPU2_SESSION_EMBREE_COMPACT=0: as before), and with shadow linking the occlusion tests' filter.
// Realtime's CPU scene in two levels: each triangle mesh a tree of its own, made once, and the scene their instances, so
// that a move sets the moved meshes' transforms and Embree sorts the instances again, building no mesh's tree (CpuScene's
// replacing constructor). XPU2_SESSION_EMBREE_DYNAMIC=0: never (one level, as a final render's), =1: always.
inline bool cpuTwoLevels(const cpu::Plan& plan) {
    static const int two = [] { const char* v = std::getenv("XPU2_SESSION_EMBREE_DYNAMIC"); return v ? (v[0] == '1' ? 1 : 0) : -1; }();
    return two == 1 || (two < 0 && plan.realtime);
}
// top: the scene of a plan's meshes (dynamic in two levels: its instances' tree made fast), not a mesh's own.
inline RTCSceneFlags cpuSceneFlags(const cpu::Plan& plan, bool top = false) {
    static const bool compact = [] { const char* v = std::getenv("XPU2_SESSION_EMBREE_COMPACT"); return !(v && v[0] == '0'); }();
    int f = RTC_SCENE_FLAG_ROBUST | (compact ? RTC_SCENE_FLAG_COMPACT : 0);
    if (top && cpuTwoLevels(plan)) f |= RTC_SCENE_FLAG_DYNAMIC;
    if (!plan.shadowLinks.empty()) f |= RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS;
    return RTCSceneFlags(f);
}
// Embree scene for the fast lanes. Mesh i is geometry i, as in the plan. Each geometry's mask is its visibility.
struct CpuScene {
    std::shared_ptr<cpu::Device> device;
    RTCScene scene = nullptr;
    RTCScene volumes = nullptr;  // the meshes that bound volumes (volume_core.h), for their crossings
    cpu::PlanView plan;
    std::vector<curves::Set> curveSets;  // a set a mesh (the user geometries' data)
    aov::Tables aovTables;               // the render outputs' AOV schema (aov_plan.h), none when they have none
    // Shared instances' data (cpu::Mesh::instance): one Embree scene of each, which its instances' geometries instance.
    std::map<const cpu::MeshData*, RTCScene> shared;
    // Two levels (cpuTwoLevels): each triangle mesh's own scene, by its data and sidedness (the back faces' filter is
    // its geometry's), made once and kept across moves.
    std::map<std::pair<const cpu::MeshData*, bool>, RTCScene> own;
    std::vector<std::shared_ptr<const cpu::MeshData>> ownData;  // held, so an address names no other mesh while kept
    // A placed mesh that bounds a volume (a moved one, cpu::placeVolume): its data's scene in the volumes' scene, by
    // instance.
    std::map<const cpu::MeshData*, RTCScene> volumeOwn;
    // The plan of the scenes taken over (the replacing constructor's), let go once it is done.
    cpu::PlanView before;
    // How many meshes the replacing constructor moved by their transforms alone, and how many it made again.
    unsigned moved = 0, remade = 0;
    explicit CpuScene(cpu::PlanView p) : device(std::make_shared<cpu::Device>()), plan(std::move(p)) {
        if (plan && plan->aovs) {
            aovTables = plan->aovs->tables();
            aovTables.primKeys = plan->aovPrimKeys.data();
            aovTables.primCount = unsigned(plan->aovPrimKeys.size());
        }
        cpu::need(plan && cpu::isSpatial(plan->domain), "Asynchronous renderer requires a spatial plan");
        scene = rtcNewScene(device->handle);
        cpu::need(scene, "Embree scene failure");
        rtcSetSceneFlags(scene, cpuSceneFlags(*plan, true));
        rtcSetSceneBuildQuality(scene, cpuTwoLevels(*plan) ? RTC_BUILD_QUALITY_LOW : RTC_BUILD_QUALITY_HIGH);
        curveSets.resize(plan->meshes.size());
        for (unsigned i = 0; i < plan->meshes.size(); ++i) add(i);
        rtcCommitScene(scene);
        if (volumes) rtcCommitScene(volumes);
        device->check();
    }
    // Mesh i's geometry, as geometry i of the scene (or of the volumes' scene, where it bounds a volume).
    // A mesh's transform as Embree's 3x4 column-major: an instance's, or none (its positions are placed already).
    const float* placement(unsigned i) const {
        static const float none[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
        return plan->meshes[i].instance ? plan->meshes[i].objectToWorld : none;
    }
    // Whether mesh i is an instance of its own scene (two levels): a still triangle mesh that is a surface.
    bool ownScene(unsigned i) const {
        const auto& m = *plan->meshes[i].data;
        return cpuTwoLevels(*plan) && !m.points && !m.curveBasis && m.steps.empty() && plan->meshes[i].material != ~0u;
    }
    void add(unsigned i) {
        const auto& m = *plan->meshes[i].data;
        if (ownScene(i)) {
            RTCScene& inner = own[{&m, plan->meshes[i].singleSided}];
            if (!inner) {
                inner = rtcNewScene(device->handle);
                cpu::need(inner, "Embree scene failure");
                ownData.push_back(plan->meshes[i].data);
                rtcSetSceneFlags(inner, cpuSceneFlags(*plan));
                rtcSetSceneBuildQuality(inner, RTC_BUILD_QUALITY_HIGH);
                RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_TRIANGLE);
                cpu::need(g, "Embree geometry failure");
                void* v = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(surface::V3), m.positions.size());
                auto* tr = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(cpu::Tri), m.triangles.size());
                if (!v || !tr) {
                    rtcReleaseGeometry(g);
                    throw cpu::Unsupported("Embree allocation failure");
                }
                std::memcpy(v, m.positions.data(), m.positions.size() * sizeof(surface::V3));
                std::memcpy(tr, m.triangles.data(), m.triangles.size() * sizeof(cpu::Tri));
                rtcSetGeometryMask(g, 0xffffffffu);
                if (plan->meshes[i].singleSided) {
                    rtcSetGeometryIntersectFilterFunction(g, backFaceCullingFilter);
                    rtcSetGeometryOccludedFilterFunction(g, backFaceCullingFilter);
                }
                rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
                rtcCommitGeometry(g);
                rtcAttachGeometryByID(inner, g, 0);
                rtcReleaseGeometry(g);
                rtcCommitScene(inner);
            }
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_INSTANCE);
            cpu::need(g, "Embree geometry failure");
            rtcSetGeometryInstancedScene(g, inner);
            rtcSetGeometryTimeStepCount(g, 1);
            rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, placement(i));
            rtcSetGeometryMask(g, plan->meshes[i].visibility);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(scene, g, i);
            rtcReleaseGeometry(g);
            // Bounding a volume as well: in the volumes' scene as before (below, its surface pass left out).
            if (plan->meshes[i].volume < 0 || surfaceOnly) return;
        } else if (plan->meshes[i].instance) {
            // An instance of shared data: its data's scene (made with the first instance), placed by the
            // instance's transform (rows vx, vy, vz, p: Embree's 3x4 column-major), masked by its visibility.
            RTCScene& inner = shared[&m];
            if (!inner) {
                inner = rtcNewScene(device->handle);
                cpu::need(inner, "Embree scene failure");
                rtcSetSceneFlags(inner, cpuSceneFlags(*plan));
                rtcSetSceneBuildQuality(inner, RTC_BUILD_QUALITY_HIGH);
                RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_TRIANGLE);
                cpu::need(g, "Embree geometry failure");
                void* v = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(surface::V3), m.positions.size());
                auto* tr = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(cpu::Tri), m.triangles.size());
                if (!v || !tr) {
                    rtcReleaseGeometry(g);
                    throw cpu::Unsupported("Embree allocation failure");
                }
                std::memcpy(v, m.positions.data(), m.positions.size() * sizeof(surface::V3));
                std::memcpy(tr, m.triangles.data(), m.triangles.size() * sizeof(cpu::Tri));
                rtcSetGeometryMask(g, 0xffffffffu);
                rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
                rtcCommitGeometry(g);
                rtcAttachGeometryByID(inner, g, 0);
                rtcReleaseGeometry(g);
                rtcCommitScene(inner);
            }
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_INSTANCE);
            cpu::need(g, "Embree geometry failure");
            rtcSetGeometryInstancedScene(g, inner);
            rtcSetGeometryTimeStepCount(g, 1);
            rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, plan->meshes[i].objectToWorld);
            rtcSetGeometryMask(g, plan->meshes[i].visibility);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(scene, g, i);
            rtcReleaseGeometry(g);
            if (plan->meshes[i].volume < 0 || surfaceOnly) return;
        }
        if (m.points) {
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_USER);
            cpu::need(g, "Embree geometry failure");
            if (!m.steps.empty()) rtcSetGeometryTimeStepCount(g, unsigned(m.steps.size()) + 1);
            rtcSetGeometryUserPrimitiveCount(g, m.points);
            rtcSetGeometryUserData(g, const_cast<cpu::MeshData*>(&m));
            rtcSetGeometryBoundsFunction(g, pointBounds, nullptr);
            rtcSetGeometryIntersectFunction(g, pointIntersect);
            rtcSetGeometryOccludedFunction(g, pointOccluded);
            rtcSetGeometryMask(g, plan->meshes[i].visibility);
            rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(scene, g, i);
            rtcReleaseGeometry(g);
            return;
        }
        if (m.curveBasis) {
            curveSets[i] = cpu::curveSet(m);
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_USER);
            cpu::need(g, "Embree geometry failure");
            if (curveSets[i].stepCount > 1) rtcSetGeometryTimeStepCount(g, curveSets[i].stepCount);
            rtcSetGeometryUserPrimitiveCount(g, curveSets[i].spanCount);
            rtcSetGeometryUserData(g, &curveSets[i]);
            rtcSetGeometryBoundsFunction(g, curveBounds, nullptr);
            rtcSetGeometryIntersectFunction(g, curveIntersect);
            rtcSetGeometryOccludedFunction(g, curveOccluded);
            rtcSetGeometryMask(g, plan->meshes[i].visibility);
            rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(scene, g, i);
            rtcReleaseGeometry(g);
            return;
        }
        // A mesh with no material bounds a volume only: it is in the volumes' scene, not the surfaces'.
        const bool surface = plan->meshes[i].material != ~0u, bounds = plan->meshes[i].volume >= 0;
        for (int pass = ownScene(i) || plan->meshes[i].instance ? 1 : 0; pass < 2; ++pass) {
            if (pass == 0 && !surface) continue;
            if (pass == 1 && !bounds) return;
            if (pass == 1 && surfaceOnly) return;
            if (pass == 1 && !volumes) {
                volumes = rtcNewScene(device->handle);
                cpu::need(volumes, "Embree scene failure");
                rtcSetSceneFlags(volumes, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_COMPACT);
            }
            if (pass == 1 && plan->meshes[i].instance) {
                addVolumeInstance(i);
                return;
            }
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_TRIANGLE);
            cpu::need(g, "Embree geometry failure");
            // A moving mesh's time steps over the shutter (MeshData::steps), linear between them as MoonRay's.
            const unsigned steps = unsigned(m.steps.size()) + 1;
            if (steps > 1) rtcSetGeometryTimeStepCount(g, steps);
            void* v = nullptr;
            for (unsigned k = 0; k < steps; ++k) {
                const std::vector<surface::V3>& src = k ? m.steps[k - 1] : m.positions;
                void* slot = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, k, RTC_FORMAT_FLOAT3, sizeof(surface::V3), src.size());
                if (!slot) {
                    rtcReleaseGeometry(g);
                    throw cpu::Unsupported("Embree allocation failure");
                }
                std::memcpy(slot, src.data(), src.size() * sizeof(surface::V3));
                if (!k) v = slot;
            }
            auto* t = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(cpu::Tri),
                                              m.triangles.size());
            if (!v || !t) {
                rtcReleaseGeometry(g);
                throw cpu::Unsupported("Embree allocation failure");
            }
            std::memcpy(t, m.triangles.data(), m.triangles.size() * sizeof(cpu::Tri));
            rtcSetGeometryMask(g, plan->meshes[i].visibility);
            if (pass == 0 && plan->meshes[i].singleSided) {
                rtcSetGeometryIntersectFilterFunction(g, backFaceCullingFilter);
                rtcSetGeometryOccludedFilterFunction(g, backFaceCullingFilter);
            }
            rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(pass == 0 ? scene : volumes, g, i);
            rtcReleaseGeometry(g);
        }
    }
    // Mesh i's surface alone (add without its bounds of a volume, which the replacing constructor places itself).
    bool surfaceOnly = false;
    void addSurface(unsigned i) {
        surfaceOnly = true;
        try {
            add(i);
        } catch (...) {
            surfaceOnly = false;
            throw;
        }
        surfaceOnly = false;
    }
    // A placed mesh's bounds in the volumes' scene: an instance of its data's scene there (made with the first).
    void addVolumeInstance(unsigned i) {
        const auto& m = *plan->meshes[i].data;
        RTCScene& inner = volumeOwn[&m];
        if (!inner) {
            inner = rtcNewScene(device->handle);
            cpu::need(inner, "Embree scene failure");
            ownData.push_back(plan->meshes[i].data);
            rtcSetSceneFlags(inner, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_COMPACT);
            RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_TRIANGLE);
            cpu::need(g, "Embree geometry failure");
            void* v = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(surface::V3), m.positions.size());
            auto* tr = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(cpu::Tri), m.triangles.size());
            if (!v || !tr) {
                rtcReleaseGeometry(g);
                throw cpu::Unsupported("Embree allocation failure");
            }
            std::memcpy(v, m.positions.data(), m.positions.size() * sizeof(surface::V3));
            std::memcpy(tr, m.triangles.data(), m.triangles.size() * sizeof(cpu::Tri));
            rtcSetGeometryMask(g, 0xffffffffu);
            rtcCommitGeometry(g);
            rtcAttachGeometryByID(inner, g, 0);
            rtcReleaseGeometry(g);
            rtcCommitScene(inner);
        }
        RTCGeometry g = rtcNewGeometry(device->handle, RTC_GEOMETRY_TYPE_INSTANCE);
        cpu::need(g, "Embree geometry failure");
        rtcSetGeometryInstancedScene(g, inner);
        rtcSetGeometryTimeStepCount(g, 1);
        rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, placement(i));
        rtcSetGeometryMask(g, plan->meshes[i].visibility);
        rtcCommitGeometry(g);
        rtcAttachGeometryByID(volumes, g, i);
        rtcReleaseGeometry(g);
    }
    // The previous frame's Embree scenes taken over, where every mesh is the one they hold (sameGeometry): a camera, a
    // light or a material edited, they are not built again. The previous scene is spent (the session's renderer, which
    // held it, is gone): its curve sets move with the scenes, whose user geometries point into them.
    CpuScene(cpu::PlanView p, CpuScene&& previous)
        : device(std::move(previous.device)), scene(previous.scene), volumes(previous.volumes), plan(std::move(p)),
          curveSets(std::move(previous.curveSets)) {
        previous.scene = previous.volumes = nullptr;
        shared = std::move(previous.shared);
        previous.shared.clear();
        own = std::move(previous.own);
        previous.own.clear();
        volumeOwn = std::move(previous.volumeOwn);
        previous.volumeOwn.clear();
        ownData = std::move(previous.ownData);
        if (plan && plan->aovs) {
            aovTables = plan->aovs->tables();
            aovTables.primKeys = plan->aovPrimKeys.data();
            aovTables.primCount = unsigned(plan->aovPrimKeys.size());
        }
    }
    // The previous frame's Embree scenes taken over and the meshes replaced alone made again (replacedMeshes: moved).
    // Two levels: a moved mesh whose own scene is the one it had takes its new transform alone, and no mesh's tree is
    // built; Embree sorts the instances again at the commit.
    CpuScene(cpu::PlanView p, CpuScene&& previous, const std::vector<unsigned>& replaced) : CpuScene(std::move(p), std::move(previous)) {
        // The plan the taken scenes were made for (the previous scene keeps it; its scenes are this one's now).
        before = previous.plan;
        bool volumesChanged = false;
        for (unsigned i : replaced) {
            const auto &x = before->meshes[i], &y = plan->meshes[i];
            const bool surface = y.material != ~0u, bounds = y.volume >= 0 && volumes;
            // Its bounds of a volume: an instance placed anew, or made an instance (the volumes' scene is sorted again).
            if (bounds) {
                volumesChanged = true;
                if (x.instance && x.data == y.data && volumeOwn.count(y.data.get())) {
                    RTCGeometry g = rtcGetGeometry(volumes, i);
                    rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, placement(i));
                    rtcSetGeometryMask(g, y.visibility);
                    rtcCommitGeometry(g);
                } else {
                    rtcDetachGeometry(volumes, i);
                    if (y.instance) addVolumeInstance(i);
                }
            }
            if (!surface) {
                ++moved;
                continue;
            }
            if (ownScene(i) && cpuTwoLevels(*before) && x.data == y.data && x.singleSided == y.singleSided && x.material != ~0u) {
                RTCGeometry g = rtcGetGeometry(scene, i);
                rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, placement(i));
                rtcSetGeometryMask(g, y.visibility);
                rtcCommitGeometry(g);
                ++moved;
                continue;
            }
            rtcDetachGeometry(scene, i);
            addSurface(i);
            ++remade;
        }
        before.reset();
        rtcCommitScene(scene);
        if (volumesChanged) rtcCommitScene(volumes);
        device->check();
    }
    // The meshes' own scenes no mesh of the plan instances any more (an edited mesh's data before its edit) let go.
    void prune() {
        std::set<std::pair<const cpu::MeshData*, bool>> used;
        std::set<const cpu::MeshData*> usedVolumes, held;
        for (const auto& e : plan->meshes) {
            used.insert({e.data.get(), e.singleSided});
            usedVolumes.insert(e.data.get());
        }
        for (auto k = own.begin(); k != own.end();)
            if (used.count(k->first)) {
                held.insert(k->first.first);
                ++k;
            } else {
                rtcReleaseScene(k->second);
                k = own.erase(k);
            }
        for (auto k = volumeOwn.begin(); k != volumeOwn.end();)
            if (usedVolumes.count(k->first)) {
                held.insert(k->first);
                ++k;
            } else {
                rtcReleaseScene(k->second);
                k = volumeOwn.erase(k);
            }
        ownData.erase(std::remove_if(ownData.begin(), ownData.end(), [&](const auto& d) { return !held.count(d.get()); }), ownData.end());
    }
    // The meshes whose geometries differ between two plans' Embree scenes, where nothing else does (the scenes' flags,
    // their count) and none of them bounds a volume (the volumes' scene is made whole): false otherwise.
    static bool replacedMeshes(const cpu::Plan& a, const cpu::Plan& b, std::vector<unsigned>& replaced) {
        replaced.clear();
        if (a.meshes.size() != b.meshes.size() || cpuSceneFlags(a, true) != cpuSceneFlags(b, true)) return false;
        for (unsigned i = 0; i < a.meshes.size(); ++i) {
            const auto &x = a.meshes[i], &y = b.meshes[i];
            if (x.data == y.data && x.visibility == y.visibility && x.volume == y.volume && x.singleSided == y.singleSided &&
                x.material == y.material && x.instance == y.instance && std::memcmp(x.objectToWorld, y.objectToWorld, sizeof x.objectToWorld) == 0)
                continue;
            if (x.volume != y.volume || (x.volume >= 0 && (y.data->steps.size() || !y.instance))) return false;
            replaced.push_back(i);
        }
        return true;
    }
    // Whether two plans' Embree scenes are the same: each mesh the same prepared data (a mesh kept across an edit), with
    // the same visibility, volume, sidedness and material, and shadow linking (the scene's flags) as it was.
    static bool sameGeometry(const cpu::Plan& a, const cpu::Plan& b) {
        if (a.meshes.size() != b.meshes.size() || a.shadowLinks.empty() != b.shadowLinks.empty()) return false;
        for (size_t i = 0; i < a.meshes.size(); ++i) {
            const auto &x = a.meshes[i], &y = b.meshes[i];
            if (x.data != y.data || x.visibility != y.visibility || x.volume != y.volume || x.singleSided != y.singleSided || x.material != y.material ||
                x.instance != y.instance || std::memcmp(x.objectToWorld, y.objectToWorld, sizeof x.objectToWorld) != 0)
                return false;
        }
        return true;
    }
    ~CpuScene() {
        if (scene) rtcReleaseScene(scene);
        if (volumes) rtcReleaseScene(volumes);
        for (auto& [data, inner] : shared) rtcReleaseScene(inner);
        for (auto& [key, inner] : own) rtcReleaseScene(inner);
        for (auto& [data, inner] : volumeOwn) rtcReleaseScene(inner);
    }
    CpuScene(const CpuScene&) = delete;
};

struct CpuBackend;
// The CPU lane's sample (integrate over CpuBackend). With XPU2_SPLIT_UNITS it is compiled in xpu2_cpu.cc alone.
#if defined(XPU2_SPLIT_UNITS)
SampleStatus integrateCpu(const mixed::Config& c, U32 pixel, U32 index, CpuBackend& backend, Radiance& out);
#else
inline SampleStatus integrateCpu(const mixed::Config& c, U32 pixel, U32 index, CpuBackend& backend, Radiance& out);
#endif
struct CpuBackend {
    const CpuScene& s;
    // The sample's time in the shutter (async_integrator.h sets it): every ray is traced at it, and the lights moving
    // under motion blur are taken at it (their records at that time, in moved).
    float time = 0.0f;
    std::vector<lights::Light> moved;
    void setTime(float t) {
        time = t;
        const auto& plan = *s.plan;
        if (plan.lightMotion.empty()) return;
        if (moved.size() != plan.lights.size()) moved = plan.lights;
        for (const auto& m : plan.lightMotion) moved[m.light] = lights::lightAtTime(plan.lights[m.light], m, t);
    }
    // A mesh's vertex at that time, and a moving mesh's time steps for its shading (shade::MeshAttributes).
    surface::V3 vertex(const cpu::MeshData& m, U32 i) const { return cpu::positionAt(m, i, time); }
    // Plan mesh geom's vertex where it is: an instance's placed by its transform (shared data is in its own space).
    surface::V3 vertexOf(U32 geom, const cpu::MeshData& m, U32 i) const {
        const surface::V3 p = vertex(m, i);
        const auto& e = s.plan->meshes[geom];
        if (!e.instance) return p;
        const float* x = e.objectToWorld;
        return {p.x * x[0] + p.y * x[3] + p.z * x[6] + x[9], p.x * x[1] + p.y * x[4] + p.z * x[7] + x[10], p.x * x[2] + p.y * x[5] + p.z * x[8] + x[11]};
    }
    // An instance's shading attributes (shade::MeshAttributes::instance): its transforms and its own primitive attributes.
    static void instanceInto(const cpu::Mesh& e, shade::MeshAttributes& a) {
        if (!e.instance) return;
        a.instance = e.instance;
        for (int k = 0; k < 12; ++k) a.objectToWorld[k] = e.objectToWorld[k];
        for (int k = 0; k < 9; ++k) a.normalToWorld[k] = e.normalToWorld[k];
        for (int k = 0; k < 12; ++k) a.worldToObject[k] = e.worldToObject[k];
        a.instancePrims = e.instancePrims.empty() ? nullptr : e.instancePrims.data();
        a.instancePrimData = e.instancePrimData.empty() ? nullptr : e.instancePrimData.data();
        a.instancePrimCount = unsigned(e.instancePrims.size());
    }
    // A plan mesh's object space: an instance's own, else its data's.
    static const float* worldToObjectOf(const cpu::Mesh& e) { return e.instance ? e.worldToObject : e.data->worldToObject; }
    void motion(const cpu::MeshData& m, shade::MeshAttributes& a) const {
        if (m.steps.empty()) return;
        if (m.subdSteps) {
            // OpenSubdivMesh::postIntersect: normals and derivatives between the time steps about the ray's time; the
            // motion vector and polyvertices from the first two steps (Mesh::computeMotion).
            const unsigned n = unsigned(m.steps.size()) + 1;
            const float f = time * float(n - 1);
            const unsigned k = std::min(unsigned(std::max(f, 0.0f)), n - 2);
            a.positions0 = m.positions.data();
            a.positions1 = m.steps[0].data();
            a.time = time;
            a.normals = k ? m.stepNormals[k - 1].data() : m.normals.data();
            a.normals1 = m.stepNormals[k].data();
            a.dpds = k ? m.stepDpds[k - 1].data() : m.dpds.data();
            a.dpds1 = m.stepDpds[k].data();
            a.dpdt = k ? m.stepDpdt[k - 1].data() : m.dpdt.data();
            a.dpdt1 = m.stepDpdt[k].data();
            a.sampleTime = f - float(k);
            return;
        }
        if (m.stepShading) {
            const unsigned n = unsigned(m.steps.size()) + 1;
            const float f = time * float(n - 1);
            const unsigned k = std::min(unsigned(std::max(f, 0.0f)), n - 2);
            a.positions0 = k ? m.steps[k - 1].data() : m.positions.data();
            a.positions1 = m.steps[k].data();
            if (!m.stepNormals.empty()) {
                a.normals = k ? m.stepNormals[k - 1].data() : m.normals.data();
                a.normals1 = m.stepNormals[k].data();
            }
            // A tessellated prototype's instance: its limit derivatives at the steps about the ray's time as well.
            if (!m.stepDpds.empty()) {
                a.dpds = k ? m.stepDpds[k - 1].data() : m.dpds.data();
                a.dpds1 = m.stepDpds[k].data();
                a.dpdt = k ? m.stepDpdt[k - 1].data() : m.dpdt.data();
                a.dpdt1 = m.stepDpdt[k].data();
                a.sampleTime = f - float(k);
            }
            a.time = f - float(k);
            a.motionScale = float(n - 1);
            return;
        }
        a.positions0 = m.positions.data();
        a.positions1 = m.steps[0].data();
        a.normals1 = m.normals1.empty() ? nullptr : m.normals1.data();
        a.time = time;
    }
    resume::Hit trace(const surface::Ray& ray) const {
        RTCRayHit h{};
        h.ray.org_x = ray.origin.x;
        h.ray.org_y = ray.origin.y;
        h.ray.org_z = ray.origin.z;
        h.ray.dir_x = ray.direction.x;
        h.ray.dir_y = ray.direction.y;
        h.ray.dir_z = ray.direction.z;
        h.ray.tnear = ray.nearT;
        h.ray.tfar = ray.farT;
        h.ray.mask = ray.mask;
        h.ray.time = time;
        h.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect1(s.scene, &h, &args);
        resume::Hit out;
        if (h.hit.geomID == RTC_INVALID_GEOMETRY_ID) return out;
        out.geom = h.hit.instID[0] != RTC_INVALID_GEOMETRY_ID ? h.hit.instID[0] : h.hit.geomID;
        out.prim = h.hit.primID;
        const auto& mesh = *s.plan->meshes[out.geom].data;
        if (mesh.points) {
            // The sphere again, from the ray (Points::postIntersect's P on it, Ng from the centre); new rays leave
            // just off it.
            float t;
            surface::V3 Ng;
            if (!points::intersect(cpu::positionAt(mesh, out.prim, time), mesh.pointRadius[out.prim], ray.origin, ray.direction, ray.nearT,
                                   ray.farT, t, Ng))
                return resume::Hit{};
            out.t = t;
            out.u = out.v = 0.0f;
            out.point = surface::add(ray.origin, surface::mul(ray.direction, t));
            surface::V3 n;
            if (!surface::normalize(Ng, n)) return resume::Hit{};
            out.normal = surface::dot(n, ray.direction) > 0.0f ? surface::mul(n, -1.0f) : n;
            const float eps = 1e-5f * fmaxf(1.0f, fmaxf(fabsf(out.point.x), fmaxf(fabsf(out.point.y), fabsf(out.point.z))));
            out.origin = surface::add(out.point, surface::mul(out.normal, eps));
            out.below = surface::add(out.point, surface::mul(out.normal, -eps));
            return out;
        }
        if (mesh.curveBasis) {
            curves::Set c = s.curveSets[out.geom];
            c.time = time;
            if (!curveHit(c, ray, out)) return resume::Hit{};
            return out;
        }
        const auto t = mesh.triangles[out.prim];
        // Shared reconstruction of distance, coordinates and the offset origin.
        if (!resume::geometryHit(out, ray, vertexOf(out.geom, mesh, t.a), vertexOf(out.geom, mesh, t.b), vertexOf(out.geom, mesh, t.c)))
            out.normal = {};
        return out;
    }
    unsigned material(const resume::Hit& h, const surface::Ray& ray, surface::Lambert& out) const {
        RTCRayHit hit{};
        hit.hit.geomID = h.geom;
        hit.hit.primID = h.prim;
        hit.hit.u = h.u;
        hit.hit.v = h.v;
        // Host RAM holds every level of every texture, so the CPU never misses.
        out = mixed::material(s.plan, hit, ray);
        return tex::Ready;
    }
    // A mesh's material identity (its record's pool offset).
    U32 materialId(U32 geom) const {
        const auto& plan = *s.plan;
        const U32 m = plan.meshes[geom].material;
        return m < plan.materialRecords.size() ? plan.materialRecords[m].offset : ~0u;
    }
    // The first hit on a mesh of the given material (MoonRay's subsurface probe rays skip the others).
    resume::Hit traceMaterial(surface::Ray ray, U32 material) const {
        for (int k = 0; k < 64; ++k) {
            const resume::Hit h = trace(ray);
            if (h.geom == ~0u || materialId(h.geom) == material) return h;
            ray.nearT = h.t + fmaxf(1e-6f, h.t * 1e-6f);
            if (!(ray.nearT < ray.farT)) break;
        }
        return resume::Hit{};
    }
    // A hit's geometric shading state and differentials, its material not run.
    // A point set's shading attributes (its primitive attributes and time steps).
    shade::MeshAttributes pointAttributes(const cpu::MeshData& data) const {
        shade::MeshAttributes attributes;
        for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = data.worldToObject[k];
        attributes.prims = data.prims.empty() ? nullptr : data.prims.data();
        attributes.primData = data.primData.empty() ? nullptr : data.primData.data();
        attributes.primCount = unsigned(data.prims.size());
        motion(data, attributes);
        return attributes;
    }
    bool surfaceState(const resume::Hit& h, const surface::Ray& ray, shade::State& st) const {
        const auto& plan = *s.plan;
        const auto& data = *plan.meshes[h.geom].data;
        if (data.points) {
            if (!points::state(pointAttributes(data), h.prim, h.normal, h.point, ray.direction, st)) return false;
            st.geom = h.geom;
            return true;
        }
        if (data.curveBasis) {
            const shade::MeshAttributes attributes = curveAttributes(data, h.geom);
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return false;
            st.geom = h.geom;
            shade::curveDifferentials(attributes, ray, st.pixelSamples ? st.pixelSamples : plan.spp, st);
            return true;
        }
        const auto t = data.triangles[h.prim];
        shade::MeshAttributes attributes;
        attributes.normals = data.normals.empty() ? nullptr : data.normals.data();
        attributes.uvs = data.uvs.empty() ? nullptr : data.uvs.data();
        attributes.normalRate = data.normalRate;
        attributes.uvRate = data.uvRate;
        attributes.dpds = data.dpds.empty() ? nullptr : data.dpds.data();
        attributes.dpdt = data.dpdt.empty() ? nullptr : data.dpdt.data();
        motion(data, attributes);
        instanceInto(plan.meshes[h.geom], attributes);
        const surface::V3 a = vertexOf(h.geom, data, t.a), b = vertexOf(h.geom, data, t.b), c = vertexOf(h.geom, data, t.c);
        if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, a, b, c, h.u, h.v, h.point, ray.direction, st)) return false;
        st.geom = h.geom;
        shade::hitDifferentials(attributes, h.prim, t.a, t.b, t.c, a, b, c, ray, st.pixelSamples ? st.pixelSamples : plan.spp, st);
        return true;
    }
    // The path tracer's hit: its shading state, and its material's Bsdf (shade_material.h).
    // A curve mesh's shading attributes (its primitive attributes and curve set).
    shade::MeshAttributes curveAttributes(const cpu::MeshData& data, U32 geom) const {
        shade::MeshAttributes attributes;
        for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = data.worldToObject[k];
        attributes.prims = data.prims.empty() ? nullptr : data.prims.data();
        attributes.primData = data.primData.empty() ? nullptr : data.primData.data();
        attributes.primCount = unsigned(data.prims.size());
        attributes.curve = s.curveSets[geom];
        attributes.curve.time = time;
        attributes.time = time;
        return attributes;
    }
    // A shadow ray's hit: its material's presence alone (presence shadows). Host RAM holds every texture: no misses.
    unsigned presenceAt(const resume::Hit& h, const surface::Ray& ray, float& presence) const {
        const auto& plan = *s.plan;
        const auto& mesh = plan.meshes[h.geom];
        const auto& data = *mesh.data;
        presence = 1.0f;
        if (mesh.material >= plan.materialRecords.size()) return tex::Ready;
        shade::State st;
        if (data.points) {
            if (!points::state(pointAttributes(data), h.prim, h.normal, h.point, ray.direction, st)) return tex::Ready;
        } else if (data.curveBasis) {
            const shade::MeshAttributes attributes = curveAttributes(data, h.geom);
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return tex::Ready;
        } else {
            const auto t = data.triangles[h.prim];
            shade::MeshAttributes attributes;
            attributes.normals = data.normals.empty() ? nullptr : data.normals.data();
            attributes.uvs = data.uvs.empty() ? nullptr : data.uvs.data();
            attributes.normalRate = data.normalRate;
            attributes.uvRate = data.uvRate;
            attributes.dpds = data.dpds.empty() ? nullptr : data.dpds.data();
            attributes.dpdt = data.dpdt.empty() ? nullptr : data.dpdt.data();
            for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = data.worldToObject[k];
            attributes.prims = data.prims.empty() ? nullptr : data.prims.data();
            attributes.primData = data.primData.empty() ? nullptr : data.primData.data();
            attributes.primCount = unsigned(data.prims.size());
            motion(data, attributes);
            instanceInto(mesh, attributes);
            if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, vertexOf(h.geom, data, t.a), vertexOf(h.geom, data, t.b),
                                 vertexOf(h.geom, data, t.c), h.u, h.v, h.point, ray.direction, st))
                return tex::Ready;
        }
        st.geom = h.geom;
        st.worldToRender = plan.worldToRender;
        st.worldToObject = worldToObjectOf(mesh);
        const HostTextures textures{plan.textureImages.data(), unsigned(plan.textureImages.size())};
        return shade::materialPresence(plan.materialRecords[mesh.material], plan.materialPool.data(), st, textures, presence);
    }
    bool textureMissed() const { return false; }
    bool deferred() const { return false; }  // the CPU renders everything
    // The render outputs' AOV schema, and a mesh's geometry label among the material AOVs'.
    const aov::Tables& aovs() const { return s.aovTables; }
    int geomLabel(U32 geom) const { return s.plan->meshes[geom].geomLabel; }
    unsigned shade(const resume::Hit& h, const surface::Ray& ray, shade::State& st, shade::Bsdf& bsdf, float& presence) const {
        const auto& plan = *s.plan;
        const auto& mesh = plan.meshes[h.geom];
        const auto& data = *mesh.data;
        if (data.points) {
            if (!points::state(pointAttributes(data), h.prim, h.normal, h.point, ray.direction, st)) return tex::Invalid;
            st.geom = h.geom;
            st.worldToRender = plan.worldToRender;
            st.worldToObject = data.worldToObject;
            if (mesh.material >= plan.materialRecords.size()) return tex::Invalid;
            const HostTextures textures{plan.textureImages.data(), unsigned(plan.textureImages.size())};
            return shade::shadeMaterial(plan.materialRecords[mesh.material], plan.materialPool.data(), st, textures, bsdf, presence,
                                        scratch().dwaSlots, &scratch().glitter);
        }
        if (data.curveBasis) {
            const shade::MeshAttributes attributes = curveAttributes(data, h.geom);
            if (!shade::curveState(attributes, h.prim, h.u, h.v, h.normal, h.point, ray.direction, st)) return tex::Invalid;
            st.geom = h.geom;
            st.worldToRender = plan.worldToRender;
            st.worldToObject = data.worldToObject;
            if (mesh.material >= plan.materialRecords.size()) return tex::Invalid;
            shade::curveDifferentials(attributes, ray, st.pixelSamples ? st.pixelSamples : plan.spp, st);
            const HostTextures textures{plan.textureImages.data(), unsigned(plan.textureImages.size())};
            return shade::shadeMaterial(plan.materialRecords[mesh.material], plan.materialPool.data(), st, textures, bsdf, presence,
                                        scratch().dwaSlots, &scratch().glitter);
        }
        const auto t = data.triangles[h.prim];
        shade::MeshAttributes attributes;
        attributes.normals = data.normals.empty() ? nullptr : data.normals.data();
        attributes.uvs = data.uvs.empty() ? nullptr : data.uvs.data();
        attributes.normalRate = data.normalRate;
        attributes.uvRate = data.uvRate;
        attributes.dpds = data.dpds.empty() ? nullptr : data.dpds.data();
        attributes.dpdt = data.dpdt.empty() ? nullptr : data.dpdt.data();
        for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = data.worldToObject[k];
        attributes.prims = data.prims.empty() ? nullptr : data.prims.data();
        attributes.primData = data.primData.empty() ? nullptr : data.primData.data();
        attributes.primCount = unsigned(data.prims.size());
        attributes.polyVertices = data.polyVertices.empty() ? nullptr : data.polyVertices.data();
        attributes.positions = data.positions.data();
        attributes.toonOffset = !data.instanced && mesh.material < plan.materialRecords.size() &&
                                (plan.materialRecords[mesh.material].flags & shade::MatModoToonMaterial) != 0;
        motion(data, attributes);
        instanceInto(mesh, attributes);
        const surface::V3 a = vertexOf(h.geom, data, t.a), b = vertexOf(h.geom, data, t.b), c = vertexOf(h.geom, data, t.c);
        if (!shade::hitState(attributes, h.prim, t.a, t.b, t.c, a, b, c, h.u, h.v, h.point, ray.direction, st)) return tex::Invalid;
        st.geom = h.geom;
        st.worldToRender = plan.worldToRender;
        st.worldToObject = worldToObjectOf(mesh);
        if (mesh.material >= plan.materialRecords.size()) return tex::Invalid;
        const auto& record = plan.materialRecords[mesh.material];
        shade::hitDifferentials(attributes, h.prim, t.a, t.b, t.c, a, b, c, ray, st.pixelSamples ? st.pixelSamples : plan.spp, st);
        // Host RAM holds every level of every texture, so the CPU never misses.
        const HostTextures textures{plan.textureImages.data(), unsigned(plan.textureImages.size())};
        const unsigned shaded = shade::shadeMaterial(record, plan.materialPool.data(), st, textures, bsdf, presence, scratch().dwaSlots,
                                                     &scratch().glitter);
        if (shaded != tex::Ready && debugFailures()) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1) < 6)
                std::printf("XPU2 debug: shading %s (material %u) gave %u at st %g %g\n", data.name.c_str(), mesh.material, shaded, st.s, st.t);
        }
        return shaded;
    }
    // A diagnostic (XPU2_SESSION_DEBUG_FAILURES): the first failed shadings and texture lookups on the CPU, said.
    static bool debugFailures() {
        static const bool on = std::getenv("XPU2_SESSION_DEBUG_FAILURES") != nullptr;
        return on;
    }
    // The scene's textures for the maps (shade_mapnodes.h).
    struct HostTextures {
        const tex::Image* const* images;
        unsigned count;
        unsigned lookup(unsigned texture, tex::Binding b, const tex::Footprint& f, surface::V3& out) const {
            if (texture >= count || !images[texture]) return tex::Invalid;
            b.image = images[texture];
            b.swapCross = b.mode != 3 ? 1u : 0u;  // as XPU's vector shading looks up (displacement: in order)
            const unsigned looked = tex::lookup(b, f, out);
            if (looked != tex::Ready && debugFailures()) {
                static std::atomic<int> told{0};
                if (told.fetch_add(1) < 6)
                    std::printf("XPU2 debug: texture %u lookup gave %u at u %g v %g (dudx %g dvdx %g dudy %g dvdy %g), mode %u\n", texture,
                                looked, f.u, f.v, f.dudx, f.dvdx, f.dudy, f.dvdy, b.mode);
            }
            return looked;
        }
    };
    bool visible(surface::V3 o, surface::V3 d) const { return visible(o, d, INFINITY); }
    // Unoccluded up to distance (a shadow ray toward a light).
    bool visible(surface::V3 o, surface::V3 d, float distance) const { return visible(o, d, 0.0f, distance); }
    // Unoccluded between near and far (a shadow ray from a curve skips its epsilon).
    void countRay(U32) const {}  // the GPU profile's ray counts (XPU2_PROFILE): none on the CPU
    bool visible(surface::V3 o, surface::V3 d, float near, float distance) const {
        RTCRay ray{};
        ray.org_x = o.x;
        ray.org_y = o.y;
        ray.org_z = o.z;
        ray.dir_x = d.x;
        ray.dir_y = d.y;
        ray.dir_z = d.z;
        ray.tnear = near;
        ray.tfar = distance;
        ray.mask = 2u;  // SHADOW
        ray.time = time;
        RTCOccludedArguments args;
        rtcInitOccludedArguments(&args);
        LinkContext link;
        if (!s.plan->shadowLinks.empty()) {
            rtcInitRayQueryContext(&link.base);
            link.backend = this;
            args.context = &link.base;
            args.filter = linkFilter;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
        }
        rtcOccluded1(s.scene, &ray, &args);
        return ray.tfar >= 0;
    }
    // The path tracer's Bsdf and pending rays, one set per worker thread.
    struct Scratch {
        shade::Bsdf bsdf;
        paths::Pending pending[paths::MaxPending];
        shade::DwaParams dwaSlots[shade::DwaMaxSlots];
        shade::GlitterScratch glitter;
    };
    static Scratch& scratch() {
        static thread_local std::unique_ptr<Scratch> s;
        if (!s) s = std::make_unique<Scratch>();
        return *s;
    }
    shade::Bsdf& bsdf() const { return scratch().bsdf; }
    paths::Pending* pending() const { return scratch().pending; }
    const lights::Light& light(U32 i) const { return moved.empty() ? s.plan->lights[i] : moved[i]; }
    // Shadow linking (skipOcclusionFilter, accumulateRayPresence): the light a shadow ray goes to and the mesh it
    // leaves (its receiver), and whether a mesh shadows it: not for a light in its shadow set, nor on a receiver its
    // receiver set leaves out (the occlusion test's; presence shadows read the lights alone).
    U32 shadowLight = ~0u, shadowReceiver = ~0u;
    void shadowFor(U32 light, U32 receiver) {
        shadowLight = light;
        shadowReceiver = receiver;
    }
    bool castsShadow(U32 caster, bool receivers) const {
        const auto& plan = *s.plan;
        if (caster >= plan.meshShadowLink.size() || plan.meshShadowLink[caster] == ~0u) return true;
        const auto& link = plan.shadowLinks[plan.meshShadowLink[caster]];
        if (shadowLight != ~0u && (plan.shadowLinkBits[link.lights + (shadowLight >> 5)] >> (shadowLight & 31) & 1u)) return false;
        if (!receivers) return true;
        const int receiver = shadowReceiver < plan.meshAssignment.size() ? plan.meshAssignment[shadowReceiver] : -1;
        bool in = false;
        for (U32 k = 0; k < link.receiverCount && !in; ++k) in = plan.shadowReceivers[link.receivers + k] == receiver;
        return link.complement ? in : !in;
    }
    struct LinkContext {
        RTCRayQueryContext base;
        const CpuBackend* backend;
    };
    static void linkFilter(const RTCFilterFunctionNArguments* args) {
        if (args->valid[0] == 0) return;
        const auto* ctx = reinterpret_cast<const LinkContext*>(args->context);
        // The caster: an instance's plan mesh where the hit is in an instanced scene.
        const U32 inst = RTCHitN_instID(args->hit, args->N, 0, 0);
        const U32 geom = inst != RTC_INVALID_GEOMETRY_ID ? inst : RTCHitN_geomID(args->hit, args->N, 0);
        if (!ctx->backend->castsShadow(geom, true)) args->valid[0] = 0;
    }
    // A lobe light set's lights (the plan's lobe light sets; ~0u: every light), and the subsurface exits' set (their
    // Bssrdf's, set as a hit's subsurface is taken).
    U32 exitLightSet = ~0u;
    bool inLobeSet(U32 set, U32 i) const {
        if (set == ~0u) return true;
        const auto& plan = *s.plan;
        const U32 words = plan.lightSetWords ? plan.lightSetWords : 1u;
        return (plan.lobeLightSetBits[size_t(set) * words + (i >> 5)] >> (i & 31) & 1u) != 0;
    }
    // The light filters (light_filter_core.h), and a light's list at a mesh's shading points (null: none).
    const lights::Filter* filters() const { return s.plan->filters.data(); }
    // A cookie's map shader at (s, t) (CookieLightFilter::sampleMapShader): a state at the render origin, no normal.
    // A VDB light filter's density at p (its grid's world space), 1 without a grid (OpenVdbSampler not made).
    float filterVdb(U32 grid, U32 mode, surface::V3 p) const {
        if (grid == volume::NoGrid) return 1.0f;
        return volume::sampleFloat(volumeGrids(), grid, p, mode);
    }
    surface::V3 filterMap(U32 program, U32 reg, float st0, float st1) const {
        const auto& plan = *s.plan;
        shade::State st;
        st.P = plan.renderOrigin;
        st.Ng = st.N = st.dPds = st.dPdt = st.wo = surface::V3{0, 0, 0};
        st.s = st0;
        st.t = st1;
        st.worldToRender = plan.worldToRender;
        shade::Color regs[shade::MapMaxRegisters];
        const HostTextures textures{plan.textureImages.data(), unsigned(plan.textureImages.size())};
        if (shade::runMapProgram(plan.materialPool.data(), program, st, textures, regs) != tex::Ready) return surface::V3{1, 1, 1};
        return surface::V3{regs[reg].x, regs[reg].y, regs[reg].z};
    }
    const lights::FilterList* filterList(U32 geom, U32 light) const {
        const auto& plan = *s.plan;
        if (geom >= plan.meshFilterTable.size()) return nullptr;
        const U32 table = plan.meshFilterTable[geom];
        if (table == ~0u) return nullptr;
        const U32 list = plan.filterTables[table + light];
        return list == ~0u ? nullptr : &plan.filterLists[list];
    }
    U32 lightSet(U32 geom) const { return s.plan->meshLights[geom]; }
    const lights::TreeNode* treeNodes() const { return s.plan->lightTree.data(); }
    int treeRoot(U32 set) const { return set < s.plan->lightTreeRoots.size() ? s.plan->lightTreeRoots[set] : -1; }
    bool inSet(U32 set, U32 i) const { return (s.plan->lightSetBits[size_t(set) * s.plan->lightSetWords + (i >> 5)] >> (i & 31) & 1u) != 0; }
    // The volumes (volume_core.h), and a ray's crossings of their meshes past tnear, nearest first (Scene::
    // intersectVolumes over the whole ray): whether each enters (the ray against the mesh's geometric normal) or
    // leaves; repeats at the same distance dropped, as MoonRay's interval filter drops Embree's.
    const volume::Shader* volumes() const { return s.plan->volumes.data(); }
    volume::Grids volumeGrids() const {
        return volume::Grids{s.plan->volumeGridViews.data(), U32(s.plan->volumeGridViews.size()), s.plan->volumeMaps.run ? &s.plan->volumeMaps : nullptr,
                             time};
    }
    volume::Regions emissiveRegions() const {
        return volume::Regions{s.plan->emissiveRegions.data(), U32(s.plan->emissiveRegions.size()), s.plan->emissionPool.data()};
    }
    // MoonRay's integrator sequences for the random walk (cpu_scene.h walkSequences), null when not loaded.
    const float* walkSequences() const { return s.plan->walkSequences ? s.plan->walkSequences->data() : nullptr; }
    // Whether a ray's half-line from tnear meets a VDB volume's gate box (MeshData::gate): the slab test in index space.
    static bool gatePasses(const cpu::MeshData& m, surface::V3 o, surface::V3 d, float tnear) {
        const float* w = m.gateToIndex;
        const float oi[3] = {w[0] * o.x + w[1] * o.y + w[2] * o.z + w[3], w[4] * o.x + w[5] * o.y + w[6] * o.z + w[7],
                             w[8] * o.x + w[9] * o.y + w[10] * o.z + w[11]};
        const float di[3] = {w[0] * d.x + w[1] * d.y + w[2] * d.z, w[4] * d.x + w[5] * d.y + w[6] * d.z, w[8] * d.x + w[9] * d.y + w[10] * d.z};
        float t0 = -INFINITY, t1 = INFINITY;
        for (int a = 0; a < 3; ++a) {
            if (di[a] == 0.0f) {
                if (oi[a] < m.gateLo[a] || oi[a] > m.gateHi[a]) return false;
                continue;
            }
            float ta = (m.gateLo[a] - oi[a]) / di[a], tb = (m.gateHi[a] - oi[a]) / di[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = fmaxf(t0, ta);
            t1 = fminf(t1, tb);
        }
        return t0 <= t1 && t1 >= tnear;
    }
    U32 volumeCrossings(surface::V3 o, surface::V3 d, float tnear, volume::Crossing* x, U32 max, U32 mask) const {
        if (!s.volumes) return 0;
        U32 n = 0;
        float t = tnear;
        // The gated meshes met so far and whether the ray passed each's gate (a VDB's box: one test a ray).
        U32 gateGeom[4], gateCount = 0;
        bool gateOpen[4];
        for (U32 k = 0; k < 4 * max && n < max; ++k) {
            RTCRayHit h{};
            h.ray.org_x = o.x;
            h.ray.org_y = o.y;
            h.ray.org_z = o.z;
            h.ray.dir_x = d.x;
            h.ray.dir_y = d.y;
            h.ray.dir_z = d.z;
            h.ray.tnear = t;
            h.ray.tfar = INFINITY;
            h.ray.mask = mask;
            h.ray.time = time;
            h.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            rtcIntersect1(s.volumes, &h, &args);
            if (h.hit.geomID == RTC_INVALID_GEOMETRY_ID) break;
            const float th = h.ray.tfar;
            // A placed mesh's bounds (an instance, addVolumeInstance): its index the instance's, its normal through the
            // instance's normal transform, and its gate met by the ray in its data's space.
            const U32 geom = h.hit.instID[0] != RTC_INVALID_GEOMETRY_ID ? h.hit.instID[0] : h.hit.geomID;
            const cpu::Mesh& entry = s.plan->meshes[geom];
            const bool placed = h.hit.instID[0] != RTC_INVALID_GEOMETRY_ID && entry.instance;
            if (placed) {
                const float* n = entry.normalToWorld;
                const float nx = h.hit.Ng_x, ny = h.hit.Ng_y, nz = h.hit.Ng_z;
                h.hit.Ng_x = nx * n[0] + ny * n[3] + nz * n[6];
                h.hit.Ng_y = nx * n[1] + ny * n[4] + nz * n[7];
                h.hit.Ng_z = nx * n[2] + ny * n[5] + nz * n[8];
            }
            const cpu::MeshData& md = *entry.data;
            if (md.gate) {
                U32 g = 0;
                while (g < gateCount && gateGeom[g] != geom) ++g;
                bool open;
                if (g < gateCount) open = gateOpen[g];
                else {
                    if (placed) {
                        // The ray in the data's space: (p - t) through the transform's 3x3 inverse (normalToWorld's
                        // transpose).
                        const float* x = entry.objectToWorld;
                        const float* n = entry.normalToWorld;
                        const surface::V3 q{o.x - x[9], o.y - x[10], o.z - x[11]};
                        auto back = [&](surface::V3 v) {
                            return surface::V3{v.x * n[0] + v.y * n[1] + v.z * n[2], v.x * n[3] + v.y * n[4] + v.z * n[5], v.x * n[6] + v.y * n[7] + v.z * n[8]};
                        };
                        open = gatePasses(md, back(q), back(d), tnear);
                    } else
                        open = gatePasses(md, o, d, tnear);
                    if (gateCount < 4) {
                        gateGeom[gateCount] = geom;
                        gateOpen[gateCount++] = open;
                    }
                }
                if (!open) {
                    t = th + fmaxf(1e-6f, th * 1e-6f);
                    continue;
                }
            }
            const U32 v = U32(entry.volume);
            const U32 flags = d.x * h.hit.Ng_x + d.y * h.hit.Ng_y + d.z * h.hit.Ng_z < 0.0f ? volume::Entry : volume::Exit;
            if (!(n > 0 && x[n - 1].t == th && x[n - 1].volume == v && x[n - 1].flags == flags)) x[n++] = volume::Crossing{th, v, flags};
            t = th + fmaxf(1e-6f, th * 1e-6f);
        }
        return n;
    }
};

// The film's math filters (Film.cc addFilteredAovSampleBundleHandler, atomicMathFilter): a value into an entry's
// accumulator. Averages and sums add; min and max keep the least and greatest; the closest keeps the nearer value. The
// force-consistent-sampling filter keeps nothing, as MoonRay's vector film's switch has no case for it.
template <class T>
inline void mergeAov(const aov::Entry& e, double* dst, float* dstDepth, const T* src, float srcDepth) {
    switch (e.filter) {
    case aov::FilterAvg:
    case aov::FilterSum:
        for (U32 c = 0; c < e.channels; ++c) dst[c] += double(src[c]);
        break;
    case aov::FilterMin:
        for (U32 c = 0; c < e.channels; ++c)
            if (double(src[c]) < dst[c]) dst[c] = double(src[c]);
        break;
    case aov::FilterMax:
        for (U32 c = 0; c < e.channels; ++c)
            if (double(src[c]) > dst[c]) dst[c] = double(src[c]);
        break;
    case aov::FilterClosest:
        if (srcDepth < *dstDepth) {
            *dstDepth = srcDepth;
            for (U32 c = 0; c < e.channels; ++c) dst[c] = double(src[c]);
        }
        break;
    default: break;
    }
}
// An accumulator's initial value (the film's cleared AOV buffers).
inline double aovInitial(const aov::Entry& e) {
    return e.filter == aov::FilterMin ? double(aov::infinity()) : e.filter == aov::FilterMax ? -double(aov::infinity()) : 0.0;
}
// A pixel's accumulators: initialised for the schema.
inline void initAovPixel(const aov::Tables& t, double* acc, float* depths) {
    for (U32 i = 0; i < t.count; ++i)
        for (U32 c = 0; c < t.entries[i].channels; ++c) acc[t.entries[i].offset + c] = aovInitial(t.entries[i]);
    for (U32 i = 0; i < t.closestCount; ++i) depths[i] = aov::infinity();
}
// One sample's (or tile's) AOV values into a pixel's accumulators.
template <class T>
inline void mergeAovPixel(const aov::Tables& t, double* acc, float* accDepths, const T* values, const float* depths) {
    for (U32 i = 0; i < t.count; ++i) {
        const aov::Entry& e = t.entries[i];
        mergeAov(e, acc + e.offset, e.closest != aov::None ? accDepths + e.closest : nullptr, values + e.offset,
                 e.closest != aov::None ? depths[e.closest] : 0.0f);
    }
}

// Per-pixel sums and sample counts in double precision (ACC-06). A chunk covers
// one tile, so a lock per tile serialises only workers on the same tile. Sums of
// the odd-index samples are kept beside the totals, for the adaptive error
// measure (MoonRay's Beauty Odd buffer). The AOVs' accumulators and each pixel's
// render time beside them, where the outputs ask for them.
class HostFilm {
public:
    HostFilm(U32 width, U32 height)
        : width_(width), height_(height), sums_(size_t(width) * height * 4, 0.0), odd_(size_t(width) * height * 4, 0.0),
          counts_(size_t(width) * height, 0),
          locks_(((width + TileSize - 1) / TileSize) * ((height + TileSize - 1) / TileSize)) {}
    // The AOV schema the film accumulates (none: empty), and whether it times its pixels.
    void initAovs(const aov::Tables& t, bool heat) {
        aovTables_ = t;
        const size_t pixels = size_t(width_) * height_;
        aov_.assign(pixels * t.channels, 0.0);
        aovDepth_.assign(pixels * t.closestCount, 0.0f);
        for (size_t p = 0; p < pixels; ++p) initAovPixel(t, aov_.data() + p * t.channels, aovDepth_.data() + p * t.closestCount);
        if (heat) heat_.assign(pixels, 0.0);
    }
    const aov::Tables& aovTables() const { return aovTables_; }
    // Cryptomatte's fragments (aov_crypto.h), when an output asks for them.
    void initCrypto(const aov::Plan* plan) {
        if (plan && plan->cryptomatte) crypto_.init(width_, height_, plan->cryptoCategories, plan->cryptoMultiPresence);
    }
    aov::CryptoBuffer& crypto() { return crypto_; }
    // A sample's events into its pixel's fragments: those that reach the film.
    void addCrypto(U32 pixel, const aov::CryptoEvent& e) {
        if (!crypto_.empty() && e.used && (e.increment || e.credit)) crypto_.add(pixel, e);
    }
    void add(U32 tile, const U32* pixels, const double (*sums)[4], const double (*odd)[4], const U32* counts, const double* aovs = nullptr,
             const float* aovDepths = nullptr, const double* heat = nullptr,
             const std::vector<std::pair<U32, aov::CryptoEvent>>* cryptoEvents = nullptr) {
        auto& lock = locks_[tile].flag;
        while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
        const U32 C = aovTables_.channels, K = aovTables_.closestCount;
        for (U32 i = 0; i < TilePixels; ++i) {
            if (pixels[i] == ~0u || !counts[i]) continue;
            for (int c = 0; c < 4; ++c) {
                sums_[size_t(pixels[i]) * 4 + c] += sums[i][c];
                odd_[size_t(pixels[i]) * 4 + c] += odd[i][c];
            }
            counts_[pixels[i]] += counts[i];
            if (aovs && C)
                mergeAovPixel(aovTables_, aov_.data() + size_t(pixels[i]) * C, aovDepth_.data() + size_t(pixels[i]) * K, aovs + size_t(i) * C,
                              aovDepths + size_t(i) * K);
            if (heat && !heat_.empty()) heat_[pixels[i]] += heat[i];
        }
        if (cryptoEvents)
            for (const auto& pe : *cryptoEvents) addCrypto(pe.first, pe.second);
        lock.clear(std::memory_order_release);
    }
    // Partial shading: the GPU's provisional samples rendered again in full here, whose values leave the GPU's
    // provisional film (GpuLane::provSums, provCounts) when the two are put together.
    void initSettled() {
        settled_.assign(sums_.size(), 0.0);
        settledCounts_.assign(counts_.size(), 0u);
    }
    void settle(U32 pixel, const float* rgba) {
        const U32 tilesX = (width_ + TileSize - 1) / TileSize;
        auto& lock = locks_[(pixel / width_ / TileSize) * tilesX + (pixel % width_) / TileSize].flag;
        while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
        for (int c = 0; c < 4; ++c) settled_[size_t(pixel) * 4 + c] += rgba[c];
        ++settledCounts_[pixel];
        lock.clear(std::memory_order_release);
    }
    // A pixel's provisional samples still waiting, given the GPU's provisional film: their count, and their sums
    // added to sum (none settled: the film as it is). A download older than a settling here can count fewer than
    // were settled: none, then.
    U32 provisional(size_t pixel, const double* provSums, const U32* provCounts, double* sum) const {
        if (!provCounts) return 0;
        const U32 settled = settledCounts_.empty() ? 0u : settledCounts_[pixel];
        if (provCounts[pixel] <= settled) return 0;
        for (int c = 0; c < 4; ++c) sum[c] += provSums[pixel * 4 + c] - (settled_.empty() ? 0.0 : settled_[pixel * 4 + c]);
        return provCounts[pixel] - settled;
    }
    void addOne(U32 pixel, U32 index, const Radiance& r, double seconds = 0.0) {
        const U32 tilesX = (width_ + TileSize - 1) / TileSize;
        const U32 tile = (pixel / width_ / TileSize) * tilesX + (pixel % width_) / TileSize;
        auto& lock = locks_[tile].flag;
        while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
        const double v[4] = {r.rgb[0], r.rgb[1], r.rgb[2], r.alpha};
        for (int c = 0; c < 4; ++c) {
            sums_[size_t(pixel) * 4 + c] += v[c];
            if (index & 1) odd_[size_t(pixel) * 4 + c] += v[c];
        }
        ++counts_[pixel];
        if (r.aov && aovTables_.channels)
            mergeAovPixel(aovTables_, aov_.data() + size_t(pixel) * aovTables_.channels, aovDepth_.data() + size_t(pixel) * aovTables_.closestCount,
                          r.aov, r.aovDepth);
        if (!heat_.empty()) heat_[pixel] += seconds;
        for (U32 k = 0; r.crypto && k < r.cryptoCount; ++k) addCrypto(pixel, r.crypto[k]);
        lock.clear(std::memory_order_release);
    }
    // A copy that is consistent tile by tile, taken while workers keep adding.
    void snapshot(std::vector<double>& sums, std::vector<double>& odd, std::vector<U32>& counts) {
        sums.resize(sums_.size());
        odd.resize(odd_.size());
        counts.resize(counts_.size());
        for (U32 tile = 0; tile < locks_.size(); ++tile) {
            auto& lock = locks_[tile].flag;
            while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
            for (U32 p = 0; p < TilePixels; ++p) {
                const U32 pixel = tilePixel(tile, p, width_, height_);
                if (pixel == ~0u) continue;
                for (int c = 0; c < 4; ++c) {
                    sums[size_t(pixel) * 4 + c] = sums_[size_t(pixel) * 4 + c];
                    odd[size_t(pixel) * 4 + c] = odd_[size_t(pixel) * 4 + c];
                }
                counts[pixel] = counts_[pixel];
            }
            lock.clear(std::memory_order_release);
        }
    }
    // The picture so far, RGB: each pixel's mean over its samples here and in a copy of the GPU's film (gpuSums
    // and gpuCounts, or none) and its provisional film (provSums and provCounts, or none), black where none has
    // landed. Consistent tile by tile, taken while workers keep adding. Returns the samples it saw.
    // covered (or none): whether each pixel has a sample.
    U64 preview(float* rgb, const double* gpuSums, const U32* gpuCounts, const double* provSums = nullptr, const U32* provCounts = nullptr,
                unsigned char* covered = nullptr) {
        U64 seen = 0;
        for (U32 tile = 0; tile < locks_.size(); ++tile) {
            auto& lock = locks_[tile].flag;
            while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
            for (U32 p = 0; p < TilePixels; ++p) {
                const U32 pixel = tilePixel(tile, p, width_, height_);
                if (pixel == ~0u) continue;
                double extra[4] = {0, 0, 0, 0};
                const U32 count = counts_[pixel] + (gpuCounts ? gpuCounts[pixel] : 0u) + provisional(pixel, provSums, provCounts, extra);
                seen += count;
                if (covered) covered[pixel] = count ? 1 : 0;
                for (int c = 0; c < 3; ++c)
                    rgb[size_t(pixel) * 3 + c] =
                        count ? float((sums_[size_t(pixel) * 4 + c] + (gpuSums ? gpuSums[size_t(pixel) * 4 + c] : 0.0) + extra[c]) / count) : 0.0f;
            }
            lock.clear(std::memory_order_release);
        }
        return seen;
    }
    // A copy of the film, consistent tile by tile, for a preview of an AOV: sums, counts, odd sums, the AOV
    // accumulators and closest depths, and the seconds each pixel took.
    void snapshot(std::vector<double>& sums, std::vector<U32>& counts, std::vector<double>& odd, std::vector<double>& aov,
                  std::vector<float>& depths, std::vector<double>& heat) {
        sums.resize(sums_.size());
        counts.resize(counts_.size());
        odd.resize(odd_.size());
        aov.resize(aov_.size());
        depths.resize(aovDepth_.size());
        heat.resize(heat_.size());
        const U32 C = aovTables_.channels, K = aovTables_.closestCount;
        for (U32 tile = 0; tile < locks_.size(); ++tile) {
            auto& lock = locks_[tile].flag;
            while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
            for (U32 p = 0; p < TilePixels; ++p) {
                const U32 pixel = tilePixel(tile, p, width_, height_);
                if (pixel == ~0u) continue;
                for (int c = 0; c < 4; ++c) {
                    sums[size_t(pixel) * 4 + c] = sums_[size_t(pixel) * 4 + c];
                    odd[size_t(pixel) * 4 + c] = odd_[size_t(pixel) * 4 + c];
                }
                counts[pixel] = counts_[pixel];
                for (U32 c = 0; c < C; ++c) aov[size_t(pixel) * C + c] = aov_[size_t(pixel) * C + c];
                for (U32 k = 0; k < K; ++k) depths[size_t(pixel) * K + k] = aovDepth_[size_t(pixel) * K + k];
                if (!heat_.empty()) heat[pixel] = heat_[pixel];
            }
            lock.clear(std::memory_order_release);
        }
    }
    // Cryptomatte's fragments so far, consistent tile by tile, taken while workers keep adding (a checkpoint's).
    void snapshotCrypto(aov::CryptoBuffer& out) {
        out.like(crypto_);
        if (crypto_.empty()) return;
        for (U32 tile = 0; tile < locks_.size(); ++tile) {
            auto& lock = locks_[tile].flag;
            while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
            for (U32 p = 0; p < TilePixels; ++p) {
                const U32 pixel = tilePixel(tile, p, width_, height_);
                if (pixel != ~0u) out.copyPixel(crypto_, pixel);
            }
            lock.clear(std::memory_order_release);
        }
    }
    const std::vector<double>& sums() const { return sums_; }
    const std::vector<double>& odd() const { return odd_; }
    const std::vector<U32>& counts() const { return counts_; }
    const std::vector<double>& aovs() const { return aov_; }
    const std::vector<float>& aovDepths() const { return aovDepth_; }
    const std::vector<double>& heat() const { return heat_; }

private:
    struct Lock { std::atomic_flag flag = ATOMIC_FLAG_INIT; };
    U32 width_, height_;
    std::vector<double> sums_, odd_;
    std::vector<U32> counts_;
    std::vector<Lock> locks_;
    aov::Tables aovTables_;
    std::vector<double> aov_;
    std::vector<float> aovDepth_;
    std::vector<double> heat_;
    aov::CryptoBuffer crypto_;
    std::vector<double> settled_;  // partial shading: provisional samples settled here (initSettled)
    std::vector<U32> settledCounts_;
};

// An adaptive frame's pass tables, host side: for each pass slot, the active-pixel
// mask of every tile and the sample-index base of every pixel; and the samples
// each pass has completed on the CPU.
struct PassTables {
    std::vector<std::uint64_t> masks[PassSlots];
    std::vector<U32> bases[PassSlots];
    std::atomic<U64> cpuDone[MaxPasses] = {};
};

// One sample handed back by the GPU after a texture miss (spec MEM-05, M1).
struct Retry {
    U32 pixel, index;
    U32 pass = 0;  // an adaptive frame's pass, which counts it done
};

// Samples migrated to the CPU lane. They are rendered before new chunks, since
// they were started first.
class RetryQueue {
public:
    void push(const Retry* items, size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        items_.insert(items_.end(), items, items + n);
        size_.store(items_.size(), std::memory_order_release);
    }
    size_t pop(Retry* out, size_t n) {
        if (!size_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t got = std::min(n, items_.size());
        std::copy_n(items_.begin(), got, out);
        items_.erase(items_.begin(), items_.begin() + got);
        size_.store(items_.size(), std::memory_order_release);
        return got;
    }
    size_t size() const { return size_.load(std::memory_order_acquire); }

private:
    std::mutex mutex_;
    std::deque<Retry> items_;
    std::atomic<size_t> size_{0};
};

// Partial shading's provisional samples waiting to be rendered again in full: the CPU lane takes them oldest first,
// the GPU's feeder those whose fine-detail tile has landed (take), whichever comes to one first.
class RefineQueue {
public:
    void push(const Refine* items, size_t n) {
        if (!n) return;
        std::lock_guard<std::mutex> lock(mutex_);
        items_.insert(items_.end(), items, items + n);
        size_.store(items_.size(), std::memory_order_release);
    }
    size_t pop(Refine* out, size_t n) {
        if (!size_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t got = std::min(n, items_.size());
        std::copy_n(items_.begin(), got, out);
        items_.erase(items_.begin(), items_.begin() + std::ptrdiff_t(got));
        size_.store(items_.size(), std::memory_order_release);
        return got;
    }
    // Up to n of the oldest `scan` for which ready(tile) holds, oldest first (ready null: any).
    template <class F>
    size_t take(F ready, Refine* out, size_t n, size_t scan) {
        if (!size_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t got = 0, kept = 0, k = 0;
        const size_t end = std::min(scan, items_.size());
        for (; k < end && got < n; ++k) {
            if (ready(items_[k].tile)) out[got++] = items_[k];
            else items_[kept++] = items_[k];
        }
        // The ones passed over close up behind those not looked at.
        if (got) {
            std::move(items_.begin() + std::ptrdiff_t(k), items_.end(), items_.begin() + std::ptrdiff_t(kept));
            items_.resize(items_.size() - got);
        }
        size_.store(items_.size(), std::memory_order_release);
        return got;
    }
    size_t size() const { return size_.load(std::memory_order_acquire); }

private:
    std::mutex mutex_;
    std::deque<Refine> items_;
    std::atomic<size_t> size_{0};
};

struct CpuCounters {
    std::atomic<U64> samples{0}, chunks{0}, retried{0}, steals{0}, moves{0}, idleSpins{0}, refined{0};
    std::atomic<U32> failures{0};
    // Telemetry (TEL-01), microseconds from the lane's start.
    std::atomic<U64> busyMicros{0}, slowestChunkMicros{0}, firstClaimMicros{~0ull}, lastClaimMicros{0};
};

// Renders one claimed chunk on the CPU into the host film and returns the samples
// rendered. With pass tables (adaptive frames) only the tile's active pixels are
// rendered, each from its own sample-index base.
inline U64 renderChunk(const CpuScene& scene, const mixed::Config& config, const Chunk& chunk, HostFilm& film,
                       CpuCounters& counters, PassTables* tables = nullptr) {
    CpuBackend backend{scene};
    U32 pixels[TilePixels], counts[TilePixels] = {}, base[TilePixels] = {};
    double sums[TilePixels][4] = {}, odd[TilePixels][4] = {};
    // The tile's AOV accumulators and each sample's values (aov_core.h), and each pixel's render time.
    const aov::Tables& at = film.aovTables();
    const U32 C = at.channels, K = at.closestCount;
    thread_local std::vector<double> tileAov;
    thread_local std::vector<float> tileDepth, sampleAov, sampleDepth, sampleBackup;
    // Cryptomatte: a sample's events, and the tile's that reach the film.
    thread_local std::vector<aov::CryptoEvent> sampleCrypto;
    thread_local std::vector<std::pair<U32, aov::CryptoEvent>> tileCrypto;
    const bool crypto = !film.crypto().empty();
    if (crypto) sampleCrypto.resize(aov::CryptoMaxEvents);
    tileCrypto.clear();
    double heat[TilePixels] = {};
    const bool timing = !film.heat().empty();
    if (C) {
        tileAov.resize(size_t(TilePixels) * C);
        tileDepth.resize(size_t(TilePixels) * K);
        sampleAov.resize(C);
        sampleDepth.resize(K);
        sampleBackup.resize(C);
        for (U32 p = 0; p < TilePixels; ++p) initAovPixel(at, tileAov.data() + size_t(p) * C, tileDepth.data() + size_t(p) * K);
    }
    U32 first = chunk.first;
    std::uint64_t mask = ~0ull;
    if (tables) {
        const U32 slot = chunkPass(chunk) % PassSlots;
        mask = tables->masks[slot][chunk.tile];
        first = chunkOffset(chunk);
    }
    for (U32 p = 0; p < TilePixels; ++p) {
        pixels[p] = (mask >> p & 1) ? tilePixel(chunk.tile, p, config.width, config.height, config.viewport) : ~0u;
        if (tables && pixels[p] != ~0u) base[p] = tables->bases[chunkPass(chunk) % PassSlots][pixels[p]];
    }
    U64 skipped = 0;
    for (U32 s = 0; s < chunk.count; ++s)
        for (U32 p = 0; p < TilePixels; ++p) {
            if (pixels[p] == ~0u) continue;
            Radiance r;
            if (C) {
                r.aov = sampleAov.data();
                r.aovDepth = sampleDepth.data();
                r.aovBackup = sampleBackup.data();
            }
            if (crypto) {
                r.crypto = sampleCrypto.data();
                r.cryptoCapacity = aov::CryptoMaxEvents;
            }
            const U32 index = base[p] + first + s;
            const auto began = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const SampleStatus status = integrateCpu(config, pixels[p], index, backend, r);
            if (status == SampleStatus::Skip) {  // no camera ray: taken, nothing added
                ++skipped;
                continue;
            }
            if (status != SampleStatus::Done) {
                counters.failures.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (timing) heat[p] += std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            const double v[4] = {r.rgb[0], r.rgb[1], r.rgb[2], r.alpha};
            for (int c = 0; c < 4; ++c) {
                sums[p][c] += v[c];
                if (index & 1) odd[p][c] += v[c];
            }
            if (C) mergeAovPixel(at, tileAov.data() + size_t(p) * C, tileDepth.data() + size_t(p) * K, sampleAov.data(), sampleDepth.data());
            for (U32 k = 0; k < r.cryptoCount; ++k)
                if (r.crypto[k].used && (r.crypto[k].increment || r.crypto[k].credit)) tileCrypto.push_back({pixels[p], r.crypto[k]});
            ++counts[p];
        }
    film.add(chunk.tile, pixels, sums, odd, counts, C ? tileAov.data() : nullptr, C ? tileDepth.data() : nullptr, timing ? heat : nullptr,
             crypto ? &tileCrypto : nullptr);
    U64 done = skipped;
    for (U32 p = 0; p < TilePixels; ++p) done += counts[p];
    counters.samples.fetch_add(done, std::memory_order_relaxed);
    counters.chunks.fetch_add(1, std::memory_order_relaxed);
    if (tables) tables->cpuDone[chunkPass(chunk)].fetch_add(done, std::memory_order_relaxed);
    return done;
}

// CPU lane workers. They persist across frames (created and pinned once) and
// wait between frames. In a frame each moves through the grid by the cellular
// rules and claims one chunk at a time. A worker counts as idle until it has
// claimed work, so a worker that wakes late never holds a frame open; an idle
// worker holds no work, so a frame may end without waiting for workers to sleep.
class CpuLane {
public:
    // affinity: the logical processors the workers may run on (0 for any).
    CpuLane(const CpuScene& scene, unsigned threads, DWORD_PTR affinity = 0)
        : scene_(scene), active_(new std::atomic<U32>[std::max(1u, threads)]) {
        for (unsigned w = 0; w < std::max(1u, threads); ++w) active_[w].store(~0u, std::memory_order_relaxed);
        workers_.reserve(threads);
        for (unsigned w = 0; w < threads; ++w)
            workers_.emplace_back([this, w, threads, affinity] {
                if (affinity) SetThreadAffinityMask(GetCurrentThread(), affinity);
                loop(w, threads);
            });
    }
    ~CpuLane() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            closing_.store(true, std::memory_order_release);
        }
        wake_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }
    CpuLane(const CpuLane&) = delete;

    // Starts a frame. Workers from the last frame are parked first.
    void begin(const mixed::Config& config, Grid& grid, HostFilm& film, RetryQueue& retries, PassTables* tables = nullptr,
               RefineQueue* refines = nullptr) {
        while (parked_.load(std::memory_order_acquire) != workers_.size()) std::this_thread::yield();
        counters.samples = counters.chunks = counters.retried = counters.steals = counters.moves = counters.idleSpins = 0;
        counters.refined = 0;
        counters.failures = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failure_.clear();
        }
        counters.busyMicros = counters.slowestChunkMicros = counters.lastClaimMicros = 0;
        counters.firstClaimMicros = ~0ull;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            config_ = config;
            grid_ = &grid;
            film_ = &film;
            retries_ = &retries;
            refines_ = refines;
            tables_ = tables;
            idle_.store(unsigned(workers_.size()), std::memory_order_release);
            closing_.store(false, std::memory_order_release);
            cancelled_.store(false, std::memory_order_release);
            started_ = std::chrono::steady_clock::now();
            parked_.store(0, std::memory_order_release);
            ++frame_;
        }
        wake_.notify_all();
    }
    // Ends the frame once its work is gone, and wakes idle workers to park.
    void finish() {
        {
            std::lock_guard<std::mutex> lock(idleMutex_);
            closing_.store(true, std::memory_order_release);
        }
        idleWake_.notify_all();
    }
    // Returns once every worker has left the frame: its grid and film may go.
    void waitParked() const {
        while (parked_.load(std::memory_order_acquire) != workers_.size()) std::this_thread::yield();
    }
    // What failed on a worker this frame, where it said.
    std::string failure() {
        std::lock_guard<std::mutex> lock(mutex_);
        return failure_;
    }
    void cancel() {
        cancelled_.store(true, std::memory_order_release);
        finish();
    }
    unsigned idle() const { return idle_.load(std::memory_order_acquire); }
    unsigned threads() const { return unsigned(workers_.size()); }
    // The tile worker w is rendering (~0u: none), for a host's outlines of the work; read from any thread.
    U32 activeTile(unsigned w) const { return active_[w].load(std::memory_order_relaxed); }
    CpuCounters counters;

private:
    void loop(unsigned w, unsigned threads) {
        U64 seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                parked_.fetch_add(1, std::memory_order_acq_rel);
                wake_.wait(lock, [&] { return stop_ || frame_ != seen; });
                if (stop_) return;
                seen = frame_;
            }
            // A failure on a worker (a texture tile that cannot be read) is the frame's, with its reason: out of the thread it
            // ended the process.
            try {
                run(w, threads);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (failure_.empty()) failure_ = e.what();
                counters.failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    void run(unsigned w, unsigned threads) {
        Grid& grid = *grid_;
        HostFilm& film = *film_;
        RetryQueue& retries = *retries_;
        U32 cell = grid.seed(Device::Cpu, w, threads);
        Retry retry[64];
        bool idle = true;  // counted idle until it claims work
        auto setIdle = [&](bool v) {
            if (v != idle) {
                idle = v;
                idle_.fetch_add(v ? 1 : unsigned(-1), std::memory_order_acq_rel);
            }
        };
        while (!cancelled_.load(std::memory_order_acquire)) {
            if (retries.size()) {
                setIdle(false);
                if (size_t n = retries.pop(retry, 64)) {
                    CpuBackend backend{scene_};
                    for (size_t i = 0; i < n; ++i) {
                        // An adaptive frame counts the sample in its pass, rendered or not (as the GPU counts its own).
                        if (tables_) tables_->cpuDone[retry[i].pass].fetch_add(1, std::memory_order_relaxed);
                        Radiance r;
                        thread_local std::vector<float> retryAov, retryDepth, retryBackup;
                        thread_local std::vector<aov::CryptoEvent> retryCrypto;
                        if (!film.crypto().empty()) {
                            retryCrypto.resize(aov::CryptoMaxEvents);
                            r.crypto = retryCrypto.data();
                            r.cryptoCapacity = aov::CryptoMaxEvents;
                        }
                        if (film.aovTables().channels) {
                            retryAov.resize(film.aovTables().channels);
                            retryDepth.resize(film.aovTables().closestCount);
                            retryBackup.resize(film.aovTables().channels);
                            r.aov = retryAov.data();
                            r.aovDepth = retryDepth.data();
                            r.aovBackup = retryBackup.data();
                        }
                        const SampleStatus status = integrateCpu(config_, retry[i].pixel, retry[i].index, backend, r);
                        if (status == SampleStatus::Skip) continue;
                        if (status != SampleStatus::Done) {
                            counters.failures.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        film.addOne(retry[i].pixel, retry[i].index, r);
                    }
                    counters.samples.fetch_add(n, std::memory_order_relaxed);
                    counters.retried.fetch_add(n, std::memory_order_relaxed);
                }
                continue;
            }
            if (refines_ && refines_->size()) {
                // The GPU's provisional samples, rendered in full: into the film, their provisional values out of the
                // GPU's (HostFilm::settle), counted in their passes now.
                setIdle(false);
                Refine refine[16];
                if (size_t n = refines_->pop(refine, 16)) {
                    CpuBackend backend{scene_};
                    for (size_t i = 0; i < n; ++i) {
                        if (tables_) tables_->cpuDone[refine[i].pass].fetch_add(1, std::memory_order_relaxed);
                        Radiance r;
                        thread_local std::vector<float> refineAov, refineDepth, refineBackup;
                        thread_local std::vector<aov::CryptoEvent> refineCrypto;
                        if (!film.crypto().empty()) {
                            refineCrypto.resize(aov::CryptoMaxEvents);
                            r.crypto = refineCrypto.data();
                            r.cryptoCapacity = aov::CryptoMaxEvents;
                        }
                        if (film.aovTables().channels) {
                            refineAov.resize(film.aovTables().channels);
                            refineDepth.resize(film.aovTables().closestCount);
                            refineBackup.resize(film.aovTables().channels);
                            r.aov = refineAov.data();
                            r.aovDepth = refineDepth.data();
                            r.aovBackup = refineBackup.data();
                        }
                        const SampleStatus status = integrateCpu(config_, refine[i].pixel, refine[i].index, backend, r);
                        film.settle(refine[i].pixel, refine[i].rgba);
                        if (status == SampleStatus::Skip) continue;
                        if (status != SampleStatus::Done) {
                            counters.failures.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        film.addOne(refine[i].pixel, refine[i].index, r);
                    }
                    counters.samples.fetch_add(n, std::memory_order_relaxed);
                    counters.refined.fetch_add(n, std::memory_order_relaxed);
                }
                continue;
            }
            const U32 next = grid.choose(Device::Cpu, cell, w);
            if (next == ~0u) {
                setIdle(true);
                if (closing_.load(std::memory_order_acquire)) break;
                // No unclaimed work anywhere: only migrated samples could still come.
                counters.idleSpins.fetch_add(1, std::memory_order_relaxed);
                std::unique_lock<std::mutex> lock(idleMutex_);
                idleWake_.wait_for(lock, std::chrono::milliseconds(1), [&] {
                    return closing_.load(std::memory_order_acquire) || retries.size() || (refines_ && refines_->size());
                });
                continue;
            }
            if (next != cell) {
                counters.moves.fetch_add(1, std::memory_order_relaxed);
                if (grid.owner(next) != Device::Cpu) counters.steals.fetch_add(1, std::memory_order_relaxed);
                cell = next;
            }
            // Busy before claiming: the coordinator must never see an empty grid
            // and every worker idle while a claimed chunk is still unrendered.
            setIdle(false);
            Chunk chunk;
            if (!grid.claim(cell, 1, &chunk, Device::Cpu)) continue;
            const auto begun = std::chrono::steady_clock::now();
            const U64 claimed = U64(std::chrono::duration_cast<std::chrono::microseconds>(begun - started_).count());
            for (U64 v = counters.firstClaimMicros.load(); claimed < v && !counters.firstClaimMicros.compare_exchange_weak(v, claimed);) {}
            for (U64 v = counters.lastClaimMicros.load(); claimed > v && !counters.lastClaimMicros.compare_exchange_weak(v, claimed);) {}
            active_[w].store(chunk.tile, std::memory_order_relaxed);
            const U64 samples = renderChunk(scene_, config_, chunk, film, counters, tables_);
            active_[w].store(~0u, std::memory_order_relaxed);
            const U64 took = U64(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begun).count());
            counters.busyMicros.fetch_add(took, std::memory_order_relaxed);
            for (U64 v = counters.slowestChunkMicros.load(); took > v && !counters.slowestChunkMicros.compare_exchange_weak(v, took);) {}
            grid.rendered(chunk.cell, Device::Cpu, samples);
        }
        setIdle(true);
    }

    const CpuScene& scene_;
    std::unique_ptr<std::atomic<U32>[]> active_;  // each worker's tile (activeTile)
    std::vector<std::thread> workers_;
    std::string failure_;
    std::mutex mutex_, idleMutex_;
    std::condition_variable wake_, idleWake_;
    bool stop_ = false;
    U64 frame_ = 0;
    mixed::Config config_{};
    Grid* grid_ = nullptr;
    HostFilm* film_ = nullptr;
    RetryQueue* retries_ = nullptr;
    RefineQueue* refines_ = nullptr;
    PassTables* tables_ = nullptr;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    std::atomic<bool> closing_{false}, cancelled_{false};
    std::atomic<unsigned> idle_{0}, parked_{0};
};

// integrateCpu: its definition
#if !defined(XPU2_SPLIT_UNITS) || defined(XPU2_CPU_UNIT)
#if defined(XPU2_SPLIT_UNITS)
SampleStatus integrateCpu(const mixed::Config& c, U32 pixel, U32 index, CpuBackend& backend, Radiance& out) {
#else
inline SampleStatus integrateCpu(const mixed::Config& c, U32 pixel, U32 index, CpuBackend& backend, Radiance& out) {
#endif
    return integrate(c, pixel, index, backend, out);
}
#endif
}  // namespace xpu2::interactive::async

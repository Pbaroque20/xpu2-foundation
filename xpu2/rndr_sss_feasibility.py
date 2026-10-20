"""Read-only capture for the "find on the GPU, confirm on the CPU" feasibility test.

Scene::intersectRay counts scalar rays by origin (subsurface probe, back-ray,
random walk, other subsurface rays such as the scalar bounces, or outside
subsurface), times a sample of Embree calls, re-intersects Embree's hit
triangle alone on the same Embree device, and sends the same sampled rays to
XPU's GPU accelerator in batches. No ray result is changed.
"""
from .native_gpu_paths import once

FILES = ('pbr/core/Scene.cc', 'pbr/integrator/PathIntegratorUtil.cc', 'pbr/integrator/PathIntegratorSubsurface.cc')
HEADERS = ('sss_probe.h',)
# Quoted includes inside rt/ resolve beside the including header, so the whole
# folder is shadowed; only EmbreeAccelerator.h changes.
HEADER_FILES = {'rt/' + h: 'moonray/rendering/rt/' + h for h in (
    'AcceleratorUtils.h', 'EmbreeAccelerator.h', 'GeomContext.h', 'GeometryManager.h',
    'GeometryManagerExecTracker.h', 'IntersectionFilters.h', 'rt.h', 'gpu/GPUAccelerator.h', 'gpu/GPURay.h')}

CHECK = r'''
#include "sss_probe.h"
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/NamedPrimitive.h>
#include <moonray/rendering/shading/RootShader.h>
#include <moonray/rendering/rt/gpu/GPUAccelerator.h>
#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cmath>

namespace {
void sssFlush(moonray::mcrt_common::ThreadLocalState* tls, moonray::pbr::TLState* pbrTls)
{
    auto& s = sssprobe::slot();
    if (s.rays.empty()) return;
    auto* accel = const_cast<moonray::rt::GPUAccelerator*>(pbrTls->mFs->mGPUAccel);
    // [0] first 512-ray call after the idle gap; [1] steady 1-ray calls; [2] steady 512-ray calls
    // (calls 6-10 of ten back to back); [3] the full batch, which also produces the results below.
    auto timed = [&](uint32_t n) { const uint64_t t0 = sssprobe::now(); accel->intersect(tls->mThreadIdx, n, s.rays.data()); return sssprobe::now() - t0; };
    if (s.rays.size() >= 512) {
        s.h.gpuNs[0] += timed(512); ++s.h.gpuCalls[0];
        for (int k = 0; k < 10; ++k) { const uint64_t d = timed(1); if (k >= 5) { s.h.gpuNs[1] += d; ++s.h.gpuCalls[1]; } }
        for (int k = 0; k < 10; ++k) { const uint64_t d = timed(512); if (k >= 5) { s.h.gpuNs[2] += d; ++s.h.gpuCalls[2]; } }
    }
    { const uint64_t d = timed(uint32_t(s.rays.size())); if (s.rays.size() == 8192) { s.h.gpuNs[3] += d; ++s.h.gpuCalls[3]; } }
    const moonray::rt::GPURayIsect* is = accel->getOutputIsectBuf(tls->mThreadIdx);
    for (size_t i = 0; i < s.pending.size(); ++i) {
        const auto& p = s.pending[i];
        auto& c = s.c[p.origin];
        ++c.gpuChecked;
        const bool gpuHit = is[i].mEmbreeGeomID != RTC_INVALID_GEOMETRY_ID;
        const bool embreeHit = p.geomID != -1;
        if (!gpuHit && !embreeHit) ++c.gpuBothMiss;
        else if (!gpuHit) ++c.gpuMissEmbreeHit;
        else if (!embreeHit) ++c.gpuHitEmbreeMiss;
        else if (is[i].mEmbreeGeomID == unsigned(p.geomID) && is[i].mPrimID == unsigned(p.primID)) {
            ++c.gpuSameTriangle;
            const double abs = std::fabs(double(is[i].mTFar) - double(p.tfar));
            const double rel = abs / std::max(double(p.tfar), 1e-30);
            const double edges[5] = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3};
            int e = 0; while (e < 5 && rel >= edges[e]) ++e;
            ++s.h.tRelErr[e]; s.h.tRelMax = std::max(s.h.tRelMax, rel); s.h.tAbsMax = std::max(s.h.tAbsMax, abs);
            if (sssprobe::sameBits(is[i].mTFar, p.tfar) && sssprobe::sameBits(is[i].mU, p.u) &&
                sssprobe::sameBits(is[i].mV, p.v)) ++c.gpuSameExact;
        } else ++c.gpuOtherTriangle;
    }
    s.rays.clear(); s.pending.clear();
}

void sssCheck(moonray::mcrt_common::ThreadLocalState* tls, moonray::pbr::TLState* pbrTls, RTCScene root,
              const moonray::rt::EmbreeAccelerator* embree, int origin, bool sample, uint64_t start,
              const moonray::mcrt_common::Ray& before, const moonray::mcrt_common::Ray& after)
{
    namespace gi = moonray::geom::internal;
    auto& s = sssprobe::slot();
    auto& c = s.c[origin];
    if (after.geomID != -1) ++c.hits;
    if (origin == sssprobe::Walk && s.curSteps >= 0) ++s.curSteps;
    if (origin == sssprobe::Walk && std::getenv("XPU2_SSS_RAYDUMP") && (s.dumpCounter++ % 8) == 0) {
        struct Rec { float org[3], dir[3], tnear, tfar; int32_t hit, geomID, primID; float t; int32_t materialID, traceSet, pad[2]; } rec;
        rec.org[0] = before.org.x; rec.org[1] = before.org.y; rec.org[2] = before.org.z;
        rec.dir[0] = before.dir.x; rec.dir[1] = before.dir.y; rec.dir[2] = before.dir.z;
        rec.tnear = before.tnear; rec.tfar = before.tfar;
        rec.hit = after.geomID != -1; rec.geomID = after.geomID; rec.primID = after.primID; rec.t = after.tfar;
        rec.materialID = before.ext.materialID;
        rec.traceSet = before.ext.geomTls && static_cast<const gi::TLState*>(before.ext.geomTls)->mSubsurfaceTraceSet ? 1 : 0;
        rec.pad[0] = rec.pad[1] = 0;
        static_assert(sizeof(Rec) == 64, "record layout");
        const char* b = reinterpret_cast<const char*>(&rec);
        s.rayDump.insert(s.rayDump.end(), b, b + sizeof(rec));
    }
    if (origin == sssprobe::Walk && after.geomID != -1) {
        if (const char* gd = std::getenv("XPU2_SSS_GEODUMP")) {
            static std::mutex geoMutex; static std::vector<const void*> done;
            const auto* ud = static_cast<const gi::BVHUserData*>(after.ext.userData);
            const auto* mesh = ud ? dynamic_cast<const gi::Mesh*>(ud->mPrimitive) : nullptr;
            if (mesh) {
                std::lock_guard<std::mutex> lock(geoMutex);
                if (std::find(done.begin(), done.end(), (const void*)mesh) == done.end()) {
                    done.push_back(mesh);
                    gi::Mesh::TessellatedMesh tm; mesh->getTessellatedMesh(tm);
                    if (tm.mIndexBufferType == gi::MeshIndexType::TRIANGLE && tm.mVertexBufferDesc.size() == 1) {
                        if (FILE* g = std::fopen(gd, "ab")) {
                            const int32_t head[3] = {after.geomID, int32_t(tm.mVertexCount), int32_t(tm.mFaceCount)};
                            std::fwrite(head, sizeof(head), 1, g);
                            const auto& vb = tm.mVertexBufferDesc[0];
                            for (size_t v = 0; v < tm.mVertexCount; ++v)
                                std::fwrite(static_cast<const char*>(vb.mData) + vb.mOffset + v * vb.mStride, 12, 1, g);
                            const auto& ib = tm.mIndexBufferDesc;
                            for (size_t f = 0; f < tm.mFaceCount; ++f)
                                std::fwrite(static_cast<const char*>(ib.mData) + ib.mOffset + f * ib.mStride, 12, 1, g);
                            // Per-face material id as the trace-set filter computes it; -2 means the filter accepts.
                            const auto* named = dynamic_cast<const gi::NamedPrimitive*>(mesh);
                            for (size_t f = 0; f < tm.mFaceCount; ++f) {
                                int32_t mat = -2;
                                const int assignment = named ? named->getIntersectionAssignmentId(int(f)) : -1;
                                if (assignment >= 0) {
                                    const auto* rdl2Material = ud->mLayer->lookupMaterial(assignment);
                                    mat = rdl2Material ? rdl2Material->get<const moonray::shading::RootShader>().getMaterialId() : -3;
                                }
                                std::fwrite(&mat, 4, 1, g);
                            }
                            std::fclose(g);
                        }
                    }
                }
            }
        }
    }
    if (!sample) return;
    const uint64_t spent = sssprobe::now() - start;
    c.embreeNs += spent; ++c.timed; ++c.sampled;
    if (after.geomID != -1) { c.hitNs += spent; ++c.hitTimed; } else { c.missNs += spent; ++c.missTimed; }
    if (after.geomID != -1) {
        // Embree on [tnear, t + margin]: must return exactly the full answer.
        // (Filters run again; a volume-interval filter could record twice, so
        // this capture's image is not used.)
        moonray::mcrt_common::Ray shortRay = before;
        shortRay.tfar = std::min(before.tfar, after.tfar + std::max(after.tfar * 1e-3f, 1e-5f));
        const uint64_t t0 = sssprobe::now();
        embree->intersect(shortRay);
        c.shortNs += sssprobe::now() - t0;
        if (shortRay.geomID == after.geomID && shortRay.primID == after.primID &&
            sssprobe::sameBits(shortRay.tfar, after.tfar) && sssprobe::sameBits(shortRay.u, after.u) &&
            sssprobe::sameBits(shortRay.v, after.v) && sssprobe::sameBits(shortRay.Ng.x, after.Ng.x) &&
            sssprobe::sameBits(shortRay.Ng.y, after.Ng.y) && sssprobe::sameBits(shortRay.Ng.z, after.Ng.z))
            ++c.shortExact;
        else {
            ++c.shortDiffer;
            static std::atomic<int> dumped{0};
            if (const char* dumpPath = std::getenv("XPU2_SSS_DUMP")) if (dumped.fetch_add(1) < 40) {
                static std::mutex dumpMutex; std::lock_guard<std::mutex> lock(dumpMutex);
                if (FILE* d = std::fopen(dumpPath, "a")) {
                    std::fprintf(d, "origin=%d org=(%.9g %.9g %.9g) dir=(%.9g %.9g %.9g) tnear=%.9g tfar0=%.9g mask=%d mat=%d "
                                    "full: geom=%d prim=%d t=%.9g u=%.9g v=%.9g Ng=(%.9g %.9g %.9g) "
                                    "short: tfar=%.9g geom=%d prim=%d t=%.9g u=%.9g v=%.9g\n",
                                 origin, before.org.x, before.org.y, before.org.z, before.dir.x, before.dir.y, before.dir.z,
                                 before.tnear, before.tfar, before.mask, before.ext.materialID,
                                 after.geomID, after.primID, after.tfar, after.u, after.v, after.Ng.x, after.Ng.y, after.Ng.z,
                                 std::min(before.tfar, after.tfar + std::max(after.tfar * 1e-3f, 1e-5f)),
                                 shortRay.geomID, shortRay.primID, shortRay.tfar, shortRay.u, shortRay.v);
                    std::fclose(d);
                }
            }
            if (shortRay.geomID == -1) ++c.shortMiss;
            else if (shortRay.geomID != after.geomID || shortRay.primID != after.primID) ++c.shortOtherPrim;
        }
    }
    const bool instance = after.ext.instance0OrLight != nullptr;
    if (after.geomID != -1) {
        const auto* ud = static_cast<const gi::BVHUserData*>(after.ext.userData);
        const auto* mesh = ud ? dynamic_cast<const gi::Mesh*>(ud->mPrimitive) : nullptr;
        if (instance) ++c.skipInstance;
        else if (!mesh) ++c.skipOther;
        else {
            gi::Mesh::TessellatedMesh tm;
            mesh->getTessellatedMesh(tm);
            if (tm.mIndexBufferType != gi::MeshIndexType::TRIANGLE) ++c.skipQuad;
            else if (tm.mVertexBufferDesc.size() != 1) ++c.skipMotion;
            else {
                uint32_t idx[3];
                std::memcpy(idx, static_cast<const char*>(tm.mIndexBufferDesc.mData) + tm.mIndexBufferDesc.mOffset +
                                 size_t(after.primID) * tm.mIndexBufferDesc.mStride, sizeof(idx));
                RTCDevice device = rtcGetSceneDevice(root);
                RTCScene mini = rtcNewScene(device);
                rtcSetSceneFlags(mini, rtcGetSceneFlags(root));
                RTCGeometry g = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                const auto& vb = tm.mVertexBufferDesc[0];
                rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                           const_cast<void*>(vb.mData), vb.mOffset, vb.mStride, tm.mVertexCount);
                auto* ix = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0,
                                                                           RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), 1));
                ix[0] = idx[0]; ix[1] = idx[1]; ix[2] = idx[2];
                rtcSetGeometryMask(g, 0xFFFFFFFFu); // accept every ray mask; the real geometry's mask already passed
                rtcCommitGeometry(g);
                rtcAttachGeometry(mini, g);
                rtcReleaseGeometry(g);
                rtcCommitScene(mini);
                RTCRayHit rh;
                std::memset(&rh, 0, sizeof(rh));
                rh.ray.org_x = before.org.x; rh.ray.org_y = before.org.y; rh.ray.org_z = before.org.z;
                rh.ray.dir_x = before.dir.x; rh.ray.dir_y = before.dir.y; rh.ray.dir_z = before.dir.z;
                rh.ray.tnear = before.tnear; rh.ray.tfar = before.tfar; rh.ray.time = before.time;
                rh.ray.mask = unsigned(before.mask);
                rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                rtcIntersect1(mini, &rh);
                rtcReleaseScene(mini);
                if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) ++c.confirmMiss;
                else if (sssprobe::sameBits(rh.ray.tfar, after.tfar) && sssprobe::sameBits(rh.hit.u, after.u) &&
                         sssprobe::sameBits(rh.hit.v, after.v) && sssprobe::sameBits(rh.hit.Ng_x, after.Ng.x) &&
                         sssprobe::sameBits(rh.hit.Ng_y, after.Ng.y) && sssprobe::sameBits(rh.hit.Ng_z, after.Ng.z))
                    ++c.confirmExact;
                else ++c.confirmDiffer;
            }
        }
    }
    if (pbrTls->mFs && pbrTls->mFs->mGPUAccel) {
        moonray::rt::GPURay r;
        r.mOriginX = before.org.x; r.mOriginY = before.org.y; r.mOriginZ = before.org.z;
        r.mDirX = before.dir.x; r.mDirY = before.dir.y; r.mDirZ = before.dir.z;
        r.mMinT = before.tnear; r.mMaxT = before.tfar; r.mTime = before.time;
        r.mMask = before.mask; r.mShadowReceiverId = 0; r.mLightId = 0;
        s.rays.push_back(r);
        s.pending.push_back({origin, after.tfar, after.u, after.v, after.geomID, after.primID, instance});
        if (s.rays.size() >= 8192) sssFlush(tls, pbrTls);
    }
}
} // namespace
'''


def patch(sources):
    out = dict(sources)
    h = sources['rt/EmbreeAccelerator.h']
    h = once(h, '    void intersect(mcrt_common::Ray& ray) const;\n',
             '    void intersect(mcrt_common::Ray& ray) const;\n'
             '    RTCScene rootSceneForDiagnostics() const { return mRootScene; } // xpu2 diagnostic only\n')
    out['rt/EmbreeAccelerator.h'] = h

    s = sources['pbr/core/Scene.cc']
    s = once(s, '#include "Util.h"\n', '#include "Util.h"\n' + CHECK)
    start = s.index('Scene::intersectRay(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray, shading::Intersection &isect,')
    end = s.index('\n}\n', start)
    p = s[start:end]
    p = once(p, '''    // Intersect with manifold geometry.
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_INTERSECTION);
        mEmbreeAccel->intersect(ray);
    }
''', '''    const int sssOrigin = sssprobe::take();
    const bool sssOn = sssprobe::enabled();
    mcrt_common::Ray sssBefore;
    bool sssSample = false;
    uint64_t sssStart = 0;
    if (sssOn) {
        sssBefore = ray;
        auto& c = sssprobe::slot().c[sssOrigin];
        ++c.rays;
        sssSample = (c.rays % sssprobe::rate()) == 0;
        if (sssSample) sssStart = sssprobe::now();
    }
    // Intersect with manifold geometry.
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_INTERSECTION);
        mEmbreeAccel->intersect(ray);
    }
    if (sssOn) {
        sssCheck(tls, pbrTls, mEmbreeAccel->rootSceneForDiagnostics(), mEmbreeAccel, sssOrigin, sssSample, sssStart, sssBefore, ray);
    }
''')
    s = s[:start] + p + s[end:]
    start = s.index('Scene::isRayOccluded(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray) const')
    end = s.index('\n}\n', start)
    p = once(s[start:end], '    ray.mask = rdl2::SHADOW;\n',
             '    ray.mask = rdl2::SHADOW;\n'
             '    if (sssprobe::enabled()) ++sssprobe::slot().c[sssprobe::depth > 0 ? sssprobe::Other : sssprobe::Outside].occlusionRays;\n')
    out['pbr/core/Scene.cc'] = s[:start] + p + s[end:]

    u = sources['pbr/integrator/PathIntegratorUtil.cc']
    u = u.replace('#include', '#include "sss_probe.h"\n#include', 1)
    start = u.index('CPP_computeRadianceSubsurface(const PathIntegrator * pathIntegrator,')
    end = u.index('\n}\n', start)
    p = once(u[start:end], '    pbrTls->stopIspcAccumulator();\n', '    pbrTls->stopIspcAccumulator();\n    sssprobe::Scope sssScope;\n')
    out['pbr/integrator/PathIntegratorUtil.cc'] = u[:start] + p + u[end:]

    t = sources['pbr/integrator/PathIntegratorSubsurface.cc']
    t = t.replace('#include', '#include "sss_probe.h"\n#include', 1)
    t = once(t, 'bool intersected = scene->intersectRay(pbrTls->mTopLevelTls, rayProj,',
             'bool intersected = sssprobe::tag(sssprobe::Probe, scene)->intersectRay(pbrTls->mTopLevelTls, rayProj,')
    t = once(t, 'if (!scene->intersectRay(pbrTls->mTopLevelTls, rayBack, isectBack, lobeType) ||',
             'if (!sssprobe::tag(sssprobe::Back, scene)->intersectRay(pbrTls->mTopLevelTls, rayBack, isectBack, lobeType) ||')
    walk = 'bool intersected = scene->intersectRay(pbrTls->mTopLevelTls, sssRay,'
    if t.count(walk) != 2:
        raise ValueError('Random-walk intersect drift')
    t = t.replace(walk, 'bool intersected = sssprobe::tag(sssprobe::Walk, scene)->intersectRay(pbrTls->mTopLevelTls, sssRay,')
    start = t.index('PathIntegrator::computeRadiancePathTraceSubsurface(')
    end = t.index('\n}\n', start)
    w = once(t[start:end], '    for (int s = 0; s < nSubsurfaceSample; ++s) {\n',
             '    for (int s = 0; s < nSubsurfaceSample; ++s) {\n        sssprobe::walkBegin();\n')
    out['pbr/integrator/PathIntegratorSubsurface.cc'] = t[:start] + w + t[end:]
    return out

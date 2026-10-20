// GPU miss proof for random-walk subsurface waves; see sss_gpu.h.
// On first use, every mesh of the Embree accelerator is read once into a
// triangle list with each face's layer and assignment id. Each walk key (the
// ray's material id and subsurface trace set) then gets its own structure, built
// on first use from the faces Embree's bssrdfTraceSetIntersectionFilter can accept
// for that key (rt/IntersectionFilters.cc, same expressions), and uploaded once.
// Masks and the other filters only remove hits, so they are left to Embree.
// Each wave goes through per-thread zero-copy host-mapped buffers and one launch
// of xpu2/sss_missproof.cu, each ray naming its key's structure.
// Scenes with anything other than single-step triangle or quad meshes (instances,
// curves, spheres, volumes, motion blur) are not supported: prove() returns false
// and the reason is in the report.
// XPU2_SSS_GPU_KEEPALIVE_US > 0: while waves are being proven, a thread launches an
// empty kernel at that period so the GPU does not drop into its idle state
// between waves; it sleeps after 200 ms without a wave.
#include "sss_gpu.h"

#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/shading/RootShader.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

namespace {

// EmbreeAccelerator keeps its BVHUserData list private; an explicit instantiation
// may name a private member, which gives read access without changing the class.
using UserDataMember = moonray::rt::BVHUserDataList moonray::rt::EmbreeAccelerator::*;
UserDataMember userDataMember();
template <UserDataMember M> struct Access { friend UserDataMember userDataMember() { return M; } };
template struct Access<&moonray::rt::EmbreeAccelerator::mBVHUserData>;

#define FN(n) decltype(&n) p_##n = nullptr;
FN(cuInit) FN(cuDeviceGet) FN(cuDevicePrimaryCtxRetain) FN(cuCtxSetCurrent) FN(cuStreamCreate)
FN(cuMemAlloc_v2) FN(cuMemcpyHtoD_v2) FN(cuMemHostAlloc) FN(cuMemHostGetDevicePointer_v2)
FN(cuModuleLoadData) FN(cuModuleGetFunction) FN(cuLaunchKernel) FN(cuStreamSynchronize)
#undef FN

struct Node { float lo[3]; int a; float hi[3]; int b; };   // leaf: a = first triangle, b = -count
struct Tri { float v[9]; float c[3]; float lo[3], hi[3]; };
struct Rec { float r[8]; uint64_t nodes, tris; };          // as in sss_missproof.cu

void grow(float* lo, float* hi, const float* plo, const float* phi)
{
    for (int i = 0; i < 3; ++i) { lo[i] = std::min(lo[i], plo[i]); hi[i] = std::max(hi[i], phi[i]); }
}
float area(const float* lo, const float* hi)
{
    float d[3]; for (int i = 0; i < 3; ++i) d[i] = std::max(0.f, hi[i] - lo[i]);
    return d[0]*d[1] + d[1]*d[2] + d[2]*d[0];
}

// Binned SAH, leaves of at most 4 triangles (as in xpu2/sss_gpu_missproof_bench.cc).
int build(std::vector<Node>& nodes, std::vector<Tri>& t, int begin, int end)
{
    const int me = int(nodes.size()); nodes.push_back({});
    Node nd{}; for (int i = 0; i < 3; ++i) { nd.lo[i] = 1e30f; nd.hi[i] = -1e30f; }
    float clo[3] = {1e30f, 1e30f, 1e30f}, chi[3] = {-1e30f, -1e30f, -1e30f};
    for (int k = begin; k < end; ++k) { grow(nd.lo, nd.hi, t[k].lo, t[k].hi); grow(clo, chi, t[k].c, t[k].c); }
    const int count = end - begin;
    if (count <= 4) { nd.a = begin; nd.b = -count; nodes[me] = nd; return me; }
    int bestAxis = -1; float bestCost = 1e30f, bestSplit = 0;
    constexpr int kBins = 16;
    for (int ax = 0; ax < 3; ++ax) {
        const float ext = chi[ax] - clo[ax];
        if (ext <= 0) continue;
        float blo[kBins][3], bhi[kBins][3]; int bn[kBins] = {};
        for (int b = 0; b < kBins; ++b) for (int i = 0; i < 3; ++i) { blo[b][i] = 1e30f; bhi[b][i] = -1e30f; }
        for (int k = begin; k < end; ++k) {
            const int b = std::min(kBins - 1, int((t[k].c[ax] - clo[ax]) / ext * kBins));
            ++bn[b]; grow(blo[b], bhi[b], t[k].lo, t[k].hi);
        }
        for (int s = 1; s < kBins; ++s) {
            float llo[3] = {1e30f, 1e30f, 1e30f}, lhi[3] = {-1e30f, -1e30f, -1e30f};
            float rlo[3] = {1e30f, 1e30f, 1e30f}, rhi[3] = {-1e30f, -1e30f, -1e30f};
            int ln = 0, rn = 0;
            for (int b = 0; b < s; ++b) if (bn[b]) { grow(llo, lhi, blo[b], bhi[b]); ln += bn[b]; }
            for (int b = s; b < kBins; ++b) if (bn[b]) { grow(rlo, rhi, blo[b], bhi[b]); rn += bn[b]; }
            if (!ln || !rn) continue;
            const float cost = area(llo, lhi) * ln + area(rlo, rhi) * rn;
            if (cost < bestCost) { bestCost = cost; bestAxis = ax; bestSplit = clo[ax] + ext * s / kBins; }
        }
    }
    int mid = begin + count / 2;
    if (bestAxis >= 0) {
        auto it = std::partition(t.begin() + begin, t.begin() + end, [&](const Tri& x) { return x.c[bestAxis] < bestSplit; });
        mid = int(it - t.begin());
        if (mid == begin || mid == end) mid = begin + count / 2;
    }
    const int left = build(nodes, t, begin, mid);
    const int right = build(nodes, t, mid, end);
    nd.a = left; nd.b = right; nodes[me] = nd;
    return me;
}

int keepAliveUs()
{
    static const int k = [] { const char* v = std::getenv("XPU2_SSS_GPU_KEEPALIVE_US"); return v ? std::max(0, std::atoi(v)) : 0; }();
    return k;
}

struct Stats {
    std::atomic<uint64_t> calls{0}, rays{0}, proven{0}, refused{0}, kernelNs{0}, verifyChecked{0}, verifyHits{0}, keepalives{0};
    double soupMs = 0, keyMs = 0; size_t meshes = 0, triangles = 0, keys = 0, keyTriangles = 0;
    std::string reason;
    ~Stats()
    {
        const char* path = std::getenv("XPU2_SSS_GPU_REPORT");
        if (!path) return;
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "threshold,verify,calls,rays,proven,refused,kernel_ms,soup_ms,key_build_ms,meshes,triangles,"
                            "keys,key_triangles,verify_checked,verify_embree_hits,keepalive_us,keepalives,reason\n"
                            "%d,%d,%llu,%llu,%llu,%llu,%.3f,%.1f,%.1f,%zu,%zu,%zu,%zu,%llu,%llu,%d,%llu,%s\n",
                         xpu2sssgpu::threshold(), int(xpu2sssgpu::verify()), (unsigned long long)calls,
                         (unsigned long long)rays, (unsigned long long)proven, (unsigned long long)refused,
                         kernelNs * 1e-6, soupMs, keyMs, meshes, triangles, keys, keyTriangles,
                         (unsigned long long)verifyChecked, (unsigned long long)verifyHits, keepAliveUs(),
                         (unsigned long long)keepalives, reason.c_str());
            std::fclose(f);
        }
    }
};
Stats& stats() { static Stats s; return s; }

void check(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) throw std::string(what) + " " + std::to_string(int(r));
}

struct KeyStructure { CUdeviceptr nodes = 0, tris = 0; };   // null: no face the key accepts

struct Shared {
    std::mutex mutex;
    const void* accel = nullptr;
    bool ready = false, failed = false;
    CUcontext ctx = nullptr;
    CUfunction fn = nullptr, keepFn = nullptr;
    bool keepStarted = false;
    std::condition_variable keepWake;
    std::atomic<int64_t> lastWave{0};
    // Every mesh face as triangles, with the face's layer and assignment id.
    std::vector<Tri> soup;
    std::vector<const scene_rdl2::rdl2::Layer*> soupLayer;
    std::vector<int> soupAssignment;
    std::map<std::pair<const void*, int>, std::unique_ptr<KeyStructure>> keys;   // (trace set, material id)
};
// Never destroyed: the keep-alive thread may still wait on it at exit.
Shared& shared() { static Shared* s = new Shared; return *s; }

// Under the shared mutex.
void readMeshes(Shared& s, const moonray::rt::EmbreeAccelerator* accel)
{
    namespace gi = moonray::geom::internal;
    Stats& st = stats();
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& ud : accel->*userDataMember()) {
        const auto* mesh = ud ? dynamic_cast<const gi::Mesh*>(ud->mPrimitive) : nullptr;
        if (!mesh) throw std::string("unsupported primitive ") + (ud && ud->mPrimitive ? typeid(*ud->mPrimitive).name() : "null");
        gi::Mesh::TessellatedMesh tm;
        mesh->getTessellatedMesh(tm);
        if (tm.mVertexBufferDesc.size() != 1) throw std::string("motion blur mesh");
        const bool quad = tm.mIndexBufferType == gi::MeshIndexType::QUAD;
        if (!quad && tm.mIndexBufferType != gi::MeshIndexType::TRIANGLE) throw std::string("mesh index type");
        const auto& vb = tm.mVertexBufferDesc[0];
        const auto& ib = tm.mIndexBufferDesc;
        auto vertex = [&](uint32_t i, float* out) {
            std::memcpy(out, static_cast<const char*>(vb.mData) + vb.mOffset + size_t(i) * vb.mStride, 12);
        };
        for (size_t f = 0; f < tm.mFaceCount; ++f) {
            // Embree's primID is the face index, which the filter passes on.
            const int assignment = mesh->getIntersectionAssignmentId(int(f));
            auto add = [&](const float* a, const float* b, const float* c) {
                Tri t{};
                std::memcpy(t.v, a, 12); std::memcpy(t.v + 3, b, 12); std::memcpy(t.v + 6, c, 12);
                for (int k = 0; k < 3; ++k) {
                    t.lo[k] = std::min({t.v[k], t.v[3+k], t.v[6+k]}); t.hi[k] = std::max({t.v[k], t.v[3+k], t.v[6+k]});
                    t.c[k] = 0.5f * (t.lo[k] + t.hi[k]);
                }
                s.soup.push_back(t); s.soupLayer.push_back(ud->mLayer); s.soupAssignment.push_back(assignment);
            };
            uint32_t idx[4] = {};
            std::memcpy(idx, static_cast<const char*>(ib.mData) + ib.mOffset + f * ib.mStride, quad ? 16 : 12);
            float p[4][3];
            for (int k = 0; k < (quad ? 4 : 3); ++k) vertex(idx[k], p[k]);
            if (!quad) { add(p[0], p[1], p[2]); continue; }
            // Both diagonals, so whichever split Embree uses is covered.
            add(p[0], p[1], p[3]); add(p[2], p[3], p[1]);
            add(p[0], p[1], p[2]); add(p[0], p[2], p[3]);
        }
        ++st.meshes;
    }
    st.triangles = s.soup.size();

    const char* ptxPath = std::getenv("XPU2_SSS_GPU_PTX");
    if (!ptxPath) throw std::string("XPU2_SSS_GPU_PTX not set");
    std::ifstream pf(ptxPath, std::ios::binary);
    std::string ptx((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    if (ptx.empty()) throw std::string("PTX not readable");
    HMODULE lib = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) throw std::string("nvcuda.dll");
#define LOAD(n) p_##n = reinterpret_cast<decltype(p_##n)>(GetProcAddress(lib, #n)); if (!p_##n) throw std::string(#n);
    LOAD(cuInit) LOAD(cuDeviceGet) LOAD(cuDevicePrimaryCtxRetain) LOAD(cuCtxSetCurrent) LOAD(cuStreamCreate)
    LOAD(cuMemAlloc_v2) LOAD(cuMemcpyHtoD_v2) LOAD(cuMemHostAlloc) LOAD(cuMemHostGetDevicePointer_v2)
    LOAD(cuModuleLoadData) LOAD(cuModuleGetFunction) LOAD(cuLaunchKernel) LOAD(cuStreamSynchronize)
#undef LOAD
    check(p_cuInit(0), "cuInit");
    CUdevice dev; check(p_cuDeviceGet(&dev, 0), "cuDeviceGet");
    check(p_cuDevicePrimaryCtxRetain(&s.ctx, dev), "cuDevicePrimaryCtxRetain");   // shared with MoonRay's CUDA runtime
    check(p_cuCtxSetCurrent(s.ctx), "cuCtxSetCurrent");
    CUmodule mod; check(p_cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
    check(p_cuModuleGetFunction(&s.fn, mod, "sss_missproof"), "cuModuleGetFunction");
    if (keepAliveUs() > 0) check(p_cuModuleGetFunction(&s.keepFn, mod, "sss_keepalive"), "cuModuleGetFunction");
    st.soupMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Under the shared mutex: the structure over the faces bssrdfTraceSetIntersectionFilter
// accepts for a ray with this material id and trace set.
const KeyStructure* keyStructure(Shared& s, const void* traceSetPtr, int materialID)
{
    // The filter ignores the trace set when the ray has no material id.
    const std::pair<const void*, int> key(materialID == -1 ? nullptr : traceSetPtr, materialID);
    auto found = s.keys.find(key);
    if (found != s.keys.end()) return found->second.get();
    Stats& st = stats();
    const auto t0 = std::chrono::steady_clock::now();
    const auto* traceSet = static_cast<const scene_rdl2::rdl2::TraceSet*>(key.first);
    std::map<std::pair<const void*, int>, bool> accepts;   // (layer, assignment id)
    std::vector<Tri> tris;
    for (size_t k = 0; k < s.soup.size(); ++k) {
        const int assignment = s.soupAssignment[k];
        bool keep = materialID == -1 || assignment < 0;   // the filter returns without rejecting
        if (!keep) {
            const auto ak = std::make_pair(static_cast<const void*>(s.soupLayer[k]), assignment);
            auto a = accepts.find(ak);
            if (a == accepts.end()) {
                bool ok;
                if (traceSet) {
                    const auto& gp = s.soupLayer[k]->lookupGeomAndPart(assignment);
                    ok = traceSet->getAssignmentId(gp.first, gp.second) >= 0;
                } else {
                    const scene_rdl2::rdl2::Material* m = s.soupLayer[k]->lookupMaterial(assignment);
                    // A face without a material is kept, to stay a superset.
                    ok = !m || int(m->get<const moonray::shading::RootShader>().getMaterialId()) == materialID;
                }
                a = accepts.emplace(ak, ok).first;
            }
            keep = a->second;
        }
        if (keep) tris.push_back(s.soup[k]);
    }
    auto ks = std::make_unique<KeyStructure>();
    if (!tris.empty()) {
        std::vector<Node> nodes; nodes.reserve(tris.size());
        build(nodes, tris, 0, int(tris.size()));
        std::vector<float> flat(tris.size() * 9);
        for (size_t k = 0; k < tris.size(); ++k) std::memcpy(&flat[9 * k], tris[k].v, 36);
        check(p_cuMemAlloc_v2(&ks->nodes, nodes.size() * sizeof(Node)), "cuMemAlloc");
        check(p_cuMemcpyHtoD_v2(ks->nodes, nodes.data(), nodes.size() * sizeof(Node)), "cuMemcpyHtoD");
        check(p_cuMemAlloc_v2(&ks->tris, flat.size() * 4), "cuMemAlloc");
        check(p_cuMemcpyHtoD_v2(ks->tris, flat.data(), flat.size() * 4), "cuMemcpyHtoD");
    }
    ++st.keys; st.keyTriangles += tris.size();
    st.keyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return s.keys.emplace(key, std::move(ks)).first->second.get();
}

int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void keepAlive()
{
    Shared& s = shared();
    if (p_cuCtxSetCurrent(s.ctx) != CUDA_SUCCESS) return;
    CUstream stream;
    if (p_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return;
    // Sleep() resolution is too coarse for sub-millisecond periods.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    // XPU2_SSS_GPU_KEEPALIVE_SPIN_US: how long each keep-alive kernel keeps one warp busy.
    unsigned long long spinNs = 0;
    if (const char* v = std::getenv("XPU2_SSS_GPU_KEEPALIVE_SPIN_US")) spinNs = 1000ull * std::strtoull(v, nullptr, 10);
    for (;;) {
        if (nowNs() - s.lastWave.load() > 200000000) {
            std::unique_lock<std::mutex> lock(s.mutex);
            s.keepWake.wait(lock, [&] { return nowNs() - s.lastWave.load() <= 200000000; });
        }
        void* args[] = {&spinNs};
        if (p_cuLaunchKernel(s.keepFn, 1, 1, 1, 32, 1, 1, 0, stream, args, nullptr) != CUDA_SUCCESS) return;
        ++stats().keepalives;
        LARGE_INTEGER due; due.QuadPart = -10LL * keepAliveUs();
        SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
        WaitForSingleObject(timer, INFINITE);
    }
}

struct ThreadState {
    bool ready = false;
    CUstream stream = nullptr;
    void *hRays = nullptr, *hOut = nullptr;
    CUdeviceptr dRays = 0, dOut = 0;
    int capacity = 0;
};
thread_local ThreadState tls;

} // namespace

namespace xpu2sssgpu {

bool enabled()
{
    static const bool e = [] { const char* v = std::getenv("XPU2_SSS_GPU"); return v && !std::strcmp(v, "1"); }();
    return e;
}

int threshold()
{
    static const int t = [] { const char* v = std::getenv("XPU2_SSS_GPU_MIN"); return v ? std::max(1, std::atoi(v)) : 128; }();
    return t;
}

bool verify()
{
    static const bool e = [] { const char* v = std::getenv("XPU2_SSS_GPU_VERIFY"); return v && !std::strcmp(v, "1"); }();
    return e;
}

void noteVerify(uint64_t checked, uint64_t embreeHits)
{
    stats().verifyChecked += checked;
    stats().verifyHits += embreeHits;
}

bool prove(const void* embreeAccelerator, const float* rays, const int* materialIDs, const void* const* traceSets,
           int n, unsigned char* candidate)
{
    Shared& s = shared();
    Stats& st = stats();
    try {
        Rec* rec = nullptr;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            if (!s.ready && !s.failed) {
                try {
                    readMeshes(s, static_cast<const moonray::rt::EmbreeAccelerator*>(embreeAccelerator));
                    s.accel = embreeAccelerator;
                    s.ready = true;
                } catch (const std::string& e) {
                    s.failed = true;
                    st.reason = e;
                }
            }
            if (!s.ready || s.accel != embreeAccelerator) { ++st.refused; return false; }
            if (!tls.ready) {
                check(p_cuCtxSetCurrent(s.ctx), "cuCtxSetCurrent");
                check(p_cuStreamCreate(&tls.stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
                tls.ready = true;
            }
            if (n > tls.capacity) {
                // Grown buffers are not freed: a few hundred kilobytes per render thread at most.
                const int cap = std::max(n, 4096);
                check(p_cuMemHostAlloc(&tls.hRays, size_t(cap) * sizeof(Rec), CU_MEMHOSTALLOC_DEVICEMAP), "cuMemHostAlloc");
                check(p_cuMemHostAlloc(&tls.hOut, size_t(cap), CU_MEMHOSTALLOC_DEVICEMAP), "cuMemHostAlloc");
                check(p_cuMemHostGetDevicePointer_v2(&tls.dRays, tls.hRays, 0), "cuMemHostGetDevicePointer");
                check(p_cuMemHostGetDevicePointer_v2(&tls.dOut, tls.hOut, 0), "cuMemHostGetDevicePointer");
                tls.capacity = cap;
            }
            rec = static_cast<Rec*>(tls.hRays);
            // A wave holds a few keys, mostly in runs: look each run up once.
            const KeyStructure* ks = nullptr;
            const void* lastSet = nullptr;
            int lastMat = 0;
            for (int i = 0; i < n; ++i) {
                if (!ks || traceSets[i] != lastSet || materialIDs[i] != lastMat) {
                    ks = keyStructure(s, traceSets[i], materialIDs[i]);
                    lastSet = traceSets[i]; lastMat = materialIDs[i];
                }
                std::memcpy(rec[i].r, rays + 8 * size_t(i), 32);
                rec[i].nodes = ks->nodes; rec[i].tris = ks->tris;
            }
        }
        unsigned count = unsigned(n);
        float margin = 1e-5f;
        void* args[] = {&tls.dRays, &tls.dOut, &count, &margin};
        const auto t0 = std::chrono::steady_clock::now();
        check(p_cuLaunchKernel(s.fn, (count + 127) / 128, 1, 1, 128, 1, 1, 0, tls.stream, args, nullptr), "cuLaunchKernel");
        check(p_cuStreamSynchronize(tls.stream), "cuStreamSynchronize");
        st.kernelNs += uint64_t(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
        std::memcpy(candidate, tls.hOut, size_t(n));
    } catch (const std::string& e) {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.ready = false; s.failed = true;
        st.reason = e;
        ++st.refused;
        return false;
    }
    if (keepAliveUs() > 0) {
        const int64_t now = nowNs();
        const bool wasIdle = now - s.lastWave.exchange(now) > 200000000;
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.keepStarted) { s.keepStarted = true; std::thread(keepAlive).detach(); }
        else if (wasIdle) s.keepWake.notify_all();
    }
    uint64_t proven = 0;
    for (int i = 0; i < n; ++i) proven += candidate[i] == 0;
    ++st.calls; st.rays += uint64_t(n); st.proven += proven;
    return true;
}

} // namespace xpu2sssgpu

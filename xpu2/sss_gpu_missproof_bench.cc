// Measurement prototype: a dedicated GPU structure over the subsurface
// trace-set geometry, and a conservative any-hit ("miss proof") kernel for
// random-walk rays, called through zero-copy host-mapped buffers.
//
// Inputs come from the capture build (xpu2/rndr_sss_feasibility.py):
//   geo.bin  : per mesh {int geomID, int nVerts, int nFaces} float3[nVerts] uint3[nFaces]
//   rays.bin : 48-byte records {float org[3], dir[3], tnear, tfar; int hit, geomID, primID; float t}
// The kernel must report a candidate for every ray Embree hit (no false
// negatives); extra candidates would go to Embree. Not part of any renderer.
//
// usage: sss_gpu_missproof_bench <geo.bin> <rays.bin> <nvrtc bin dir> <out.csv>
#include <windows.h>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define FN(n) static decltype(&n) p_##n = nullptr;
FN(cuInit) FN(cuDeviceGet) FN(cuDevicePrimaryCtxRetain) FN(cuCtxSetCurrent) FN(cuStreamCreate)
FN(cuMemAlloc_v2) FN(cuMemcpyHtoD_v2) FN(cuMemHostAlloc) FN(cuMemHostGetDevicePointer_v2)
FN(cuModuleLoadData) FN(cuModuleGetFunction) FN(cuLaunchKernel) FN(cuStreamSynchronize)
#undef FN
typedef int (*NvrtcCreate)(void**, const char*, const char*, int, const char* const*, const char* const*);
typedef int (*NvrtcCompile)(void*, int, const char* const*);
typedef int (*NvrtcSize)(void*, size_t*);
typedef int (*NvrtcGet)(void*, char*);

static void check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(int(r)));
}

static const char* kKernel = R"CUDA(
struct Node { float lo[3]; int a; float hi[3]; int b; };   // leaf: a = first triangle, b = -count
struct Ray { float org[3], dir[3], tnear, tfar; int hit, geomID, primID; float t; int materialID, traceSet, pad0, pad1; };

__device__ double dot3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Squared distance from point p to triangle (a, b, c), double precision (Ericson, RTCD 5.1.5).
__device__ double pointTriangleDist2(const double* p, const double* a, const double* b, const double* c) {
    double ab[3], ac[3], ap[3];
    for (int i = 0; i < 3; ++i) { ab[i] = b[i]-a[i]; ac[i] = c[i]-a[i]; ap[i] = p[i]-a[i]; }
    double d1 = dot3(ab, ap), d2 = dot3(ac, ap), q[3];
    if (d1 <= 0 && d2 <= 0) { for (int i = 0; i < 3; ++i) q[i] = a[i]; goto done; }
    {
    double bp[3]; for (int i = 0; i < 3; ++i) bp[i] = p[i]-b[i];
    double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) { for (int i = 0; i < 3; ++i) q[i] = b[i]; goto done; }
    double vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { double v = d1/(d1-d3); for (int i = 0; i < 3; ++i) q[i] = a[i]+v*ab[i]; goto done; }
    double cp[3]; for (int i = 0; i < 3; ++i) cp[i] = p[i]-c[i];
    double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) { for (int i = 0; i < 3; ++i) q[i] = c[i]; goto done; }
    double vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { double w = d2/(d2-d6); for (int i = 0; i < 3; ++i) q[i] = a[i]+w*ac[i]; goto done; }
    double va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        double w = (d4-d3)/((d4-d3)+(d5-d6)); for (int i = 0; i < 3; ++i) q[i] = b[i]+w*(c[i]-b[i]); goto done; }
    double denom = 1.0/(va+vb+vc), v = vb*denom, w = vc*denom;
    for (int i = 0; i < 3; ++i) q[i] = a[i]+ab[i]*v+ac[i]*w;
    }
done:
    double dd[3]; for (int i = 0; i < 3; ++i) dd[i] = p[i]-q[i];
    return dot3(dd, dd);
}

extern "C" __global__ void missproof(const Ray* rays, const Node* nodes, const float* tris, const int* mats,
                                     unsigned char* out, unsigned n, float relMargin, int useMaterial)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const Ray r = rays[id];
    // Non-finite rays go to Embree untouched (a NaN never culls a box).
    bool finite = isfinite(r.tnear) && isfinite(r.tfar);
    for (int i = 0; i < 3; ++i) finite = finite && isfinite(r.org[i]) && isfinite(r.dir[i]);
    if (!finite) { out[id] = 1; return; }
    float scale = 1.f;
    for (int i = 0; i < 3; ++i) scale = fmaxf(scale, fabsf(r.org[i]));
    // Side margin (distance from the triangle), generous against float rounding at
    // this coordinate scale; the interval margin stays a few float steps, below the
    // walk's start offset from its surface (1e-5).
    const float m = relMargin * (scale + r.tfar);
    const float mt = 4e-7f * (scale + r.tfar);
    const float t0 = r.tnear - mt, t1 = r.tfar + mt;
    float inv[3];
    for (int i = 0; i < 3; ++i) inv[i] = 1.f / (fabsf(r.dir[i]) > 1e-20f ? r.dir[i] : copysignf(1e-20f, r.dir[i]));
    int stack[64]; int sp = 0; stack[sp++] = 0;
    unsigned char found = 0;
    int visits = 0;
    while (sp && !found) {
        if (++visits > 4096) { found = 1; break; }   // bounded work per ray: give up to Embree
        const Node nd = nodes[stack[--sp]];
        float tmin = t0, tmax = t1;
        bool cull = false;
        for (int i = 0; i < 3 && !cull; ++i) {
            float a = (nd.lo[i] - m - r.org[i]) * inv[i], b = (nd.hi[i] + m - r.org[i]) * inv[i];
            if (a > b) { float t = a; a = b; b = t; }
            // Slack for rounding in the slab arithmetic itself.
            a -= 1e-5f * fabsf(a) + m; b += 1e-5f * fabsf(b) + m;
            tmin = fmaxf(tmin, a); tmax = fminf(tmax, b);
            cull = tmin > tmax;
        }
        if (cull) continue;
        if (nd.b < 0) {
            for (int k = nd.a; k < nd.a - nd.b && !found; ++k) {
                // Embree's default trace-set rule: a different material id rejects the hit.
                if ((useMaterial == 1 || useMaterial == 11) && r.materialID != -1 && !r.traceSet && mats[k] >= 0 && mats[k] != r.materialID) {
                    if (useMaterial == 2 && id < 2) printf("reject id=%u k=%d mat=%d ray=%d trace=%d\n", id, k, mats[k], r.materialID, r.traceSet);
                    continue;
                }
                const float* tv = tris + 9 * k;
                if (useMaterial >= 10) {
                    // Float Moller-Trumbore. Clear misses are rejected and clear hits accepted;
                    // anything within tau of an edge or of the interval falls through to the
                    // double-precision test below.
                    const float tau = 1e-4f;
                    float fe1[3], fe2[3], fs[3], P[3], Q[3];
                    for (int i = 0; i < 3; ++i) { fe1[i] = tv[3+i]-tv[i]; fe2[i] = tv[6+i]-tv[i]; fs[i] = r.org[i]-tv[i]; }
                    P[0] = r.dir[1]*fe2[2]-r.dir[2]*fe2[1]; P[1] = r.dir[2]*fe2[0]-r.dir[0]*fe2[2]; P[2] = r.dir[0]*fe2[1]-r.dir[1]*fe2[0];
                    const float det = fe1[0]*P[0]+fe1[1]*P[1]+fe1[2]*P[2];
                    const float e1n = sqrtf(fe1[0]*fe1[0]+fe1[1]*fe1[1]+fe1[2]*fe1[2]);
                    const float e2n = sqrtf(fe2[0]*fe2[0]+fe2[1]*fe2[1]+fe2[2]*fe2[2]);
                    if (fabsf(det) > 1e-3f * e1n * e2n) {
                        const float inv = 1.f / det;
                        const float u = (fs[0]*P[0]+fs[1]*P[1]+fs[2]*P[2]) * inv;
                        Q[0] = fs[1]*fe1[2]-fs[2]*fe1[1]; Q[1] = fs[2]*fe1[0]-fs[0]*fe1[2]; Q[2] = fs[0]*fe1[1]-fs[1]*fe1[0];
                        const float v = (r.dir[0]*Q[0]+r.dir[1]*Q[1]+r.dir[2]*Q[2]) * inv;
                        const float t = (fe2[0]*Q[0]+fe2[1]*Q[1]+fe2[2]*Q[2]) * inv;
                        // Side margin m expressed in barycentric units of this triangle.
                        const float bm = tau + m / fmaxf(1e-30f, fminf(e1n, e2n));
                        const float tm = tau * fabsf(t) + mt;
                        const bool out_ = u < -bm || v < -bm || u + v > 1.f + bm || t < t0 - tm || t > t1 + tm;
                        if (out_) continue;
                        const bool in_ = u > bm && v > bm && u + v < 1.f - bm && t > t0 + tm && t < t1 - tm;
                        if (in_) { found = 1; break; }
                    }
                }
                double A[3] = {tv[0], tv[1], tv[2]}, B[3] = {tv[3], tv[4], tv[5]}, C[3] = {tv[6], tv[7], tv[8]};
                double e1[3], e2[3], nrm[3], o[3] = {r.org[0], r.org[1], r.org[2]}, d[3] = {r.dir[0], r.dir[1], r.dir[2]};
                for (int i = 0; i < 3; ++i) { e1[i] = B[i]-A[i]; e2[i] = C[i]-A[i]; }
                nrm[0] = e1[1]*e2[2]-e1[2]*e2[1]; nrm[1] = e1[2]*e2[0]-e1[0]*e2[2]; nrm[2] = e1[0]*e2[1]-e1[1]*e2[0];
                const double nlen = sqrt(dot3(nrm, nrm)), dlen = sqrt(dot3(d, d));
                const double denom = dot3(nrm, d);
                if (nlen == 0.0 || fabs(denom) < 1e-6 * nlen * dlen) { found = 1; break; }  // grazing: candidate
                double ao[3]; for (int i = 0; i < 3; ++i) ao[i] = A[i]-o[i];
                const double t = dot3(nrm, ao) / denom;
                if (t < t0 || t > t1) {
                    // The plane may sit just outside the interval: test the interval ends' points too.
                    const double te = t < t0 ? t0 : t1;
                    double pe[3]; for (int i = 0; i < 3; ++i) pe[i] = o[i] + te*d[i];
                    if (pointTriangleDist2(pe, A, B, C) <= double(m)*m) found = 1;
                    continue;
                }
                double p[3]; for (int i = 0; i < 3; ++i) p[i] = o[i] + t*d[i];
                if (pointTriangleDist2(p, A, B, C) <= double(m)*m) found = 1;
            }
        } else {
            if (sp < 62) { stack[sp++] = nd.b; stack[sp++] = nd.a; }
            else found = 1;  // stack overflow: conservative
        }
    }
    out[id] = found;
}
)CUDA";

struct Tri { float v[9]; float c[3]; float lo[3], hi[3]; int mat; };
struct Node { float lo[3]; int a; float hi[3]; int b; };

static void grow(float* lo, float* hi, const float* plo, const float* phi) {
    for (int i = 0; i < 3; ++i) { lo[i] = std::min(lo[i], plo[i]); hi[i] = std::max(hi[i], phi[i]); }
}
static float area(const float* lo, const float* hi) {
    float d[3]; for (int i = 0; i < 3; ++i) d[i] = std::max(0.f, hi[i] - lo[i]);
    return d[0]*d[1] + d[1]*d[2] + d[2]*d[0];
}

// Binned SAH build, leaves of at most 4 triangles.
static int build(std::vector<Node>& nodes, std::vector<Tri>& t, int begin, int end) {
    const int me = int(nodes.size()); nodes.push_back({});
    Node nd{}; for (int i = 0; i < 3; ++i) { nd.lo[i] = 1e30f; nd.hi[i] = -1e30f; }
    float clo[3] = {1e30f, 1e30f, 1e30f}, chi[3] = {-1e30f, -1e30f, -1e30f};
    for (int k = begin; k < end; ++k) { grow(nd.lo, nd.hi, t[k].lo, t[k].hi); grow(clo, chi, t[k].c, t[k].c); }
    const int count = end - begin;
    if (count <= 4) { nd.a = begin; nd.b = -count; nodes[me] = nd; return me; }
    int bestAxis = -1; float bestCost = 1e30f; float bestSplit = 0;
    constexpr int kBins = 16;
    for (int ax = 0; ax < 3; ++ax) {
        const float ext = chi[ax] - clo[ax];
        if (ext <= 0) continue;
        float blo[kBins][3], bhi[kBins][3]; int bn[kBins] = {};
        for (int b = 0; b < kBins; ++b) for (int i = 0; i < 3; ++i) { blo[b][i] = 1e30f; bhi[b][i] = -1e30f; }
        for (int k = begin; k < end; ++k) {
            int b = std::min(kBins - 1, int((t[k].c[ax] - clo[ax]) / ext * kBins));
            ++bn[b]; grow(blo[b], bhi[b], t[k].lo, t[k].hi);
        }
        for (int s = 1; s < kBins; ++s) {
            float llo[3] = {1e30f, 1e30f, 1e30f}, lhi[3] = {-1e30f, -1e30f, -1e30f}, rlo[3] = {1e30f, 1e30f, 1e30f}, rhi[3] = {-1e30f, -1e30f, -1e30f};
            int ln = 0, rn = 0;
            for (int b = 0; b < s; ++b) if (bn[b]) { grow(llo, lhi, blo[b], bhi[b]); ln += bn[b]; }
            for (int b = s; b < kBins; ++b) if (bn[b]) { grow(rlo, rhi, blo[b], bhi[b]); rn += bn[b]; }
            if (!ln || !rn) continue;
            const float cost = area(llo, lhi) * ln + area(rlo, rhi) * rn;
            if (cost < bestCost) { bestCost = cost; bestAxis = ax; bestSplit = clo[ax] + ext * s / kBins; }
        }
    }
    int mid;
    if (bestAxis < 0) mid = begin + count / 2;
    else {
        auto it = std::partition(t.begin() + begin, t.begin() + end, [&](const Tri& x) { return x.c[bestAxis] < bestSplit; });
        mid = int(it - t.begin());
        if (mid == begin || mid == end) mid = begin + count / 2;
    }
    const int left = build(nodes, t, begin, mid);
    const int right = build(nodes, t, mid, end);
    nd.a = left; nd.b = right; nodes[me] = nd;
    return me;
}

static std::vector<char> readAll(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s geo.bin rays.bin nvrtc_bin_dir out.csv\n", argv[0]); return 2; }
    try {
        // Geometry.
        const std::vector<char> geo = readAll(argv[1]);
        std::vector<Tri> tris;
        size_t i = 0;
        while (i + 12 <= geo.size()) {
            int32_t head[3]; std::memcpy(head, &geo[i], 12); i += 12;
            const float* v = reinterpret_cast<const float*>(&geo[i]); i += size_t(head[1]) * 12;
            const uint32_t* f = reinterpret_cast<const uint32_t*>(&geo[i]); i += size_t(head[2]) * 12;
            const int32_t* fm = reinterpret_cast<const int32_t*>(&geo[i]); i += size_t(head[2]) * 4;
            for (int k = 0; k < head[2]; ++k) {
                Tri t{};
                t.mat = fm[k];
                for (int c = 0; c < 3; ++c) for (int a = 0; a < 3; ++a) t.v[3*c+a] = v[3*f[3*k+c]+a];
                for (int a = 0; a < 3; ++a) {
                    t.lo[a] = std::min({t.v[a], t.v[3+a], t.v[6+a]}); t.hi[a] = std::max({t.v[a], t.v[3+a], t.v[6+a]});
                    t.c[a] = 0.5f * (t.lo[a] + t.hi[a]);
                }
                tris.push_back(t);
            }
        }
        const auto b0 = std::chrono::steady_clock::now();
        std::vector<Node> nodes; nodes.reserve(tris.size());
        build(nodes, tris, 0, int(tris.size()));
        const double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();
        std::vector<float> flat(tris.size() * 9);
        std::vector<int> mats(tris.size());
        for (size_t k = 0; k < tris.size(); ++k) { std::memcpy(&flat[9*k], tris[k].v, 36); mats[k] = tris[k].mat; }
        const std::vector<char> raysRaw = readAll(argv[2]);
        constexpr size_t kRec = 64;
        const size_t nRays = raysRaw.size() / kRec;
        std::printf("triangles %zu nodes %zu build %.1f ms rays %zu\n", tris.size(), nodes.size(), buildMs, nRays);

        // NVRTC compile.
        const std::string nv = argv[3];
        SetDllDirectoryA(nv.c_str());
        HMODULE nvrtc = LoadLibraryA((nv + "\\nvrtc64_120_0.dll").c_str());
        if (!nvrtc) throw std::runtime_error("nvrtc dll");
        auto create = (NvrtcCreate)GetProcAddress(nvrtc, "nvrtcCreateProgram");
        auto compile = (NvrtcCompile)GetProcAddress(nvrtc, "nvrtcCompileProgram");
        auto ptxSize = (NvrtcSize)GetProcAddress(nvrtc, "nvrtcGetPTXSize");
        auto getPtx = (NvrtcGet)GetProcAddress(nvrtc, "nvrtcGetPTX");
        auto logSize = (NvrtcSize)GetProcAddress(nvrtc, "nvrtcGetProgramLogSize");
        auto getLog = (NvrtcGet)GetProcAddress(nvrtc, "nvrtcGetProgramLog");
        void* prog = nullptr;
        if (create(&prog, kKernel, "missproof.cu", 0, nullptr, nullptr)) throw std::runtime_error("nvrtcCreateProgram");
        const char* opts[] = {"--gpu-architecture=compute_86", "--std=c++14"};
        const int cr = compile(prog, 2, opts);
        size_t ls = 0; logSize(prog, &ls); std::string log(ls, '\0'); getLog(prog, &log[0]);
        if (cr) throw std::runtime_error("nvrtc compile: " + log);
        size_t ps = 0; ptxSize(prog, &ps); std::string ptx(ps, '\0'); getPtx(prog, &ptx[0]);

        // CUDA.
        HMODULE lib = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!lib) throw std::runtime_error("nvcuda.dll");
#define LOAD(n) p_##n = reinterpret_cast<decltype(p_##n)>(GetProcAddress(lib, #n)); if (!p_##n) throw std::runtime_error(#n);
        LOAD(cuInit) LOAD(cuDeviceGet) LOAD(cuDevicePrimaryCtxRetain) LOAD(cuCtxSetCurrent) LOAD(cuStreamCreate)
        LOAD(cuMemAlloc_v2) LOAD(cuMemcpyHtoD_v2) LOAD(cuMemHostAlloc) LOAD(cuMemHostGetDevicePointer_v2)
        LOAD(cuModuleLoadData) LOAD(cuModuleGetFunction) LOAD(cuLaunchKernel) LOAD(cuStreamSynchronize)
#undef LOAD
        check(p_cuInit(0), "cuInit");
        CUdevice dev; check(p_cuDeviceGet(&dev, 0), "device");
        CUcontext ctx; check(p_cuDevicePrimaryCtxRetain(&ctx, dev), "ctx"); check(p_cuCtxSetCurrent(ctx), "current");
        CUstream stream; check(p_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "stream");
        CUmodule mod; check(p_cuModuleLoadData(&mod, ptx.c_str()), "module");
        CUfunction fn; check(p_cuModuleGetFunction(&fn, mod, "missproof"), "function");
        CUdeviceptr dNodes, dTris, dMats;
        check(p_cuMemAlloc_v2(&dNodes, nodes.size() * sizeof(Node)), "alloc"); check(p_cuMemcpyHtoD_v2(dNodes, nodes.data(), nodes.size() * sizeof(Node)), "nodes");
        check(p_cuMemAlloc_v2(&dTris, flat.size() * 4), "alloc"); check(p_cuMemcpyHtoD_v2(dTris, flat.data(), flat.size() * 4), "tris");
        check(p_cuMemAlloc_v2(&dMats, mats.size() * 4), "alloc"); check(p_cuMemcpyHtoD_v2(dMats, mats.data(), mats.size() * 4), "mats");
        constexpr size_t kMaxBatch = 8192;
        void *hRays, *hOut; CUdeviceptr mRays, mOut;
        check(p_cuMemHostAlloc(&hRays, kRec * kMaxBatch, CU_MEMHOSTALLOC_DEVICEMAP), "mapped");
        check(p_cuMemHostAlloc(&hOut, kMaxBatch, CU_MEMHOSTALLOC_DEVICEMAP), "mapped");
        check(p_cuMemHostGetDevicePointer_v2(&mRays, hRays, 0), "dptr"); check(p_cuMemHostGetDevicePointer_v2(&mOut, hOut, 0), "dptr");

        FILE* out = std::fopen(argv[4], "a");
        for (int useMaterial : {0, 10}) for (float margin : {1e-5f}) {
            for (unsigned batch : {1u, 32u, 512u, 2048u, 8192u}) {
                uint64_t fn_ = 0, fp = 0, tp = 0, tn = 0; std::vector<double> times;
                const size_t limit = batch < 512 ? std::min<size_t>(nRays, size_t(batch) * 3000) : nRays;
                for (size_t start = 0; start + batch <= limit; start += batch) {
                    std::memcpy(hRays, &raysRaw[start * kRec], size_t(batch) * kRec);  // as if written in place by the walks
                    unsigned n = batch; float mg = margin;
                    int um = useMaterial;
                    void* args[] = {&mRays, &dNodes, &dTris, &dMats, &mOut, &n, &mg, &um};
                    const auto a = std::chrono::steady_clock::now();
                    check(p_cuLaunchKernel(fn, (n + 127) / 128, 1, 1, 128, 1, 1, 0, stream, args, nullptr), "launch");
                    check(p_cuStreamSynchronize(stream), "sync");
                    times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
                    const unsigned char* flags = static_cast<const unsigned char*>(hOut);
                    for (unsigned k = 0; k < n; ++k) {
                        int32_t hit; std::memcpy(&hit, &raysRaw[(start + k) * kRec + 32], 4);
                        if (hit && !flags[k]) ++fn_; else if (!hit && flags[k]) ++fp; else if (hit) ++tp; else ++tn;
                    }
                }
                std::vector<double> s = times; std::sort(s.begin(), s.end());
                double sum = 0; for (double x : times) sum += x;
                const double perRay = sum / (times.size() * double(batch));
                std::printf("material %d batch %5u calls %5zu median %7.1f us  p90 %7.1f us  max %8.1f us  mean/ray %.3f us  "
                            "missed_hits %llu  extra_candidates %llu (%.2f%% of misses)  proven_misses %llu\n",
                            useMaterial, batch, times.size(), s[s.size()/2], s[s.size()*9/10], s.back(), perRay,
                            (unsigned long long)fn_, (unsigned long long)fp, 100.0 * fp / std::max<uint64_t>(1, fp + tn),
                            (unsigned long long)tn);
                if (out) std::fprintf(out, "%d,%u,%zu,%.2f,%.2f,%.4f,%llu,%llu,%llu,%llu\n", useMaterial, batch, times.size(),
                                      s[s.size()/2], s[s.size()*9/10], perRay, (unsigned long long)fn_, (unsigned long long)fp,
                                      (unsigned long long)tp, (unsigned long long)tn);
            }
        }
        if (out) std::fclose(out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

// Feasibility diagnostic for "find on the GPU, confirm on the CPU" (scalar
// Scene::intersectRay calls, mostly from subsurface scattering). Read-only:
// it never changes a ray's result. Inserted into copied sources only.
//
// For a sample of rays that Embree answered:
//  * confirm: re-intersect Embree's hit triangle alone, in a one-triangle
//    Embree scene on the same device, and compare t, u, v and Ng bit for bit;
//  * gpu: send the same rays (with their original tfar) to XPU's GPU
//    accelerator in batches and compare hit/miss, geometry, primitive and
//    t, u, v with Embree's answer.
#pragma once
#include <embree4/rtcore.h>
#include <moonray/rendering/rt/gpu/GPURay.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace sssprobe {
enum Origin { Outside = 0, Probe, Back, Walk, Other, OriginCount };
inline const char* const originNames[OriginCount] = {"outside_subsurface", "sss_probe", "sss_back", "sss_walk", "sss_other"};

struct Counts {
    uint64_t rays = 0, hits = 0, embreeNs = 0, timed = 0, occlusionRays = 0;
    uint64_t sampled = 0;
    // confirmation
    uint64_t confirmExact = 0, confirmDiffer = 0, confirmMiss = 0, skipInstance = 0, skipQuad = 0,
             skipMotion = 0, skipOther = 0;
    // GPU agreement
    uint64_t gpuChecked = 0, gpuSameExact = 0, gpuSameTriangle = 0, gpuOtherTriangle = 0,
             gpuMissEmbreeHit = 0, gpuHitEmbreeMiss = 0, gpuBothMiss = 0;
    // Embree cost split, and the shortened-ray trace [tnear, t + margin]
    uint64_t hitNs = 0, hitTimed = 0, missNs = 0, missTimed = 0;
    uint64_t shortExact = 0, shortDiffer = 0, shortNs = 0, shortOtherPrim = 0, shortMiss = 0;
};

// Whole-frame histograms (not per origin).
constexpr int kBins = 10;
struct Hist {
    uint64_t stepsPerWalk[kBins] = {};      // 1, 2, 3-4, 5-8, ..., 257+
    uint64_t walksPerCall[kBins] = {};      // same bins
    uint64_t waveRays[kBins] = {};          // rays in waves of size 1, 2, 3-4, ... (if all walks of a call stepped together)
    uint64_t waveTrips[kBins] = {};         // number of such waves
    uint64_t gpuNs[4] = {}, gpuCalls[4] = {}; // batch sizes 1, 512, 2048, 8192
    uint64_t tRelErr[6] = {};               // <1e-7, <1e-6, <1e-5, <1e-4, <1e-3, larger
    double tRelMax = 0, tAbsMax = 0;
    uint64_t calls = 0, callNs = 0, walks = 0;
};
inline int bin(uint64_t n) { int b = 0; uint64_t edge = 1; while (b < kBins - 1 && n > edge) { edge *= 2; ++b; } return b; }

struct Slot {
    Counts c[OriginCount];
    // Pending GPU batch: rays and Embree's answers.
    struct Pending { int origin; float tfar, u, v; int geomID, primID; bool instance; };
    std::vector<moonray::rt::GPURay> rays;
    std::vector<Pending> pending;
    Hist h;
    std::vector<int> callSteps;
    std::vector<char> rayDump;  // 48-byte walk-ray records, written at exit
    uint64_t dumpCounter = 0;
    int curSteps = -1;
    uint64_t callStart = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<Slot*> slots;
    ~Registry() {
        const char* path = std::getenv("XPU2_SSS_PROBE");
        if (!path) return;
        FILE* f = std::fopen(path, "w");
        if (!f) return;
        std::fprintf(f, "origin,rays,hits,embree_ns,timed,occlusion_rays,sampled,confirm_exact,confirm_differ,"
                        "confirm_miss,skip_instance,skip_quad,skip_motion,skip_other,gpu_checked,gpu_same_exact,"
                        "gpu_same_triangle,gpu_other_triangle,gpu_miss_embree_hit,gpu_hit_embree_miss,gpu_both_miss,"
                        "hit_ns,hit_timed,miss_ns,miss_timed,short_exact,short_differ,short_ns,short_other_prim,short_miss\n");
        for (int o = 0; o < OriginCount; ++o) {
            Counts t;
            for (Slot* s : slots) {
                const Counts& c = s->c[o];
                uint64_t* dst = reinterpret_cast<uint64_t*>(&t);
                const uint64_t* src = reinterpret_cast<const uint64_t*>(&c);
                for (size_t k = 0; k < sizeof(Counts) / sizeof(uint64_t); ++k) dst[k] += src[k];
            }
            const uint64_t* v = reinterpret_cast<const uint64_t*>(&t);
            std::fprintf(f, "%s", originNames[o]);
            for (size_t k = 0; k < sizeof(Counts) / sizeof(uint64_t); ++k) std::fprintf(f, ",%llu", (unsigned long long)v[k]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        writeHistograms(std::string(path) + ".hist.csv");
        if (const char* rd = std::getenv("XPU2_SSS_RAYDUMP")) {
            if (FILE* r = std::fopen(rd, "wb")) {
                for (Slot* sl : slots) if (!sl->rayDump.empty()) std::fwrite(sl->rayDump.data(), 1, sl->rayDump.size(), r);
                std::fclose(r);
            }
        }
    }
    void writeHistograms(const std::string& hpath) {
        FILE* g = std::fopen(hpath.c_str(), "w");
        if (!g) return;
        Hist t;
        for (Slot* sl : slots) {
            const Hist& h = sl->h;
            for (int b = 0; b < kBins; ++b) {
                t.stepsPerWalk[b] += h.stepsPerWalk[b]; t.walksPerCall[b] += h.walksPerCall[b];
                t.waveRays[b] += h.waveRays[b]; t.waveTrips[b] += h.waveTrips[b];
            }
            for (int b = 0; b < 4; ++b) { t.gpuNs[b] += h.gpuNs[b]; t.gpuCalls[b] += h.gpuCalls[b]; }
            for (int b = 0; b < 6; ++b) t.tRelErr[b] += h.tRelErr[b];
            t.tRelMax = std::max(t.tRelMax, h.tRelMax); t.tAbsMax = std::max(t.tAbsMax, h.tAbsMax);
            t.calls += h.calls; t.callNs += h.callNs; t.walks += h.walks;
        }
        std::fprintf(g, "name,bin,value\n");
        const char* edges[kBins] = {"1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257+"};
        for (int b = 0; b < kBins; ++b) {
            std::fprintf(g, "steps_per_walk,%s,%llu\n", edges[b], (unsigned long long)t.stepsPerWalk[b]);
            std::fprintf(g, "walks_per_call,%s,%llu\n", edges[b], (unsigned long long)t.walksPerCall[b]);
            std::fprintf(g, "wave_rays,%s,%llu\n", edges[b], (unsigned long long)t.waveRays[b]);
            std::fprintf(g, "wave_trips,%s,%llu\n", edges[b], (unsigned long long)t.waveTrips[b]);
        }
        const int sizes[4] = {-512, 1, 512, 8192}; // -512: first call after idle; 1 and 512: steady back to back
        for (int b = 0; b < 4; ++b)
            std::fprintf(g, "gpu_batch_ns,%d,%.1f\n", sizes[b], t.gpuCalls[b] ? double(t.gpuNs[b]) / t.gpuCalls[b] : 0.0);
        const char* terr[6] = {"<1e-7", "<1e-6", "<1e-5", "<1e-4", "<1e-3", ">=1e-3"};
        for (int b = 0; b < 6; ++b) std::fprintf(g, "t_rel_err,%s,%llu\n", terr[b], (unsigned long long)t.tRelErr[b]);
        std::fprintf(g, "t_rel_max,-,%.9g\nt_abs_max,-,%.9g\n", t.tRelMax, t.tAbsMax);
        std::fprintf(g, "calls,-,%llu\ncall_ns,-,%llu\nwalks,-,%llu\n", (unsigned long long)t.calls,
                     (unsigned long long)t.callNs, (unsigned long long)t.walks);
        std::fclose(g);
    }
};
inline Registry& registry() { static Registry r; return r; }
inline Slot& slot() {
    thread_local Slot* s = [] {
        auto* p = new Slot;
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        r.slots.push_back(p);
        return p;
    }();
    return *s;
}
inline bool enabled() { static const bool e = std::getenv("XPU2_SSS_PROBE") != nullptr; return e; }
inline uint64_t rate() {
    static const uint64_t r = [] {
        const char* v = std::getenv("XPU2_SSS_PROBE_RATE");
        return v ? (std::strtoull(v, nullptr, 10) ? std::strtoull(v, nullptr, 10) : 1) : 64;
    }();
    return r;
}
inline thread_local int depth = 0;   // > 0 inside CPP_computeRadianceSubsurface
inline thread_local int tagged = 0;  // origin of the next Scene::intersectRay call
inline uint64_t now();
// One CPP_computeRadianceSubsurface call: walks and their steps, for wave sizes.
inline void endWalk(struct Slot& s);
struct Scope {
    Scope() { ++depth; if (enabledFlag()) { Slot& s = slot(); s.callSteps.clear(); s.curSteps = -1; s.callStart = now(); } }
    ~Scope();
    static bool enabledFlag();
};
inline void walkBegin() {
    if (!Scope::enabledFlag()) return;
    Slot& s = slot();
    endWalk(s);
    s.curSteps = 0;
}
template <class T> inline T* tag(Origin o, T* scene) { tagged = o; return scene; }
inline int take() {
    const int t = tagged; tagged = 0;
    if (depth <= 0) return Outside;
    return t ? t : Other;
}
inline uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
inline bool Scope::enabledFlag() { return enabled(); }
inline void endWalk(Slot& s) {
    if (s.curSteps > 0) { s.callSteps.push_back(s.curSteps); ++s.h.stepsPerWalk[bin(s.curSteps)]; ++s.h.walks; }
    s.curSteps = -1;
}
inline Scope::~Scope() {
    --depth;
    if (!enabled()) return;
    Slot& s = slot();
    endWalk(s);
    ++s.h.calls; s.h.callNs += now() - s.callStart;
    if (s.callSteps.empty()) return;
    ++s.h.walksPerCall[bin(s.callSteps.size())];
    std::vector<int> st = s.callSteps;
    std::sort(st.begin(), st.end());
    // Wave k holds every walk with more than k steps.
    size_t i = 0;
    for (int k = 0; k < st.back(); ++k) {
        while (i < st.size() && st[i] <= k) ++i;
        const uint64_t active = st.size() - i;
        if (!active) break;
        s.h.waveRays[bin(active)] += active; ++s.h.waveTrips[bin(active)];
    }
}
} // namespace sssprobe

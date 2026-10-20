// Switch and shadow-mode comparison for the random-walk subsurface wavefront.
// XPU2_SSS_WAVEFRONT: unset or "0" = MoonRay's original walk loop (XPU exactly),
// "1" = wavefront, "shadow" = run the original for real, then the wavefront from
// the same starting state into scratch copies, and compare every output bit for
// bit. XPU2_SSS_WAVEFRONT_REPORT names a CSV written at exit.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace xpu2sss {

// 0 original, 1 per-lane wavefront, 2 per-lane shadow, 3 merged lanes, 4 merged-lane shadow.
inline int mode()
{
    static const int m = [] {
        const char* v = std::getenv("XPU2_SSS_WAVEFRONT");
        if (!v) return 0;
        if (!std::strcmp(v, "1")) return 1;
        if (!std::strcmp(v, "shadow")) return 2;
        if (!std::strcmp(v, "merged")) return 3;
        if (!std::strcmp(v, "merged-shadow")) return 4;
        return 0;
    }();
    return m;
}

// One-shot phase for the next computeRadiancePathTraceSubsurface call, set by
// CPP_computeRadianceSubsurface in merged mode. Read and cleared on entry, so
// nested calls (from exit shading) run normally.
enum Phase { None = 0, Prepare, Waves, Finish };
inline thread_local int phase = None;
inline thread_local int phaseLane = -1;
inline thread_local void* batch = nullptr;
inline int takePhase() { const int p = phase; phase = None; return p; }

struct Report {
    std::atomic<uint64_t> calls{0}, walks{0}, waves{0}, radianceDiff{0}, ssAovDiff{0}, sequenceDiff{0},
                          aovDiff{0}, subpixelDiff{0}, anyDiff{0};
    std::atomic<uint64_t> mergedCalls{0}, mergedFallback{0}, mergedWaves{0}, mergedRays{0}, pathVertexDiff{0};
    std::atomic<uint64_t> waveRays[10] = {}, waveCount[10] = {};   // by wave size: 1, 2, 3-4, ..., 257+
    static int bin(uint64_t n) { int b = 0; uint64_t e = 1; while (b < 9 && n > e) { e *= 2; ++b; } return b; }
    std::mutex mutex;
    int dumped = 0;
    ~Report() {
        const char* path = std::getenv("XPU2_SSS_WAVEFRONT_REPORT");
        if (!path) return;
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "mode,calls,walks,waves,radiance_diff,ssaov_diff,sequence_diff,aov_diff,subpixel_diff,any_diff\n"
                            "%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", mode(),
                         (unsigned long long)calls, (unsigned long long)walks, (unsigned long long)waves,
                         (unsigned long long)radianceDiff, (unsigned long long)ssAovDiff,
                         (unsigned long long)sequenceDiff, (unsigned long long)aovDiff,
                         (unsigned long long)subpixelDiff, (unsigned long long)anyDiff);
            std::fprintf(f, "merged_calls,merged_fallback,merged_waves,merged_rays,pathvertex_diff\n%llu,%llu,%llu,%llu,%llu\n",
                         (unsigned long long)mergedCalls, (unsigned long long)mergedFallback,
                         (unsigned long long)mergedWaves, (unsigned long long)mergedRays, (unsigned long long)pathVertexDiff);
            const char* edges[10] = {"1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257+"};
            std::fprintf(f, "wave_size,waves,rays\n");
            for (int b = 0; b < 10; ++b)
                std::fprintf(f, "%s,%llu,%llu\n", edges[b], (unsigned long long)waveCount[b], (unsigned long long)waveRays[b]);
            std::fclose(f);
        }
    }
};
inline Report& report() { static Report r; return r; }

template <class Color, class Subpixel>
inline void compare(const Color& r0, const Color& r1, const Color& ss0, const Color& ss1,
                    unsigned q0, unsigned q1, const float* a0, const float* a1, int nAov,
                    const Subpixel& sp0, const Subpixel& sp1)
{
    Report& rep = report();
    ++rep.calls;
    const bool dr = std::memcmp(&r0, &r1, sizeof(Color)) != 0;
    const bool ds = std::memcmp(&ss0, &ss1, sizeof(Color)) != 0;
    const bool dq = q0 != q1;
    const bool da = nAov > 0 && std::memcmp(a0, a1, sizeof(float) * size_t(nAov)) != 0;
    const bool dp = std::memcmp(&sp0, &sp1, sizeof(Subpixel)) != 0;
    rep.radianceDiff += dr; rep.ssAovDiff += ds; rep.sequenceDiff += dq; rep.aovDiff += da; rep.subpixelDiff += dp;
    if (dr || ds || dq || da || dp) {
        ++rep.anyDiff;
        std::lock_guard<std::mutex> lock(rep.mutex);
        if (rep.dumped++ < 20) {
            const float* x = reinterpret_cast<const float*>(&r0);
            const float* y = reinterpret_cast<const float*>(&r1);
            std::fprintf(stderr, "XPU2_SSS_DIFF radiance %d (%.9g %.9g %.9g | %.9g %.9g %.9g) ssaov %d seq %d (%u|%u) aov %d subpixel %d\n",
                         int(dr), x[0], x[1], x[2], y[0], y[1], y[2], int(ds), int(dq), q0, q1, int(da), int(dp));
        }
    }
}

} // namespace xpu2sss

namespace moonray { namespace pbr {
// Owned by PathIntegratorSubsurface.cc (merged-lane jobs).
void* xpu2sssCreateBatch();
void xpu2sssDestroyBatch(void* batch);
} }

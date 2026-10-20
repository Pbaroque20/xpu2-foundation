// Sampled, exclusive stage timing for geom::initIntersectionPhase1 and mesh
// postIntersect. Diagnostic only: inserted into copied sources, never shipped.
// Each sampled call records consecutive timestamps, so stages never overlap.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace isectprof {
enum Stage {
    InitP, PostInclusive, Curvature, Validate, Rest,                 // phase 1
    MTable, MRequired, MGeometry, MToon, MDiffGeo, MAttrDeriv, MTail, // mesh postIntersect
    Phase2,
    MInstance, MSt, MExplicit,                                         // split of the old mesh_st_n_dpds
    CSt, CNormal, CDeriv,                                              // inside TriMesh curvature
    StageCount
};
inline const char* const names[StageCount] = {
    "init_setP", "post_intersect_inclusive", "curvature", "validate", "rest",
    "mesh_table", "mesh_required_attributes", "mesh_dpds", "mesh_toon_offset",
    "mesh_differential_geometry", "mesh_attribute_derivatives", "mesh_tail", "phase2",
    "mesh_instance_override", "mesh_st", "mesh_ng_explicit",
    "curv_st", "curv_normal", "curv_derivatives"};

struct Slot {
    uint64_t ns[StageCount] = {}, samples[StageCount] = {};
    uint64_t calls = 0, sampled = 0, tri = 0, quad = 0, other = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<Slot*> slots;
    ~Registry() {
        const char* path = std::getenv("XPU2_ISECT_PROFILE");
        if (!path) return;
        FILE* f = std::fopen(path, "w");
        if (!f) return;
        std::fprintf(f, "thread,stage,ns,samples,calls,sampled,tri,quad,other\n");
        for (size_t t = 0; t < slots.size(); ++t) {
            const Slot& s = *slots[t];
            for (int k = 0; k < StageCount; ++k)
                std::fprintf(f, "%zu,%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", t, names[k],
                             (unsigned long long)s.ns[k], (unsigned long long)s.samples[k],
                             (unsigned long long)s.calls, (unsigned long long)s.sampled,
                             (unsigned long long)s.tri, (unsigned long long)s.quad,
                             (unsigned long long)s.other);
        }
        std::fclose(f);
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
inline bool enabled() { static const bool e = std::getenv("XPU2_ISECT_PROFILE") != nullptr; return e; }
inline uint64_t rate() {
    static const uint64_t r = [] {
        const char* v = std::getenv("XPU2_ISECT_PROFILE_RATE");
        return v ? std::max<uint64_t>(1, std::strtoull(v, nullptr, 10)) : 16;
    }();
    return r;
}
inline thread_local bool sampled = false;
inline uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Called once per phase-1 call: counts every call, samples one in rate().
inline bool begin() {
    if (!enabled()) return sampled = false;
    Slot& s = slot();
    sampled = (++s.calls % rate()) == 0;
    if (sampled) ++s.sampled;
    return sampled;
}
struct Marks {
    uint64_t last = 0;
    bool on;
    Marks() : on(sampled) { if (on) last = now(); }
    void mark(Stage stage) {
        if (!on) return;
        const uint64_t t = now();
        Slot& s = slot();
        s.ns[stage] += t - last; ++s.samples[stage];
        last = t;
    }
    void skip() { if (on) last = now(); }
};
inline void primitive(int kind) {
    if (!sampled) return;
    Slot& s = slot();
    (kind == 0 ? s.tri : kind == 1 ? s.quad : s.other)++;
}
} // namespace isectprof

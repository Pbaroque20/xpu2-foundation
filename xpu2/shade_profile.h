// Sampled stage timing for pbr::shadeBundleHandler. Diagnostic only: inserted
// into a copied source, never shipped. One call in XPU2_SHADE_PROFILE_RATE
// (default 8) records consecutive timestamps; XPU2_SHADE_PROFILE names the CSV
// written at exit. Stages marked "incl" may run other handlers through queue
// flushes (and nested shade-handler calls), so they are inclusive.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace shadeprof {
enum Stage {
    Prelude, Alloc, InitIsect, EntryMisc, Derivs, SortKey, Aovs, Free1, Sort, TerminatedIncl,
    ChunkAlloc, ChunkSetup, Resort, Presence, LightCulling, IsectSoa, BsdfClear, Shade, AovExtra,
    PresenceRaysIncl, RsSoa, IntegrateIncl, Free2, Tail,
    StageCount
};
inline const char* const names[StageCount] = {
    "prelude", "alloc", "init_intersection", "entry_misc", "derivatives", "sort_key", "aovs", "free_terminated",
    "sort", "terminated_queue_incl", "chunk_alloc", "chunk_setup", "resort", "presence", "light_culling",
    "isect_aos_to_soa", "bsdf_clear", "shadev", "aov_extra", "presence_rays_incl", "raystate_aos_to_soa",
    "integrate_incl", "free_raystates", "tail"};

struct Slot {
    uint64_t ns[StageCount] = {}, samples[StageCount] = {};
    uint64_t calls = 0, sampled = 0, entries = 0, sampledEntries = 0, chunks = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<Slot*> slots;
    ~Registry() {
        const char* path = std::getenv("XPU2_SHADE_PROFILE");
        if (!path) return;
        FILE* f = std::fopen(path, "w");
        if (!f) return;
        std::fprintf(f, "thread,stage,ns,samples,calls,sampled,entries,sampled_entries,chunks\n");
        for (size_t t = 0; t < slots.size(); ++t) {
            const Slot& s = *slots[t];
            for (int k = 0; k < StageCount; ++k)
                std::fprintf(f, "%zu,%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", t, names[k],
                             (unsigned long long)s.ns[k], (unsigned long long)s.samples[k],
                             (unsigned long long)s.calls, (unsigned long long)s.sampled,
                             (unsigned long long)s.entries, (unsigned long long)s.sampledEntries,
                             (unsigned long long)s.chunks);
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
inline bool enabled() { static const bool e = std::getenv("XPU2_SHADE_PROFILE") != nullptr; return e; }
inline uint64_t rate() {
    static const uint64_t r = [] {
        const char* v = std::getenv("XPU2_SHADE_PROFILE_RATE");
        return v ? std::max<uint64_t>(1, std::strtoull(v, nullptr, 10)) : 8;
    }();
    return r;
}
inline uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Marks {
    uint64_t last = 0;
    bool on = false;
    Marks(unsigned numEntries) {
        if (!enabled()) return;
        Slot& s = slot();
        s.entries += numEntries;
        on = (++s.calls % rate()) == 0;
        if (on) { ++s.sampled; s.sampledEntries += numEntries; last = now(); }
    }
    void mark(Stage stage) {
        if (!on) return;
        const uint64_t t = now();
        Slot& s = slot();
        s.ns[stage] += t - last; ++s.samples[stage];
        last = t;
    }
    void chunk() { if (on) ++slot().chunks; }
};
} // namespace shadeprof

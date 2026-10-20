// Sampled stage timing for bundled integration (PathIntegrator_integrateBundled and
// computeRadianceBsdfMultiSampler), called from marks inserted into copied ISPC
// sources by xpu2/rndr_integ_profile.py. Diagnostic only, never shipped.
// XPU2_INTEG_PROFILE names the CSV written at exit; one call in
// XPU2_INTEG_PROFILE_RATE (default 4) is timed, at any nesting depth. A queue
// flush can run other handlers and nested integration calls: the nested calls'
// time is taken out of the parent's stage (they are timed themselves), the other
// handlers' time stays in it, so stages that flush end in "_incl".
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace {
const char* const kNames[] = {
    "prelude", "loop_top", "camera_light_hit", "volumes_cutout", "self_emission", "setup", "active_lights",
    "subsurface_incl", "emissive_regions", "fill_radiance", "no_lobe_aovs", "loop_tail", "radiance_queue_incl",
    "ms_enter", "ms_bsdf_sampler_init", "ms_light_sampler_init", "ms_draw_bsdf_samples", "ms_material_aovs",
    "ms_russian_roulette", "ms_direct_light", "ms_indirect_rest", "ms_occlusion_queue_incl",
    "ind_spawn", "ind_occl_queue_incl", "ind_ray_queue_incl",
    "dl_light_select", "ls_prologue", "ls_seq_setup", "ls_xpu2_setup", "ls_sample_eval", "ls_integrate", "ls_tail",
    "dl_rr", "dl_occl_prepare", "bs_prologue", "bs_sample", "bs_light_hit", "bs_integrate",
    "wrapper_tail"};
constexpr int kStages = int(sizeof(kNames) / sizeof(kNames[0]));

struct Slot {
    uint64_t ns[kStages] = {}, hits[kStages] = {};
    uint64_t calls = 0, sampled = 0, entries = 0, sampledEntries = 0;
};
struct Registry {
    std::mutex mutex;
    std::vector<Slot*> slots;
    ~Registry()
    {
        const char* path = std::getenv("XPU2_INTEG_PROFILE");
        if (!path) return;
        FILE* f = std::fopen(path, "w");
        if (!f) return;
        std::fprintf(f, "thread,stage,ns,hits,calls,sampled,entries,sampled_entries\n");
        for (size_t t = 0; t < slots.size(); ++t)
            for (int k = 0; k < kStages; ++k)
                std::fprintf(f, "%zu,%s,%llu,%llu,%llu,%llu,%llu,%llu\n", t, kNames[k],
                             (unsigned long long)slots[t]->ns[k], (unsigned long long)slots[t]->hits[k],
                             (unsigned long long)slots[t]->calls, (unsigned long long)slots[t]->sampled,
                             (unsigned long long)slots[t]->entries, (unsigned long long)slots[t]->sampledEntries);
        std::fclose(f);
    }
};
Registry& registry() { static Registry r; return r; }
Slot& slot()
{
    thread_local Slot* s = [] {
        auto* p = new Slot;
        std::lock_guard<std::mutex> lock(registry().mutex);
        registry().slots.push_back(p);
        return p;
    }();
    return *s;
}
bool enabled() { static const bool e = std::getenv("XPU2_INTEG_PROFILE") != nullptr; return e; }
uint64_t rate()
{
    static const uint64_t r = [] {
        const char* v = std::getenv("XPU2_INTEG_PROFILE_RATE");
        return v ? std::max<uint64_t>(1, std::strtoull(v, nullptr, 10)) : 4;
    }();
    return r;
}
uint64_t now()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Frame { bool timing = false; uint64_t enter = 0, last = 0, nested = 0; };
constexpr int kMaxDepth = 64;
thread_local Frame frames[kMaxDepth + 1];
thread_local int depth = 0;
} // namespace

extern "C" void XPU2_integBegin(unsigned numEntries)
{
    if (!enabled()) return;
    ++depth;
    if (depth > kMaxDepth) return;
    Slot& s = slot();
    s.entries += numEntries;
    Frame& f = frames[depth];
    f.timing = (++s.calls % rate()) == 0;
    if (f.timing) { ++s.sampled; s.sampledEntries += numEntries; }
    f.enter = f.last = now();
    f.nested = 0;
}

extern "C" void XPU2_integMark(int stage)
{
    if (depth < 1 || depth > kMaxDepth || stage < 0 || stage >= kStages) return;
    Frame& f = frames[depth];
    if (!f.timing) return;
    const uint64_t t = now();
    Slot& s = slot();
    s.ns[stage] += (t - f.last) - std::min(f.nested, t - f.last); ++s.hits[stage];
    f.last = t;
    f.nested = 0;
}

extern "C" void XPU2_integEnd()
{
    if (!enabled()) return;
    if (depth >= 1 && depth <= kMaxDepth) {
        XPU2_integMark(kStages - 1);
        const uint64_t spent = now() - frames[depth].enter;
        if (depth > 1) frames[depth - 1].nested += spent;   // the parent's stage excludes this call
    }
    --depth;
}

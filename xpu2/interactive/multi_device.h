// The multi-device extension: more than one processor group, more than one graphics card.
//
// It switches on only where the machine has them. With one processor group and one
// graphics card nothing here changes what XPU2 does: every function below answers
// as the single-device code did (spread() is false, logicalProcessors() is
// std::thread::hardware_concurrency(), cards() is 1).
//
// Processor groups. Windows numbers more than 64 logical processors, and the
// sockets of many two-socket machines, in groups of up to 64; on Windows 10 a
// process's threads stay in the group they started in unless placed. The CPU
// lane's workers are spread over every group in proportion to its processors
// (placeWorker); the feeder keeps its own processor, as it does with one group.
//
// Graphics cards. Each card past the first is a lane of its own (async_render.h,
// Renderer's extra lanes): its own CUDA context, its own copy of the scene, its own
// film. The cards claim chunks from the one grid as the first does, each chunk
// exactly once, and their films are added into the first card's host copy as it
// is read, so the previews, the picture and the adaptive errors hold every
// sample. XPU2_CARDS=N uses the first N cards (1: the first alone); unset, all.
#pragma once
#include <windows.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace xpu2::interactive::multi {

// The active processor groups and each one's processors.
struct Group {
    WORD number = 0;
    KAFFINITY mask = 0;
    unsigned processors = 0;
};

// Simulation, for a machine with one group and one card (the simulation app's): XPU2_SIMULATE_GROUPS=K splits the
// first group's processors into K groups of the same number, so the workers are placed as on K groups;
// XPU2_SIMULATE_CARDS=N makes N lanes on the first card, each with its own context, scene and film, fed as N cards.
// For those lanes: XPU2_SIMULATE_CARD_THREADS=a,b,... caps each lane's launch width (a smaller card), and
// XPU2_SIMULATE_CARD_FAIL=k fails lane k (1 or more) as it takes a scene, so it is left out.
inline unsigned simulated(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? unsigned(std::max(0, std::atoi(v))) : 0u;
}

inline const std::vector<Group>& groups() {
    static const std::vector<Group> found = [] {
        std::vector<Group> out;
        if (const unsigned k = simulated("XPU2_SIMULATE_GROUPS"); k > 1) {
            GROUP_AFFINITY a{};
            if (!GetThreadGroupAffinity(GetCurrentThread(), &a)) return out;
            std::vector<KAFFINITY> bits;
            for (unsigned b = 0; b < 8 * sizeof(KAFFINITY); ++b)
                if (a.Mask & (KAFFINITY(1) << b)) bits.push_back(KAFFINITY(1) << b);
            const unsigned n = std::min<unsigned>(k, unsigned(bits.size()));
            for (unsigned g = 0; g < n; ++g) {
                Group group{a.Group, 0, 0};
                for (size_t b = g * bits.size() / n; b < (g + 1) * bits.size() / n; ++b) {
                    group.mask |= bits[b];
                    ++group.processors;
                }
                out.push_back(group);
            }
            return out;
        }
        DWORD bytes = 0;
        GetLogicalProcessorInformationEx(RelationGroup, nullptr, &bytes);
        if (!bytes) return out;
        std::vector<char> buffer(bytes);
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
        if (!GetLogicalProcessorInformationEx(RelationGroup, info, &bytes) || info->Relationship != RelationGroup) return out;
        for (WORD g = 0; g < info->Group.ActiveGroupCount; ++g) {
            const PROCESSOR_GROUP_INFO& p = info->Group.GroupInfo[g];
            if (p.ActiveProcessorCount) out.push_back({g, p.ActiveProcessorMask, p.ActiveProcessorCount});
        }
        return out;
    }();
    return found;
}

// More than one processor group: the CPU lane's workers are spread over them.
inline bool spread() { return groups().size() > 1; }

// The logical processors XPU2 may use: every group's with more than one, else as before.
inline unsigned logicalProcessors() {
    if (!spread()) return std::max(1u, std::thread::hardware_concurrency());
    unsigned n = 0;
    for (const Group& g : groups()) n += g.processors;
    return std::max(1u, n);
}

// The group of the calling thread (the feeder's, where the CPU lane is made).
inline WORD currentGroup() {
    GROUP_AFFINITY a{};
    return GetThreadGroupAffinity(GetCurrentThread(), &a) ? a.Group : 0;
}

// Worker w of n placed on a group, the groups taking workers in proportion to their processors. In the feeder's own
// group (home) a worker keeps to homeMask where one is given (every processor but the feeder's), as with one group.
inline void placeWorker(unsigned w, unsigned n, WORD home, KAFFINITY homeMask) {
    const auto& all = groups();
    unsigned total = 0;
    for (const Group& g : all) total += g.processors;
    if (!total || !n) return;
    const unsigned slot = unsigned((unsigned long long)w * total / n);
    unsigned at = 0;
    for (const Group& g : all) {
        if (slot < at + g.processors) {
            GROUP_AFFINITY a{};
            a.Group = g.number;
            a.Mask = g.number == home && homeMask ? (g.mask & homeMask) : g.mask;
            if (a.Mask) SetThreadGroupAffinity(GetCurrentThread(), &a, nullptr);
            return;
        }
        at += g.processors;
    }
}

// Entry i of a comma-separated simulation list (0: none there).
inline unsigned simulatedEntry(const char* name, size_t i) {
    const char* v = std::getenv(name);
    for (size_t at = 0; v && *v; ++at) {
        if (at == i) return unsigned(std::max(0, std::atoi(v)));
        v = std::strchr(v, ',');
        if (v) ++v;
    }
    return 0;
}

// The graphics cards to use, of count found: XPU2_CARDS=N caps them (at least 1).
inline int cardsToUse(int found) {
    if (found < 1) return 0;
    const char* v = std::getenv("XPU2_CARDS");
    const int cap = v && *v ? std::atoi(v) : found;
    return std::max(1, std::min(found, cap));
}

// Each lane's card (CUDA's numbering), the first lane's first: the cards to use, or N lanes on the first card where
// XPU2_SIMULATE_CARDS=N.
inline std::vector<int> cardOrdinals(int found) {
    std::vector<int> out;
    if (found < 1) return out;
    if (const unsigned n = simulated("XPU2_SIMULATE_CARDS"); n > 1) return std::vector<int>(n, 0);
    for (int k = 0; k < cardsToUse(found); ++k) out.push_back(k);
    return out;
}

}  // namespace xpu2::interactive::multi

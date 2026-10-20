"""Exact CPU savings in pbr::shadeBundleHandler (copied source), each behind a switch.

XPU2_SHADE_PRESENCE_SKIP (on unless "0"): the stochastic presence test draws a
1D integrator sample and keeps the surface when sample < presence. Every such
sample is in [0, 1) (the precomputed table holds values in [3.5e-7,
0.99999994], the Cranley-Patterson rotation and the R1 and random-engine
fallbacks stay below 1), so a presence of 1 or more always gives 1 and a
presence of 0 or less always gives 0: the draw is skipped. Anything else,
including NaN, takes the original path. Nothing else reads the sample.

XPU2_SHADE_CHUNK_SCAN (on unless "0"): each chunk scans for the run of entries
with the same light-set assignment, to the end of the call, though only the
first shadingWorkloadChunkSize entries are used; the scan stops there. The
chunk size it yields is the same.

XPU2_SHADE_FIX_CHECK=1 still computes the original answer at every shortcut and
counts disagreements; XPU2_SHADE_FIX_REPORT names a CSV written at exit.

Both are applied wherever the code occurs (MoonRay's chunk loop, and the XPU2
GPU-owned copy in the candidate's file).
"""

FILES = ('pbr/handlers/ShadeBundleHandler.cc',)
HEADERS = ()

SWITCHES = r'''
#include <atomic>
#include <cstdio>
#include <cstdlib>
namespace {
bool xpu2ShadeSwitch(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    if (!v) return fallback;
    return !(v[0] == '0' && v[1] == 0);
}
const bool xpu2PresenceSkip = xpu2ShadeSwitch("XPU2_SHADE_PRESENCE_SKIP", true);
const bool xpu2ChunkScan = xpu2ShadeSwitch("XPU2_SHADE_CHUNK_SCAN", true);
const bool xpu2FixCheck = xpu2ShadeSwitch("XPU2_SHADE_FIX_CHECK", false);
struct Xpu2ShadeFixReport {
    std::atomic<unsigned long long> presenceShort{0}, presenceSampled{0}, presenceDiff{0},
                                    chunks{0}, chunkShort{0}, chunkDiff{0};
    ~Xpu2ShadeFixReport()
    {
        const char* path = std::getenv("XPU2_SHADE_FIX_REPORT");
        if (!path) return;
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "presence_skip,chunk_scan,check,presence_short,presence_sampled,presence_diff,"
                            "chunks,chunk_short,chunk_diff\n%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n",
                         int(xpu2PresenceSkip), int(xpu2ChunkScan), int(xpu2FixCheck),
                         presenceShort.load(), presenceSampled.load(), presenceDiff.load(),
                         chunks.load(), chunkShort.load(), chunkDiff.load());
            std::fclose(f);
        }
    }
};
Xpu2ShadeFixReport& xpu2ShadeFixReport() { static Xpu2ShadeFixReport r; return r; }
}
'''

PRESENCE = '''            if (luminance(rs->mPathVertex.pathThroughput) < (1.0f - fs.mPresenceQuality)) {
                SequenceIDIntegrator sidPresence('''
PRESENCE_NEW = '''            if (luminance(rs->mPathVertex.pathThroughput) < (1.0f - fs.mPresenceQuality)) {
                // Every 1D integrator sample is in [0, 1): at presence >= 1 or <= 0 the draw cannot matter.
                const float xpu2In = presences[i];
                const bool xpu2Short = xpu2PresenceSkip && (xpu2In >= 1.f || xpu2In <= 0.f);
                if (!xpu2Short || xpu2FixCheck) {
                SequenceIDIntegrator sidPresence('''
PRESENCE_END = '''                } else {
                    presences[i] = 0.f;
                }
            }
'''
PRESENCE_END_NEW = '''                } else {
                    presences[i] = 0.f;
                }
                }
                if (xpu2Short) {
                    const float xpu2Fast = xpu2In >= 1.f ? 1.f : 0.f;
                    if (xpu2FixCheck) {
                        xpu2ShadeFixReport().presenceDiff += presences[i] != xpu2Fast;
                    }
                    ++xpu2ShadeFixReport().presenceShort;
                    presences[i] = xpu2Fast;
                } else if (xpu2FixCheck) {
                    ++xpu2ShadeFixReport().presenceSampled;
                }
            }
'''

SCAN = '''        SortedEntry *currEntry = sortedEntries + 1;
        while (currEntry != endEntry && currLayerAssnId == currEntry->mLayerAssignmentId) {
            ++currEntry;
        }
'''
SCAN_NEW = '''        SortedEntry *currEntry = sortedEntries + 1;
        // Only the first shadingWorkloadChunkSize entries of the run are used.
        const SortedEntry *xpu2ScanEnd = xpu2ChunkScan && unsigned(endEntry - sortedEntries) > shadingWorkloadChunkSize
                                       ? sortedEntries + shadingWorkloadChunkSize : endEntry;
        while (currEntry != xpu2ScanEnd && currLayerAssnId == currEntry->mLayerAssignmentId) {
            ++currEntry;
        }
        if (xpu2FixCheck) {
            const SortedEntry *full = sortedEntries + 1;
            while (full != endEntry && currLayerAssnId == full->mLayerAssignmentId) {
                ++full;
            }
            ++xpu2ShadeFixReport().chunks;
            xpu2ShadeFixReport().chunkShort += xpu2ScanEnd != endEntry;
            xpu2ShadeFixReport().chunkDiff +=
                scene_rdl2::math::min(unsigned(full - sortedEntries), shadingWorkloadChunkSize) !=
                scene_rdl2::math::min(unsigned(currEntry - sortedEntries), shadingWorkloadChunkSize);
        }
'''


def _all(s, old, new, what):
    if s.count(old) < 1:
        raise ValueError('shade fix drift: ' + what)
    return s.replace(old, new)


def patch(sources):
    f = FILES[0]
    s = sources[f]
    s = _all(s, '\nnamespace moonray {\n', '\n' + SWITCHES + '\nnamespace moonray {\n', 'namespace')
    n = s.count(PRESENCE)
    s = _all(s, PRESENCE, PRESENCE_NEW, 'presence')
    if s.count(PRESENCE_END) != n:
        raise ValueError('shade fix drift: presence end')
    s = s.replace(PRESENCE_END, PRESENCE_END_NEW)
    s = _all(s, SCAN, SCAN_NEW, 'scan')
    return {f: s}

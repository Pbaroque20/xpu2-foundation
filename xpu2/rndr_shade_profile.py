"""Insert sampled stage marks (xpu2/shade_profile.h) into a copied ShadeBundleHandler.cc."""
from .native_gpu_paths import once

FILES = ('pbr/handlers/ShadeBundleHandler.cc',)
HEADERS = ('shade_profile.h',)

# (anchor, text inserted before it) in source order.
BEFORE = [
    ('    // Allocate temp working memory buffers', '    marks.mark(shadeprof::Prelude);\n'),
    ('    // If we have a ray that just came from a presence object,', '    marks.mark(shadeprof::Alloc);\n'),
    ('            // Save the t value of the intersection point', '            marks.mark(shadeprof::EntryMisc);\n'),
    ('            // state and prim attr aovs', '            marks.mark(shadeprof::SortKey);\n'),
    ('        sortedEntries[i].mRsIdx = rs - baseRayState;', '        marks.mark(shadeprof::Aovs);\n'),
    ('    numEntries -= numTerminated;', '    marks.mark(shadeprof::TerminatedIncl);\n'),
    ('    while (numRemaining) {', '    marks.mark(shadeprof::ChunkAlloc);\n'),
    ('        // Resort ray state pointers based on sort results.', '        marks.mark(shadeprof::ChunkSetup);\n        marks.chunk();\n'),
    ('            presences[i] = shading::presence(', '            marks.mark(shadeprof::Resort);\n'),
    ('            // Some NPR materials that want to allow', '            marks.mark(shadeprof::Presence);\n'),
    ('        // TODO: overlapping dielectrics.', '        marks.mark(shadeprof::LightCulling);\n'),
    ('        // Cutout materials in scalar will return immediately', '        marks.mark(shadeprof::Shade);\n'),
    ('        {\n            // ***** Presence handling code for regular rays.', '        marks.mark(shadeprof::AovExtra);\n'),
    ('        // Convert AOS RayState objects into SOA bundles.', '        marks.mark(shadeprof::PresenceRaysIncl);\n'),
    ('    // Increment shader evaluation stats.', '    marks.mark(shadeprof::Tail);\n'),
]
# (anchor, text inserted after it).
AFTER = [
    ('    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SHADE_HANDLER);\n', '    shadeprof::Marks marks(numEntries);\n'),
    ('                                   -ray->getDirection());\n', '        marks.mark(shadeprof::InitIsect);\n'),
    ('                rs->mSubpixel.mTextureDiffScale);\n', '            marks.mark(shadeprof::Derivs);\n'),
    ('    pbrTls->freeRayStates(numRayStatesToFree, rayStatesToFree);\n', '    marks.mark(shadeprof::Free1);\n'),
    ('    sortedEntries = scene_rdl2::util::smartSort32<SortedEntry, 0>(numEntries, sortedEntries, 0xffffffff, arena);\n',
     '    marks.mark(shadeprof::Sort);\n'),
    ('        convertAOSIntersectionsToSOA(pbrTls, workLoadSize, isectMemory, isectsSOA,\n'
     '                                     &sortedEntries[0].mIsectIdx);\n', '        marks.mark(shadeprof::IsectSoa);\n'),
    ('        memset(bsdfv, 0, sizeof(shading::Bsdfv) * numBlocks);\n', '        marks.mark(shadeprof::BsdfClear);\n'),
    ('        convertAOSRayStatesToSOA(pbrTls, workLoadSize, baseRayState,\n'
     '                                 rayStatesSOA, &sortedEntries[0].mRsIdx);\n', '        marks.mark(shadeprof::RsSoa);\n'),
    ('lightList, lightFilterLists, lightAcc, presences);\n', '        marks.mark(shadeprof::IntegrateIncl);\n'),
    ('        pbrTls->freeRayStates(workLoadSize, rayStates);\n', '        marks.mark(shadeprof::Free2);\n'),
]


CHUNK_LOOP = '\n    while (numRemaining) {\n'


def patch(sources):
    """Marks go into the entry loop and MoonRay's own chunk loop. An XPU2 GPU-owned
    branch between them (in the candidate's copy: a duplicate of the chunk body,
    used only with XPU2 on) is left unmarked and falls into chunk_alloc."""
    f = FILES[0]
    s = sources[f]
    s = once(s, '#include "ShadeBundleHandler.h"\n', '#include "ShadeBundleHandler.h"\n#include "shade_profile.h"\n')
    if s.count(CHUNK_LOOP) != 1:
        raise ValueError('chunk loop drift')
    w = s.index(CHUNK_LOOP)
    o = s.find('\n    const unsigned xpu2Depth=')
    o = w if o < 0 else o
    head, mid, tail = s[:o], s[o:w], s[w:]
    for anchor, text in BEFORE:
        if anchor in head:
            head = once(head, anchor, text + anchor)
        else:
            tail = once(tail, anchor, text + anchor)
    for anchor, text in AFTER:
        if anchor in head:
            head = once(head, anchor, anchor + text)
        else:
            tail = once(tail, anchor, anchor + text)
    return {f: head + mid + tail}

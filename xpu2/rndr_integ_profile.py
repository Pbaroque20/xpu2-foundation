"""Insert stage marks (xpu2/integ_profile.cc) into copied bundled-integration ISPC sources.

Works on the XPU2 candidate's own copies of PathIntegratorBundled.ispc and
PathIntegratorMultiSampler.ispc (tools/xpu2_rebase_build.py picks them up from
its link list). Stage numbers index kNames in integ_profile.cc.
"""
from .native_gpu_paths import once

FILES = ('pbr/integrator/PathIntegratorBundled.ispc', 'pbr/integrator/PathIntegratorMultiSampler.ispc',
         'pbr/integrator/PathIntegratorUtil.ispc')
HEADERS = ()
NEW_SOURCES = ('integ_profile.cc',)

DECL = ('extern "C" void XPU2_integBegin(uniform unsigned int numEntries);\n'
        'extern "C" void XPU2_integMark(uniform int stage);\n'
        'extern "C" void XPU2_integEnd();\n')

BUNDLED = [
    ('    for (uniform uint32_t i = 0; i < numBlocks; ++i) {\n        if(xpu2Window', 0),
    ('        CHECK_CANCELLATION(pbrTls, return);\n\n        varying RayState * uniform rs = &rayStates[i];', 1),
    ('        // Volumes\n        if (rs->mVolHit) {', 2),
    ('        // Only the presence continuation ray contributes to the radiance', 3),
    ("        //---------------------------------------------------------------------\n"
     "        // Early out if we don't have any Bsdf lobes nor Bssrdf", 4),
    ('        // Gather up all lights that can affect the intersection point/normal.', 5),
    ('        //---------------------------------------------------------------------\n'
     '        // Estimate sub-surface scattering', 6),
    ('#ifdef PBR_CHECK_FOR_NAN\n        checkForNan(radiance, "Subsurface scattering"', 7),
    ('        float minTransparency = reduceTransparency(rs->mVolTm);', 8),
    ("        //---------------------------------------------------------------------\n"
     "        // Early out if we don't have any Bsdf lobes\n        if (Bsdf_getLobeCount(bsdf) == 0) {\n\n            if (aovs) {", 9),
    ('        snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_COUNTER_A);', 10),
    ('    MNRY_ASSERT(numRadiances <= numEntries);', 11),
    ('    Arena_setPtr(arena, memoryBookmark1);\n}', 12),
]

MULTI = [
    ('    // We use a LightSetSampler object to keep track of sampling strategies and', 14),
    ('    // Draw Bsdf and LightSet samples and compute tentative contributions', 15),
    ('    // now that we have drawn our bsdf samples, we can pack our material aovs', 16),
    ('    snapshotLaneUtilization(pbrTls->mStatistics, STATS_VEC_COUNTER_B);', 17),
    ("    //---------------------------------------------------------------------\n    // Let's trace some rays", 18),
    ('    if(xpu2LightFirst){\n        CPP_gpuMaterialStageResume', 19),
    ('    MNRY_ASSERT(numOcclRaysFilled <= numOcclRaysToAlloc * VLEN);', 20),
    ('    Arena_setPtr(arena, memoryBookmark);\n}', 21),
]


# Inside addIndirectOrDirectVisibleContributionsBundled: spawning bounce rays, then the queue adds.
INDIRECT = [
    ('    if (numSpawnedOcclRays) {\n        MNRY_ASSERT(numSpawnedOcclRays <= sampleCount * VLEN);', 22),
    ('    if (numSpawnedRays) {\n        MNRY_ASSERT(numSpawnedRays <= sampleCount * VLEN);', 23),
    ('    Arena_setPtr(arena, memoryBookmark1);\n}', 24),
]


# Inside sampleAndAddDirectLightContributionsBundled's per-light loop (MultiSampler).
DIRECT = [
    ('        // ---------------  Draw light samples from the light', 25, 'before'),
    ('        // ---------------- Apply Russian roulette to the light samples', 31, 'before'),
    ('        // ---------------- Trace shadow rays and add any light sample contributions', 32, 'before'),
    ('dstPresenceShadowsRayIndices, numPresenceShadowsRays, this->mPresenceQuality, isect, light, lightSamplesSoFar);\n',
     33, 'after'),
]
# drawLightSamples and drawBsdfSamples (PathIntegratorUtil.ispc).
LIGHT = [
    ('        const varying Rdl2LightSetList * uniform parentLobeLightSets)\n{\n', 26, 'after'),
    ('    uniform int lightMode=CPP_lightMode(pbrTls);', 27, 'before'),
    ("    // Loop over each light' samples", 28, 'before'),
    ('                P, cullingNormal, lightFilterSample, time, lightSample, currSamp, rayDirFootprint);\n', 29, 'after'),
    ('        addToCounter(pbrTls->mStatistics, STATS_LIGHT_SAMPLES, getActiveLaneCount());', 30, 'before'),
]
BSDF = [
    ('        const varying Rdl2LightSetList * uniform parentLobeLightSets)\n{\n', 34, 'after'),
    # The candidate's copy (GPU-early branch first; with XPU2 off it takes BsdfSampler_sample).
    ('            else isValid=BsdfSampler_sample(pbrTls,&bSampler,lobeIndex,bsdfSample[0],bsdfSample[1],bsmp[s]);\n',
     35, 'after'),
    ('                                             BsdfLobe_getLightSet(lobe), parentLobeLightSets);\n', 36, 'after'),
    ('            integrateBsdfSample(bSampler, lobeIndex,\n'
     '                    lSampler, pv.pathThroughput, lCo, bsmp[s]);\n', 37, 'after'),
]


def _apply(text, anchors, indent):
    for anchor, stage, where in anchors:
        mark = _mark(stage, indent)
        text = once(text, anchor, mark + anchor if where == 'before' else anchor + mark)
    return text


def _function(text, start, anchors, indent):
    """Apply anchors inside the top-level function starting at `start` (up to the next one)."""
    if text.count(start) != 1:
        raise ValueError('function drift: ' + start)
    i = text.index(start)
    j = text.index('\n}\n', i) + 3   # the function's closing brace at column 0
    return text[:i] + _apply(text[i:j], anchors, indent) + text[j:]


def _mark(stage, indent):
    return indent + 'XPU2_integMark(%d);\n' % stage


def patch(sources):
    b = sources[FILES[0]]
    for anchor, stage in BUNDLED:
        indent = '    ' if anchor.startswith('    ') and not anchor.startswith('        ') else '        '
        b = once(b, anchor, _mark(stage, indent) + anchor)
    b = once(b, '    uniform bool owned=CPP_gpuOwnedWindowBound();',
             '    XPU2_integBegin(numEntries);\n    uniform bool owned=CPP_gpuOwnedWindowBound();')
    b = once(b, '    CPP_gpuWindowEnd(window);\n', '    CPP_gpuWindowEnd(window);\n    XPU2_integEnd();\n')
    b = DECL + b

    m = sources[FILES[1]]
    start = m.index('computeRadianceBsdfMultiSampler(const uniform PathIntegrator * uniform this,')
    head, body = m[:start], m[start:]
    i = head.index('addIndirectOrDirectVisibleContributionsBundled( const uniform PathIntegrator * uniform this,')
    ind = head[i:]
    for anchor, stage in INDIRECT:
        ind = once(ind, anchor, _mark(stage, '    ') + anchor)
    head = head[:i] + ind
    head = _function(head, '\nsampleAndAddDirectLightContributionsBundled(', DIRECT, '        ')
    body = once(body, '    uniform uint8_t *uniform memoryBookmark = Arena_getPtr(arena);\n',
                '    uniform uint8_t *uniform memoryBookmark = Arena_getPtr(arena);\n' + _mark(13, '    '))
    for anchor, stage in MULTI:
        body = once(body, anchor, _mark(stage, '    ') + anchor)
    m = DECL + head + body

    u = sources[FILES[2]]
    u = _function(u, '\ndrawLightSamples(uniform PbrTLState * uniform pbrTls,', LIGHT, '        ')
    u = _function(u, '\ndrawBsdfSamples(uniform PbrTLState * uniform pbrTls,', BSDF, '            ')
    u = DECL + u
    return {FILES[0]: b, FILES[1]: m, FILES[2]: u}

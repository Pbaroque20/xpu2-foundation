"""Exact savings in bundled integration, on the XPU2 candidate's ISPC copies.

XPU2_INTEG_EMISSIVE_SKIP (on unless "0"): integrateBundled calls
CPP_computeRadianceEmissiveRegionsBundled for every block with lobes; with no
emissive volume regions in the scene it only stops and starts two profiling
accumulators per lane and adds black into a local that starts black. Skipped
when the scene has none; the local stays black, so the radiance is the same.

XPU2_INTEG_LIGHT_SPLIT (on unless "0"): the candidate's drawLightSamples carries
the XPU2 light-group paths, whose stack arrays (about 100 KB) are allocated and
probed on every call, even when that mode is off. drawLightSamples now decides
the mode first (as the candidate does) and calls either the full candidate
function (mode on, or the switch off) or a copy without the mode's code (mode
off). With the mode off the full function runs exactly the copy's statements,
and ISPC here does not reassociate or contract floating-point operations (no
fast-math, AVX1 without FMA), so the results are the same.
"""
from .native_gpu_paths import once

FILES = ('pbr/integrator/PathIntegratorBundled.ispc', 'pbr/integrator/PathIntegratorUtil.ispc')
HEADERS = ()
NEW_SOURCES = ('integ_fixes.cc',)

ARGS = ('pbrTls, shadingTls, lSampler, bSampler, sp, pv, P, cullingNormal, time, sequenceID, lsmp, '
        'clampingDepth, clampingValue, rayDirFootprint, rs, lightIndex, lightSelectionPdf, parentLobeLightSets')

LIGHT_START = '\ndrawLightSamples(uniform PbrTLState * uniform pbrTls,'
MODE_BEGIN = '    uniform int lightMode=CPP_lightMode(pbrTls);\n'
MODE_END = '    for(uniform int j=0;j<16;++j){lightHandles[j]=0;unmasked {lightValid[j]=false;}}\n'
LOOP_BEGIN = '        if(lightMode){\n            if(lazyWork)'
LOOP_END = ('}else integrateLightSetSample(lSampler,lightIndex,bSampler,pv,currSamp,clampingDepth,clampingValue,'
            'parentLobeLightSets,NULL,0);\n')
LOOP_NEW = ('        integrateLightSetSample(lSampler,lightIndex,bSampler,pv,currSamp,clampingDepth,clampingValue,'
            'parentLobeLightSets,NULL,0);\n')
POST_BEGIN = '    if(lazyWork){\n        uniform unsigned int tileIds[32];'


def _lean(f):
    """The candidate's drawLightSamples with the light-group mode's code removed (mode off)."""
    a = f.index(MODE_BEGIN); b = f.index(MODE_END) + len(MODE_END)
    f = f[:a] + f[b:]
    a = f.index(LOOP_BEGIN); b = f.index(LOOP_END) + len(LOOP_END)
    f = f[:a] + LOOP_NEW + f[b:]
    a = f.index(POST_BEGIN)
    f = f[:a] + '}\n'
    for word in ('lightMode', 'lazyWork', 'tileWork', 'groupLight', 'groupInput', 'lightHandles', 'lightValid'):
        if word in f:
            raise ValueError('light split drift: ' + word + ' left in the lean copy')
    return f


def patch(sources):
    b = sources[FILES[0]]
    anchor = 'extern "C" uniform uint64 CPP_gpuWindowBegin(uniform PbrTLState *uniform tls);\n'
    b = once(b, anchor, anchor + 'extern "C" uniform int CPP_xpu2SkipEmissiveRegions(uniform PbrTLState * uniform tls);\n')
    b = once(b, '        if (Bsdf_getLobeCount(bsdf) > 0) {\n            varying Color volumeEmission = sBlack;',
             '        if (Bsdf_getLobeCount(bsdf) > 0 && !xpu2SkipEmissive) {\n            varying Color volumeEmission = sBlack;')
    b = once(b, '    const uniform FrameState &fs = *pbrTls->mFs;\n\n    // get the scene\'s visible-in-camera LightSet',
             '    const uniform FrameState &fs = *pbrTls->mFs;\n'
             '    const uniform bool xpu2SkipEmissive = CPP_xpu2SkipEmissiveRegions(pbrTls) != 0;\n\n'
             '    // get the scene\'s visible-in-camera LightSet')

    u = sources[FILES[1]]
    if u.count(LIGHT_START) != 1:
        raise ValueError('drawLightSamples drift')
    i = u.index(LIGHT_START) + 1                    # the function name, after "void\n"
    if not u[:i].endswith('void\n'):
        raise ValueError('drawLightSamples return type drift')
    j = u.index('\n}\n', i) + 3
    f = u[i:j]
    full = 'xpu2DrawLightSamplesFull' + f[len('drawLightSamples'):]
    lean = 'xpu2DrawLightSamplesLean' + _lean(f)[len('drawLightSamples'):]
    signature = f[:f.index('\n{\n') + 1]
    wrapper = (signature + '{\n'
               '    uniform int lightMode=CPP_lightMode(pbrTls);\n'
               '    varying int xpu2SampleCount = LightSetSampler_getLightSampleCount(&lSampler);\n'
               '    if(!all(pv.nonMirrorDepth==0) || reduce_max(xpu2SampleCount)>16 || BsdfSampler_getLobeCount(&bSampler)>16)lightMode=0;\n'
               '    if(lightMode || !CPP_xpu2LightSplit()) xpu2DrawLightSamplesFull(' + ARGS + ');\n'
               '    else xpu2DrawLightSamplesLean(' + ARGS + ');\n'
               '}\n')
    u = (u[:i - len('void\n')] + 'noinline void\n' + full + '\nvoid\n' + lean + '\nvoid\n' + wrapper + u[j:])
    u = 'extern "C" uniform int CPP_xpu2LightSplit();\n' + u
    return {FILES[0]: b, FILES[1]: u}

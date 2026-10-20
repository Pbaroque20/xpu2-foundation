"""Instance fix plus: curvature reuses the UVs postIntersect just read.

geom::initIntersectionPhase1 calls Primitive::postIntersect and then
computeIntersectCurvature for the same hit, back to back; both read the hit
triangle's three surface UVs. postIntersect now leaves them in a one-shot,
thread-local slot keyed by mesh, face and triangle half; curvature consumes
and clears it, or reads the UVs itself on any mismatch. The UVs are a pure
function of that key, so results are unchanged.
"""
from .native_gpu_paths import once
from . import geom_instance_override

FILES = ('geom/prim/Util.cc', 'geom/prim/TriMesh.cc', 'geom/prim/QuadMesh.cc')
HEADERS = ()

SLOT = '''
namespace {
// One-shot hand-over of a hit's UVs from postIntersect to computeIntersectCurvature.
struct StHandOver {
    const void* mesh = nullptr;
    int face = -1;
    bool first = false;
    Vec2f st1, st2, st3;
};
thread_local StHandOver tlStHandOver;
} // namespace
'''


def _reuse(s, cls, getter):
    call = ('    Vec2f st1, st2, st3;\n'
            '    %s(baseFaceId,\n'
            '%stessFaceId,\n'
            '%sst1, st2, st3,\n'
            '%sisFirst);\n') % (getter, *[' ' * (len(getter) + 5)] * 3)
    # The slot goes after the last namespace opening, where Vec2f is visible.
    anchor = 'namespace internal {\n'
    s = once(s, anchor, anchor + SLOT)
    start = s.index(cls + '::postIntersect(')
    end = s.index('\n}\n', start)
    p = once(s[start:end], call, call +
             '    tlStHandOver.mesh = this; tlStHandOver.face = tessFaceId; tlStHandOver.first = isFirst;\n'
             '    tlStHandOver.st1 = st1; tlStHandOver.st2 = st2; tlStHandOver.st3 = st3;\n')
    s = s[:start] + p + s[end:]
    start = s.index(cls + '::computeIntersectCurvature(')
    end = s.index('\n}\n', start)
    reuse = ('    Vec2f st1, st2, st3;\n'
             '    StHandOver& handed = tlStHandOver;\n'
             '    if (handed.mesh == this && handed.face == tessFaceId && handed.first == isFirst) {\n'
             '        st1 = handed.st1; st2 = handed.st2; st3 = handed.st3;\n'
             '    } else {\n'
             '        %s(baseFaceId, tessFaceId, st1, st2, st3, isFirst);\n'
             '    }\n'
             '    handed.mesh = nullptr;\n') % getter
    p = once(s[start:end], call, reuse)
    return s[:start] + p + s[end:]


def patch(sources):
    out = geom_instance_override.patch(sources)
    out['geom/prim/TriMesh.cc'] = _reuse(sources['geom/prim/TriMesh.cc'], 'TriMesh', 'getTriangleST')
    out['geom/prim/QuadMesh.cc'] = _reuse(sources['geom/prim/QuadMesh.cc'], 'QuadMesh', 'getQuadST')
    return out

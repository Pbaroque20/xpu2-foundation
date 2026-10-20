"""Insert sampled exclusive Init Intersection stage marks into copied geom sources."""
from .native_gpu_paths import once

FILES = ('geom/IntersectionInit.cc', 'geom/prim/TriMesh.cc', 'geom/prim/QuadMesh.cc')
HEADERS = ('isect_profile.h',)


def _phase1(s):
    s = once(s, '#include "IntersectionInit.h"\n', '#include "IntersectionInit.h"\n#include "isect_profile.h"\n')
    start = s.index('void initIntersectionPhase1(')
    end = s.index('void initIntersectionPhase2(')
    p = s[start:end]
    p = once(p, '    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);\n',
             '    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);\n'
             '    isectprof::begin();\n    isectprof::Marks marks;\n')
    p = once(p, '    isect.setP(ray.getOrigin() + ray.getDirection() * ray.getEnd());\n',
             '    isect.setP(ray.getOrigin() + ray.getDirection() * ray.getEnd());\n'
             '    marks.mark(isectprof::InitP);\n')
    p = once(p, '                           isect);\n\n    // shadow ray epsilon',
             '                           isect);\n    marks.mark(isectprof::PostInclusive);\n\n    // shadow ray epsilon')
    p = once(p, '    scene_rdl2::math::Vec3f dNds, dNdt;\n',
             '    marks.mark(isectprof::Rest);\n    scene_rdl2::math::Vec3f dNds, dNdt;\n')
    p = once(p, '    // We are done setting/overriding instance attributes',
             '    marks.mark(isectprof::Curvature);\n    // We are done setting/overriding instance attributes')
    p = once(p, '    isect.validateRequiredAttributes();\n',
             '    isect.validateRequiredAttributes();\n    marks.mark(isectprof::Validate);\n')
    p = once(p, '        isect.setIsEntering(false);\n    }\n}',
             '        isect.setIsEntering(false);\n    }\n    marks.mark(isectprof::Rest);\n}')
    s = s[:start] + p + s[end:]
    start = s.index('void initIntersectionPhase2(')
    end = s.index('void initIntersectionFull(')
    p = s[start:end]
    p = once(p, '    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);\n',
             '    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);\n    isectprof::Marks marks;\n')
    p = once(p, '    isect.setUseAdaptNormal(true);\n}', '    isect.setUseAdaptNormal(true);\n    marks.mark(isectprof::Phase2);\n}')
    return s[:start] + p + s[end:]


def _mesh(s, cls, kind, table_call, entry):
    s = s.replace('#include "', '#include "isect_profile.h"\n#include "', 1)
    start = s.index(cls + '::postIntersect(')
    end = s.index('\n}\n', start) + 3
    p = s[start:end]
    p = once(p, entry, '    isectprof::primitive(%d);\n    isectprof::Marks marks;\n' % kind + entry)
    p = once(p, table_call, table_call + '    marks.mark(isectprof::MTable);\n')
    p = once(p, '    uint32_t isecId1, isecId2, isecId3;\n    intersection.getIds(isecId1, isecId2, isecId3);\n',
             '    marks.mark(isectprof::MRequired);\n'
             '    uint32_t isecId1, isecId2, isecId3;\n    intersection.getIds(isecId1, isecId2, isecId3);\n')
    p = once(p, '    // The St value is read from', '    marks.mark(isectprof::MInstance);\n    // The St value is read from')
    p = once(p, '    const Vec3f Ng = normalize(ray.getNg());\n', '    marks.mark(isectprof::MSt);\n    const Vec3f Ng = normalize(ray.getNg());\n')
    p = once(p, '    if (!hasExplicitAttributes) {\n        if (primitiveAttributes->isSupported(StandardAttributes::sNormal)) {',
             '    marks.mark(isectprof::MExplicit);\n    if (!hasExplicitAttributes) {\n        if (primitiveAttributes->isSupported(StandardAttributes::sNormal)) {')
    p = once(p, '    // Modo: shadow rays of a toon material', '    marks.mark(isectprof::MGeometry);\n    // Modo: shadow rays of a toon material')
    p = once(p, '    intersection.setDifferentialGeometry(Ng,', '    marks.mark(isectprof::MToon);\n    intersection.setDifferentialGeometry(Ng,')
    p = once(p, '    // calculate dfds/dfdt', '    marks.mark(isectprof::MDiffGeo);\n    // calculate dfds/dfdt')
    p = once(p, '    // fill in ray epsilon', '    marks.mark(isectprof::MAttrDeriv);\n    // fill in ray epsilon')
    if not p.endswith('\n}\n'):
        raise ValueError('postIntersect end drift in ' + cls)
    p = p[:-3] + '\n    marks.mark(isectprof::MTail);\n}\n'
    return s[:start] + p + s[end:]


def _curvature(s):
    start = s.index('TriMesh::computeIntersectCurvature(')
    end = s.index('\n}\n', start) + 3
    p = s[start:end]
    p = once(p, '    int tessFaceId = ray.primID;\n', '    isectprof::Marks marks;\n    int tessFaceId = ray.primID;\n')
    p = once(p, '    Vec3f n1, n2, n3;\n', '    marks.mark(isectprof::CSt);\n    Vec3f n1, n2, n3;\n')
    p = once(p, '    Vec3f dn[2];\n', '    marks.mark(isectprof::CNormal);\n    Vec3f dn[2];\n')
    p = once(p, '        dndt = dn[1];\n        return true;', '        dndt = dn[1];\n        marks.mark(isectprof::CDeriv);\n        return true;')
    return s[:start] + p + s[end:]


def patch(sources):
    """sources: {relative path under moonray/lib/rendering: text} -> patched texts."""
    out = dict(sources)
    out['geom/IntersectionInit.cc'] = _phase1(sources['geom/IntersectionInit.cc'])
    out['geom/prim/TriMesh.cc'] = _mesh(sources['geom/prim/TriMesh.cc'], 'TriMesh', 0,
        '    intersection.setTable(&tls.mArena, table);\n',
        '    int tessFaceId = ray.primID;\n    int baseFaceId = getBaseFaceId(tessFaceId);\n\n    // barycentric coordinate\n')
    out['geom/prim/TriMesh.cc'] = _curvature(out['geom/prim/TriMesh.cc'])
    out['geom/prim/QuadMesh.cc'] = _mesh(sources['geom/prim/QuadMesh.cc'], 'QuadMesh', 1,
        '    intersection.setTable(&tls.mArena,\n                          table);\n',
        '    int primId = ray.primID;\n')
    return out

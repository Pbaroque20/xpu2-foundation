"""Instance hits without a temporary key vector per instance level.

Intersection::setInstanceAttributesOverride builds a std::vector from an
initializer list on every call (one heap allocation and free per instance
level per hit). The replacement performs the same three fills, in the same
order, with a key list built once. Results are unchanged by construction.
"""
from .native_gpu_paths import once

FILES = ('geom/prim/Util.cc',)
HEADERS = ()

HELPER = '''namespace {
// Same fills, in the same order, as Intersection::setInstanceAttributesOverride.
const std::vector<shading::AttributeKey>& explicitShadingKeys()
{
    static const std::vector<shading::AttributeKey> keys = {
        shading::StandardAttributes::sExplicitShading,
        shading::StandardAttributes::sNormal,
        shading::StandardAttributes::sdPds,
        shading::StandardAttributes::sdPdt };
    return keys;
}

inline void overrideFromInstance(shading::Intersection& intersection,
                                 const shading::InstanceAttributes* instanceAttrs)
{
    if (instanceAttrs == nullptr) {
        return;
    }
    intersection.fillInstanceAttributes(instanceAttrs, intersection.getTable()->getRequiredAttributes());
    intersection.fillInstanceAttributes(instanceAttrs, intersection.getTable()->getOptionalAttributes());
    intersection.fillInstanceAttributes(instanceAttrs, explicitShadingKeys());
}
} // namespace

void overrideInstanceAttrs(const mcrt_common::Ray& ray,'''


def patch(sources):
    s = sources['geom/prim/Util.cc']
    s = once(s, 'void overrideInstanceAttrs(const mcrt_common::Ray& ray,', HELPER)
    start = s.index('void overrideInstanceAttrs(const mcrt_common::Ray& ray,\n')
    end = s.index('\n}\n', start)
    body = s[start:end]
    if body.count('intersection.setInstanceAttributesOverride(instanceAttrs);') != 4:
        raise ValueError('overrideInstanceAttrs drift')
    body = body.replace('intersection.setInstanceAttributesOverride(instanceAttrs);',
                        'overrideFromInstance(intersection, instanceAttrs);')
    return {**sources, 'geom/prim/Util.cc': s[:start] + body + s[end:]}

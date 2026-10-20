// Switches for xpu2/rndr_integ_fixes.py (exact savings in bundled integration).
// XPU2_INTEG_EMISSIVE_SKIP and XPU2_INTEG_LIGHT_SPLIT are on unless set to "0".
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/core/Scene.h>

#include <cstdlib>

namespace {
bool xpu2IntegSwitch(const char* name)
{
    const char* v = std::getenv(name);
    return !(v && v[0] == '0' && v[1] == 0);
}
const bool emissiveSkip = xpu2IntegSwitch("XPU2_INTEG_EMISSIVE_SKIP");
const bool lightSplit = xpu2IntegSwitch("XPU2_INTEG_LIGHT_SPLIT");
}

// 1 when the scene has no emissive volume regions, so the per-lane C++ call that
// would only add black can be skipped.
extern "C" int CPP_xpu2SkipEmissiveRegions(const moonray::pbr::TLState* tls)
{
    return emissiveSkip && tls->mFs->mScene->getEmissiveRegions().empty() ? 1 : 0;
}

extern "C" int CPP_xpu2LightSplit()
{
    return lightSplit ? 1 : 0;
}

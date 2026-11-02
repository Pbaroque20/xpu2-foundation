// Private XPU2 shading library, shared CPU/CUDA: map bindings and the map
// programs that evaluate them. A material's bound attributes, its combinators'
// and its maps' own, are a list of (parameter, rule, register); a material's maps
// are a program run once per hit in dependency order, each map leaving its
// result in a register (MoonRay evaluates bindings lazily through function
// pointers; maps are pure, so evaluating them up front gives the same values).
// Derived from MoonRay Apache-2.0 (shading/ispc/EvalAttribute.isph and the map
// DSOs), Copyright 2023-2024 DreamWorks Animation LLC. See MOONRAY_LICENSE.txt.
#pragma once
#include "shade_state.h"
// offsetof without <cstddef>, which NVRTC does not have.
#if defined(__CUDACC_RTC__)
#define X2_OFFSETOF(T, m) ((unsigned long long)(&(((T*)0)->m)))
#else
#include <cstddef>
#define X2_OFFSETOF(T, m) offsetof(T, m)
#endif
namespace xpu2 { namespace interactive { namespace shade {

// The registers of a running map program. On the GPU in the launch thread's scratch, as many as the scene's largest
// program has (which 0: the program's; 1: a bump map's shifted run): OptiX reserves a program's stack for every hardware
// thread, so that 3 KB on the stack is some 400 MB of the card, whatever the scene.
#if defined(__CUDACC__)
#define X2_MAP_REGISTERS(name, tex, which) Color* const name = (tex).registers + (which) * (tex).registerCount
#else
#define X2_MAP_REGISTERS(name, tex, which) Color name[MapMaxRegisters]
#endif
// A material's map results, one register a map: room for the map trees Modo's shader trees make (a layer is a few
// maps). The register file is indexed by value, so it sits in local memory however large it is; only what a material
// uses is touched.
constexpr U32 MapMaxRegisters = 256, MapNoRegister = 0xffffffffu, NoMapProgram = 0xffffffffu;

// EvalAttribute.isph: how a bound attribute takes its map's result.
enum BindRule : U32 {
    BindFloat = 0,  // evalFloat: a nonzero value times the average of the result's channels
    BindColor = 1,  // evalColor: a non-black value times the result
    BindVec2 = 2,   // evalVec2f: the result's first two channels replace the value
    BindVec3 = 3,   // evalVec3f: the result replaces the value
    BindInt = 4,    // evalInt: the result's first channel, truncated, replaces the value
};
struct MapBinding {
    U32 offset, rule, reg;  // offset: floats into the parameter block
};
// The value at v (one to three floats) under rule, given its map's result r.
X2_HD inline void applyRule(float* v, U32 rule, Color r) {
    switch (rule) {
    case BindFloat:
        if (!isZero(v[0])) v[0] *= (r.x + r.y + r.z) / 3.0f;
        break;
    case BindColor:
        if (!isBlack(Color{v[0], v[1], v[2]})) {
            v[0] *= r.x;
            v[1] *= r.y;
            v[2] *= r.z;
        }
        break;
    case BindVec2:
        v[0] = r.x;
        v[1] = r.y;
        break;
    case BindVec3:
        v[0] = r.x;
        v[1] = r.y;
        v[2] = r.z;
        break;
    case BindInt: v[0] = float(int(r.x)); break;
    default: break;
    }
}
X2_HD inline void applyBinding(float* params, const MapBinding& b, const Color* regs) { applyRule(params + b.offset, b.rule, regs[b.reg]); }
X2_HD inline void applyBindings(float* params, const MapBinding* bindings, U32 count, const Color* regs) {
    for (U32 i = 0; i < count; ++i) applyBinding(params, bindings[i], regs);
}
}}}

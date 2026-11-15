// MoonRay's AOVs (pbr/core/Aov.cc, Aov.ispc; rndr/Film.cc), shared by the CPU and CUDA lanes.
//
// A sample gathers its AOV values in one float array laid out as MoonRay's AovSchema (aov_plan.h builds it from the
// scene's render outputs, in MoonRay's file, part and output order): each entry its channels, a value added for the
// average, sum and consistent-sampling filters, kept for min and max, and replaced by a nearer one for the closest
// filter (its depth beside it). The film then merges the samples' arrays by the same filters (async_cpu.h,
// async_gpu.cu) and the writer divides the averages by the pixel's samples (aov_write.h).
//
// Light and visibility AOVs follow MoonRay's light path expressions: MoonRay's own state machine (lpe::StateMachine)
// is built on the host and its transitions tabulated for every state it can reach (aov_plan.h), so a path vertex
// carries a state number as MoonRay's PathVertex::lpeStateId does.
#pragma once
#include "sample_stream.h"
#include "surface_core.h"

namespace xpu2::interactive::aov {
using sample::U32;
using surface::V3;

// AovType: a mask of these.
enum Type : U32 {
    TypeBeauty = 1u << 0,
    TypeAlpha = 1u << 1,
    TypeState = 1u << 3,
    TypePrim = 1u << 4,
    TypeMaterial = 1u << 5,
    TypeVisibility = 1u << 6,
    TypeLight = 1u << 7,
};
// AovFilter (RenderOutput::MathFilter).
enum Filter : U32 { FilterAvg = 0, FilterSum, FilterMin, FilterMax, FilterConsistent, FilterClosest };
// AovSchemaId's fixed values: the beauty, the alpha and the state variables.
enum Schema : U32 {
    SchemaUnknown = 0,
    SchemaBeauty,
    SchemaAlpha,
    StateP,
    StateNg,
    StateN,
    StateSt,
    StateDpds,
    StateDpdt,
    StateDsdx,
    StateDsdy,
    StateDtdx,
    StateDtdy,
    StateWireframe,
    StateWp,
    StateDepth,
    StateMotion,
};
// AovSchema::sLpePrefixUnoccluded.
constexpr U32 PrefixUnoccluded = 1u;
constexpr U32 None = 0xffffffffu;
// A primitive attribute's type (RenderOutput::PrimitiveAttributeType).
enum PrimType : U32 { PrimFloat = 0, PrimVec2, PrimVec3, PrimRgb };

// One AovSchema entry.
struct Entry {
    U32 type, schema, filter, channels;
    U32 offset;   // its first channel in a sample's array
    U32 closest;  // closest filter: its depth's slot (None otherwise)
    U32 lpe;      // light and visibility AOVs: its expression's bit in the state machine's valid sets
    U32 prefixFlags;
    U32 state;    // a state variable (StateP...), or a material AOV's state variable (SchemaUnknown: none)
    U32 material; // a material AOV's descriptor (Tables::materials)
    U32 prim;     // a primitive attribute AOV's attribute (Tables::prims), or a material AOV's (None: none)
    U32 primType; // its PrimType
};

// MaterialAovs::Entry's semantics (ParsedMaterialExpression after MaterialAovs::parseExpression).
enum Property : U32 {
    PropertyUnknown = 0,
    PropertyAlbedo,
    PropertyColor,
    PropertyFactor,
    PropertyEmission,
    PropertyNormal,
    PropertyRadius,
    PropertyRoughness,
    PropertyMatte,
    PropertyPbrValidity,
    PropertyEyeMask,
    PropertyState,
    PropertyPrim,
};
struct MaterialAov {
    U32 property, fresnel, lobeFlags, subsurface;
    // Its label lists (indices into Tables::labelLists): the lobe labels, the material labels, the geometry labels,
    // each a list of label indices (empty: any).
    U32 labels, labelCount, materialLabels, materialLabelCount, geomLabels, geomLabelCount;
    U32 secondary;  // getMaterialAovSecondaryRays: evaluated on secondary rays too (with an LPE)
    int lpeLabel;   // its 'M:' label in the state machine (-1: no LPE)
    U32 lpe;        // its LPE's valid bit (None: no LPE)
    U32 pad;
};

// The state machine's transitions: for each state, the event kinds below each by label (label + 1, 0 for none), then
// the two label-less events.
enum LpeKind : U32 {
    LpeReflectDiffuse = 0,
    LpeReflectGlossy,
    LpeReflectMirror,
    LpeTransmitDiffuse,
    LpeTransmitGlossy,
    LpeTransmitMirror,
    LpeLight,
    LpeEmission,
    LpeExtra,
    LpeMaterial,
    LpeLabelledKinds,
};
struct Tables {
    const Entry* entries = nullptr;
    U32 count = 0, channels = 0, closestCount = 0;
    U32 types = 0;  // the types present (a mask)
    // The state machine: next states, lpeStride a state (LpeLabelledKinds * (lpeLabels + 1) + 2: straight, volume).
    const int* lpe = nullptr;
    U32 lpeStates = 0, lpeLabels = 0, lpeStride = 0;
    int lpeCamera = -1;  // the camera event's state (-1: no light or visibility AOVs)
    int lpeBackground = -1;  // LightAovs' background extra AOV's label ('U:bg_white', white), -1 when no expression reads it
    const U32* lpeValid = nullptr;  // for each state, lpeWords words of the expressions valid there
    U32 lpeWords = 0;
    const MaterialAov* materials = nullptr;
    const int* labelLists = nullptr;
    const U32* primKeys = nullptr;  // each primitive attribute's key (shade::primKey)
    U32 primCount = 0;
    U32 materialCount = 0;
    U32 materialSecondary = 0;  // a material AOV asks for secondary rays
    // Cryptomatte (aov_crypto.h): on, its deep ID attribute's key and its UV attribute's (0: st).
    U32 crypto = 0, cryptoIdKey = 0, cryptoUvKey = 0;
    // Deep output (aov_deep.h): on, the deep ID attributes' keys (deep_id_attribute_names, up to three).
    U32 deep = 0, deepIdCount = 0, deepIdKeys[3] = {0, 0, 0};
};

X2_HD inline bool finitef(float v) { return v == v && v - v == 0.0f; }
X2_HD inline float infinity() {
    union {
        U32 u;
        float f;
    } x{0x7f800000u};
    return x.f;
}

// initFloatArray, for a sample: zero for the average, sum, consistent and closest filters, +inf for min, -inf for max;
// the closest entries' depths +inf.
X2_HD inline void init(const Tables& t, float* aovs, float* depths) {
    for (U32 i = 0; i < t.count; ++i) {
        const Entry& e = t.entries[i];
        const float v = e.filter == FilterMin ? infinity() : e.filter == FilterMax ? -infinity() : 0.0f;
        for (U32 c = 0; c < e.channels; ++c) aovs[e.offset + c] = v;
    }
    for (U32 i = 0; i < t.closestCount; ++i) depths[i] = infinity();
}

// One value for an entry, by its filter: what the film's math filter does with a queued value (Film.cc
// addFilteredAovSampleBundleHandler), within the sample. Non-finite values of min, max and closest are left out, as
// MoonRay queues none; the closest value replaces the entry's own when nearer.
X2_HD inline void put(const Entry& e, float* aovs, float* depths, const float* v, float depth = 0.0f) {
    float* d = aovs + e.offset;
    switch (e.filter) {
    case FilterMin:
        for (U32 c = 0; c < e.channels; ++c)
            if (finitef(v[c]) && v[c] < d[c]) d[c] = v[c];
        break;
    case FilterMax:
        for (U32 c = 0; c < e.channels; ++c)
            if (finitef(v[c]) && v[c] > d[c]) d[c] = v[c];
        break;
    case FilterClosest:
        if (!finitef(depth)) return;
        for (U32 c = 0; c < e.channels; ++c)
            if (!finitef(v[c])) return;
        if (depth < depths[e.closest]) {
            depths[e.closest] = depth;
            for (U32 c = 0; c < e.channels; ++c) d[c] = v[c];
        }
        break;
    default:
        for (U32 c = 0; c < e.channels; ++c) d[c] += v[c];
    }
}

// The state machine.
X2_HD inline int lpeNext(const Tables& t, int state, U32 kind, int label) {
    if (state < 0 || U32(state) >= t.lpeStates) return -1;
    const U32 l = label < 0 || U32(label) >= t.lpeLabels ? 0u : U32(label) + 1u;
    return t.lpe[size_t(state) * t.lpeStride + kind * (t.lpeLabels + 1) + l];
}
X2_HD inline int lpeStraight(const Tables& t, int state) {
    if (state < 0 || U32(state) >= t.lpeStates) return -1;
    return t.lpe[size_t(state) * t.lpeStride + LpeLabelledKinds * (t.lpeLabels + 1)];
}
X2_HD inline int lpeVolume(const Tables& t, int state) {
    if (state < 0 || U32(state) >= t.lpeStates) return -1;
    return t.lpe[size_t(state) * t.lpeStride + LpeLabelledKinds * (t.lpeLabels + 1) + 1];
}
X2_HD inline bool lpeValid(const Tables& t, int state, U32 bit) {
    if (state < 0 || U32(state) >= t.lpeStates || bit == None) return false;
    return (t.lpeValid[size_t(state) * t.lpeWords + (bit >> 5)] >> (bit & 31)) & 1u;
}
// scatterEventTransition: the event by the lobe's side and kind (its type's reflection first, its diffuse first), the
// label its lobe carries (lpe label id, -1: none).
X2_HD inline int lpeScatter(const Tables& t, int state, U32 lobeType, int label) {
    // shade::LobeType's bits: reflection 1, transmission 2, diffuse 4, glossy 16, mirror 64.
    const U32 side = (lobeType & 1u) ? 0u : (lobeType & 2u) ? 3u : None;
    const U32 kind = (lobeType & 4u) ? 0u : (lobeType & 16u) ? 1u : (lobeType & 64u) ? 2u : None;
    if (side == None || kind == None) return -1;
    return lpeNext(t, state, side + kind, label);
}
// subsurfaceEventTransition: <TD> with the subsurface's label.
X2_HD inline int lpeSubsurface(const Tables& t, int state, int label) { return lpeNext(t, state, LpeTransmitDiffuse, label); }

// aovAccumLightAovs (aovAccumLpeAovs for light AOVs): matchValue to the light AOVs valid in this state whose prefix
// flags include prefixFlags; nonMatch, when given, to the others valid in this state. Sum, min and max take the value
// as it is (they are for extra AOVs only), the rest add it. True when one matched.
X2_HD inline bool lightHere(const Tables& t, float* aovs, float* depths, V3 match, const V3* nonMatch, U32 prefixFlags, int state) {
    if (state < 0 || !(t.types & TypeLight)) return false;
    bool any = false;
    for (U32 i = 0; i < t.count; ++i) {
        const Entry& e = t.entries[i];
        if (e.type != TypeLight || !lpeValid(t, state, e.lpe)) continue;
        const bool matches = (e.prefixFlags & prefixFlags) == prefixFlags;
        if (!matches && !nonMatch) continue;
        const V3 value = matches ? match : *nonMatch;
        const float v[3] = {value.x, value.y, value.z};
        float* d = aovs + e.offset;
        for (U32 c = 0; c < 3; ++c) {
            switch (e.filter) {
            case FilterSum: d[c] += v[c]; break;
            case FilterMax: d[c] = d[c] < v[c] ? v[c] : d[c]; break;
            case FilterMin: d[c] = d[c] < v[c] ? d[c] : v[c]; break;
            default: d[c] += v[c];
            }
        }
        any = true;
    }
    return any;
}
// The extras module's callables (async_extras.cu): enumerators, so no global in every relocatable module.
enum : U32 { AovLightCallable = 15, AovVisibilityCallable = 16 };
// A light AOV contribution as the path tracer adds it: on the GPU, in a program built in modules (XPU2_EXTRAS_MODULE),
// the extras module's callable; elsewhere the function itself.
X2_HD inline bool light(const Tables& t, float* aovs, float* depths, V3 match, const V3* nonMatch, U32 prefixFlags, int state) {
#if defined(__CUDACC__) && defined(XPU2_EXTRAS_MODULE)
    return optixDirectCall<bool, const Tables*, float*, float*, V3, const V3*, U32, int>(AovLightCallable, &t, aovs, depths, match, nonMatch,
                                                                                        prefixFlags, state);
#else
    return lightHere(t, aovs, depths, match, nonMatch, prefixFlags, state);
#endif
}
// aovAccumVisibilityAovs: (hits, attempts) to the visibility AOVs valid in this state; passthrough (lpe ignored) for
// aovAccumVisibilityAttempts.
X2_HD inline bool visibilityHere(const Tables& t, float* aovs, float hits, float attempts, int state, bool passthrough = false) {
    if (!(t.types & TypeVisibility)) return false;
    bool any = false;
    for (U32 i = 0; i < t.count; ++i) {
        const Entry& e = t.entries[i];
        if (e.type != TypeVisibility || !(passthrough || lpeValid(t, state, e.lpe))) continue;
        aovs[e.offset] += hits;
        aovs[e.offset + 1] += attempts;
        any = true;
    }
    return any;
}
// A visibility AOV's hits and attempts as the path tracer adds them: the extras module's callable in a program built
// in modules, the function itself elsewhere.
X2_HD inline bool visibility(const Tables& t, float* aovs, float hits, float attempts, int state, bool passthrough = false) {
#if defined(__CUDACC__) && defined(XPU2_EXTRAS_MODULE)
    return optixDirectCall<bool, const Tables*, float*, float, float, int, bool>(AovVisibilityCallable, &t, aovs, hits, attempts, state,
                                                                                passthrough);
#else
    return visibilityHere(t, aovs, hits, attempts, state, passthrough);
#endif
}

}  // namespace xpu2::interactive::aov

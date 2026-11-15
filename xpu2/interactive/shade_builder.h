// Private XPU2 shading library, shared CPU/CUDA: MoonRay's BsdfBuilder, its
// components and the lobe attenuators that layer them (shading/ispc:
// BsdfBuilder.ispc, BsdfComponent.ispc, LobeAttenuator.ispc). Derived from
// MoonRay Apache-2.0, Copyright 2023-2024 DreamWorks Animation LLC. See
// MOONRAY_LICENSE.txt.
//
// The ISPC builder decides per gang ("any lane"); this one per sample. A lobe
// MoonRay would create with zero weight for a lane contributes nothing there, so
// the two agree in expectation.
#pragma once
#include "shade_state.h"
namespace xpu2 { namespace interactive { namespace shade {

enum Behavior : U32 { Additive = 0, UnderPrevious = 1, OverSubsequent = 2, Physical = 3 };

// BsdfComponent.isph
struct Iridescence {
    V3 N;
    float strength;
    U32 colorControl;  // 0 hue interpolation, 1 ramp
    Color primary, secondary;
    bool flipHue;
    ColorRamp ramp;
    float thickness, exponent, at0, at90;
};
struct MicrofacetAnisotropicClearcoat {
    V3 N;
    float eta, roughnessU, roughnessV;
    V3 shadingTangent;
    float thickness;
    Color attenuationColor;
    bool refracts;
    U32 distribution;
    const Iridescence* iridescence;
};
struct MicrofacetIsotropicClearcoat {
    V3 N;
    float eta, roughness, thickness;
    Color attenuationColor;
    bool refracts;
    U32 distribution;
    const Iridescence* iridescence;
};
struct MirrorClearcoat {
    V3 N;
    float eta, thickness;
    Color attenuationColor;
    bool refracts;
    const Iridescence* iridescence;
};
struct MirrorBRDF {
    V3 N;
    Color eta, k;
    bool isConductor;
    const Iridescence* iridescence;
};
struct MirrorBTDF {
    V3 N;
    float eta;
    Color tint;
    float abbeNumber;
};
struct MirrorBSDF {
    V3 N;
    float eta;
    Color tint;
    float abbeNumber, refractionEta, reflectionWeight, transmissionWeight;
    const Iridescence* iridescence;
};
struct MicrofacetAnisotropicBRDF {
    V3 N;
    Color eta, k;
    float roughnessU, roughnessV;
    V3 shadingTangent;
    bool isConductor;
    U32 distribution;
    const Iridescence* iridescence;
    Color favg;
};
struct MicrofacetIsotropicBRDF {
    V3 N;
    Color eta, k;
    float roughness;
    bool isConductor;
    U32 distribution;
    const Iridescence* iridescence;
    Color favg;
};
struct MicrofacetAnisotropicBTDF {
    V3 N;
    float eta, roughnessU, roughnessV;
    V3 shadingTangent;
    Color tint;
    float abbeNumber;
    U32 distribution;
};
struct MicrofacetIsotropicBTDF {
    V3 N;
    float eta, roughness;
    Color tint;
    float abbeNumber;
    U32 distribution;
};
struct MicrofacetAnisotropicBSDF {
    V3 N;
    float eta, roughnessU, roughnessV, transmissionRoughnessU, transmissionRoughnessV;
    bool useTransmissionRoughness;
    V3 shadingTangent;
    Color tint;
    float abbeNumber, refractionEta, reflectionWeight, transmissionWeight;
    U32 distribution;
    const Iridescence* iridescence;
};
struct MicrofacetIsotropicBSDF {
    V3 N;
    float eta, roughness, transmissionRoughness;
    bool useTransmissionRoughness;
    Color tint;
    float abbeNumber, refractionEta, reflectionWeight, transmissionWeight;
    U32 distribution;
    const Iridescence* iridescence;
};
struct LambertianBRDF {
    V3 N;
    Color albedo;
    const Iridescence* iridescence;
};
struct LambertianBTDF {
    V3 N;  // the caller passes the negated normal
    Color tint;
};
struct OrenNayarBRDF {
    V3 N;
    Color albedo;
    float roughness;
    const Iridescence* iridescence;
};
struct FlatDiffuseBRDF {
    V3 N;
    Color albedo;
    float roughness, terminatorShift, flatness, flatnessFalloff;
};
struct FabricBRDF {
    V3 N, shadingTangent;
    float roughness;
    V3 threadDirection;
    float threadElevation;
    Color threadColor;
    const Iridescence* iridescence;
};
struct VelvetBRDF {
    V3 N;
    float roughness;
    Color color;
    bool useAbsorbingFibers;
    const Iridescence* iridescence;
};
struct EyeCausticBRDF {
    V3 N, irisNormal;
    Color causticColor;
    float exponent;
    Color eyeMask;
};
struct ToonBRDF {
    V3 N;
    Color albedo;
    float rampInputScale;
    U32 rampNumPoints;
    const float* rampPositions;
    const U32* rampInterpolators;
    const Color* rampColors;
    bool extendRamp;
};
struct ToonSpecularBRDF {
    V3 N;
    float intensity, fresnelBlend;
    Color tint;
    float rampInputScale;
    U32 rampNumPoints;
    const float* rampPositions;
    const U32* rampInterpolators;
    const float* rampValues;
    float stretchU, stretchV;
    V3 dPds, dPdt;
    bool enableIndirectReflections;
    float indirectReflectionsIntensity, indirectReflectionsRoughness;
};
struct GlitterFlakeBRDF {
    V3 N, flakeN;
    Color eta, k;
    float roughness;
    const Iridescence* iridescence;
    Color favg;
};

// Component initialisers whose arithmetic is more than copying.
X2_HD inline MirrorBRDF mirrorConductor(V3 N, Color reflectivity, Color edgeTint, const Iridescence* iri = nullptr) {
    MirrorBRDF m{N, rgb(0), rgb(0), true, iri};
    complexIorFromColor(reflectivity, edgeTint, m.eta, m.k);
    return m;
}
X2_HD inline MirrorBRDF mirrorDielectric(V3 N, float eta, const Iridescence* iri = nullptr) {
    return MirrorBRDF{N, rgb(eta), rgb(0), false, iri};
}
X2_HD inline MicrofacetIsotropicBRDF microfacetDielectric(V3 N, float eta, float roughness, U32 distribution,
                                                         const Iridescence* iri = nullptr) {
    return MicrofacetIsotropicBRDF{N, rgb(eta), rgb(0), roughness, false, distribution, iri, rgb(averageFresnelReflectance(eta))};
}
X2_HD inline MicrofacetIsotropicBRDF microfacetConductor(V3 N, Color reflectivity, Color edgeTint, float roughness,
                                                        U32 distribution, const Iridescence* iri = nullptr) {
    MicrofacetIsotropicBRDF m{N, rgb(0), rgb(0), roughness, true, distribution, iri, averageFresnelReflectance(reflectivity, edgeTint)};
    complexIorFromColor(reflectivity, edgeTint, m.eta, m.k);
    return m;
}
// MicrofacetIsotropicBRDF_init(eta, k, ...): the conductor from a complex IOR. MoonRay leaves
// its favg unset; energy compensation then depends on memory. White here (no compensation is
// what a zeroed arena gives: favg black). Kept black for the zeroed arena.
X2_HD inline MicrofacetIsotropicBRDF microfacetConductorIor(V3 N, Color eta, Color k, float roughness, U32 distribution,
                                                           const Iridescence* iri = nullptr) {
    return MicrofacetIsotropicBRDF{N, eta, k, roughness, true, distribution, iri, rgb(0)};
}
X2_HD inline MicrofacetAnisotropicBRDF microfacetAnisoConductor(V3 N, Color reflectivity, Color edgeTint, float ru, float rv, V3 T,
                                                               U32 distribution, const Iridescence* iri = nullptr) {
    MicrofacetAnisotropicBRDF m{N, rgb(0), rgb(0), ru, rv, T, true, distribution, iri, averageFresnelReflectance(reflectivity, edgeTint)};
    complexIorFromColor(reflectivity, edgeTint, m.eta, m.k);
    return m;
}
X2_HD inline MicrofacetAnisotropicBRDF microfacetAnisoDielectric(V3 N, float eta, float ru, float rv, V3 T, U32 distribution,
                                                                const Iridescence* iri = nullptr) {
    return MicrofacetAnisotropicBRDF{N, rgb(eta), rgb(0), ru, rv, T, false, distribution, iri, rgb(1)};
}
X2_HD inline GlitterFlakeBRDF glitterFlake(V3 N, V3 flakeN, Color reflectivity, Color edge, float roughness,
                                           const Iridescence* iri = nullptr) {
    GlitterFlakeBRDF g{N, flakeN, rgb(0), rgb(0), roughness, iri, averageFresnelReflectance(reflectivity, edge)};
    complexIorFromColor(reflectivity, edge, g.eta, g.k);
    return g;
}

// LobeAttenuator.isph
enum AttenuatorKind : U32 { AttenuatorSimple, AttenuatorHair, AttenuatorVelvet, AttenuatorClearcoat };
struct Attenuator {
    U32 kind;
    V3 N;
    U32 oneMinusFresnel;
    // clearcoat
    float etaI, etaT, thickness, weight;
    Color attenuationColor;
    bool refracts;
    U32 exitingFresnel;
};

constexpr U32 MaxAttenuators = 16;
struct Builder {
    Bsdf& bsdf;
    const State& state;
    Attenuator active[MaxAttenuators], staged[MaxAttenuators];
    U32 activeCount = 0, stagedCount = 0;
    float currentMediumIor = 1.0f, currentTransmittance = 1.0f, weightAccum = 0.0f;
    bool thinGeo = false, preventLightCulling = false, inAdjacentBlock = false;
    // The kit's MoonRay source switches (MaterialFlags): MOONRAY_MODO_TOON_TERMINATOR, MOONRAY_MODO_TRANSMISSION_ROUGHNESS.
    bool modoToon = false, modoTransmissionRoughness = false;
    bool overflow = false;  // a lobe or Fresnel was dropped: the Bsdf limits were reached
    // The material's label table (aov_plan.h, in the material pool): its count, then the encoded labels of its shader's
    // label 0 (the material's own) and of each lobe label; null when no AOV asks for labels.
    const U32* labels = nullptr;
    // The light set the lobes now added take (BsdfBuilder's lightSet argument), ~0u none.
    U32 lightSet = 0xffffffffu;
    X2_HD Builder(Bsdf& b, const State& s) : bsdf(b), state(s) {}
};
// aovEncodeLabels: a shader's label (0: the material's own, else its lobe label) as the material's table has it.
X2_HD inline U32 encodeLabel(const Builder& me, U32 label) {
    if (!me.labels) return 0;
    return label < me.labels[0] ? me.labels[1 + label] : me.labels[1];
}

X2_HD inline bool isUnderB(U32 b) { return (b & UnderPrevious) != 0; }
X2_HD inline bool isOverB(U32 b) { return (b & OverSubsequent) != 0; }
X2_HD inline bool visible(const Builder& me, float weight, U32 behavior) {
    if (weight < Epsilon) return false;
    return !isUnderB(behavior) || me.currentTransmittance > 0.f;
}
X2_HD inline bool lobeLimit(Builder& me) {
    if (me.bsdf.topCount >= MaxLobes) {
        me.overflow = true;
        return true;
    }
    return false;
}
X2_HD inline U32 createFresnel(Builder& me, Color eta, Color k, bool isConductor, float weight) {
    if (isConductor) return conductorFresnel(me.bsdf.fresnels, eta, k, weight);
    const ShaderIor ior = shaderIor(me.state, eta.x, me.thinGeo);
    return dielectricFresnel(me.bsdf.fresnels, ior.incident, ior.transmitted, weight);
}
X2_HD inline U32 applyFresnel(Builder& me, Color eta, Color k, bool isConductor, float weight, U32 lobe) {
    if (lobe == NoIndex) return NoIndex;
    const U32 f = createFresnel(me, eta, k, isConductor, weight);
    if (f == NoIndex) me.overflow = true;
    me.bsdf.records[lobe].fresnel = f;
    return f;
}
X2_HD inline U32 handleIridescence(Builder& me, U32 lobe, const Iridescence* iri) {
    if (!iri || lobe == NoIndex) return lobe;
    U32 ramp = NoIndex;
    if (iri->colorControl == 1) {
        if (me.bsdf.rampCount >= MaxRamps) {
            me.overflow = true;
            return lobe;
        }
        ramp = me.bsdf.rampCount++;
        me.bsdf.ramps[ramp] = iri->ramp;
    }
    const U32 w = iridescenceLobe(me.bsdf, lobe, iri->N, iri->strength, iri->colorControl, iri->primary, iri->secondary,
                                  iri->flipHue, ramp, iri->thickness, iri->exponent, iri->at0, iri->at90);
    if (w == NoIndex) {
        me.overflow = true;
        return lobe;
    }
    return w;
}
X2_HD inline void stage(Builder& me, const Attenuator& a) {
    if (me.stagedCount < MaxAttenuators - 1) me.staged[me.stagedCount++] = a;
    else me.overflow = true;
}
X2_HD inline void accumulateAttenuation(Builder& me) {
    if (me.inAdjacentBlock) return;
    for (U32 i = 0; i < me.stagedCount && me.activeCount < MaxAttenuators; ++i) me.active[me.activeCount++] = me.staged[i];
    me.stagedCount = 0;
    me.currentTransmittance *= 1.0f - me.weightAccum;
    me.weightAccum = 0.f;
}
// LobeAttenuator_attenuate for a lobe: the wrapper record that replaces it.
X2_HD inline U32 attenuate(Builder& me, const Attenuator& a, U32 lobe) {
    if (lobe == NoIndex) return lobe;
    Bsdf& b = me.bsdf;
    switch (a.kind) {
    case AttenuatorHair:
        b.records[lobe].fresnel = a.oneMinusFresnel;
        return lobe;
    case AttenuatorClearcoat: {
        U32 r;
        if (a.refracts && !isEqual(a.etaI, a.etaT)) {
            const bool refl = (b.records[lobe].type & Reflection) != 0;
            r = underClearcoatLobe(b, refl ? KindUnderClearcoat : KindUnderClearcoatTransmission, lobe, a.N, a.exitingFresnel,
                                   a.etaI, a.etaT, a.thickness, a.attenuationColor, a.weight);
        } else {
            r = underLobe(b, lobe, a.N, a.thickness, a.attenuationColor, a.weight);
        }
        if (r == NoIndex) {
            me.overflow = true;
            return lobe;
        }
        b.records[r].fresnel = a.oneMinusFresnel;
        return r;
    }
    default: {  // simple, velvet
        if (a.oneMinusFresnel == NoIndex) return lobe;
        const U32 r = underLobe(b, lobe, a.N, 0.0f, rgb(1), 1.0f);
        if (r == NoIndex) {
            me.overflow = true;
            return lobe;
        }
        b.records[r].fresnel = a.oneMinusFresnel;
        return r;
    }
    }
}
X2_HD inline U32 placeUnder(Builder& me, U32 lobe) {
    for (int i = int(me.activeCount) - 1; i >= 0; --i) lobe = attenuate(me, me.active[i], lobe);
    return lobe;
}
// The Fresnels that attenuate a subsurface profile under the active layers (attenuate*Simple
// for a Bssrdf or VolumeSubsurface): added to its multiple-transmission Fresnel.
X2_HD inline void placeUnderFresnel(Builder& me, U32 multiple) {
    for (int i = int(me.activeCount) - 1; i >= 0; --i)
        if (me.active[i].oneMinusFresnel != NoIndex) multipleTransmissionAdd(me.bsdf.fresnels, multiple, me.active[i].oneMinusFresnel);
}
X2_HD inline Attenuator simpleAttenuator(Builder& me, V3 N, float roughness, U32 fresnel) {
    Attenuator a{};
    a.kind = AttenuatorSimple;
    a.N = N;
    a.oneMinusFresnel = oneMinusRoughFresnel(me.bsdf.fresnels, fresnel, roughness);
    return a;
}
X2_HD inline Attenuator clearcoatAttenuator(Builder& me, V3 N, float roughness, float etaI, float etaT, bool refracts,
                                            float thickness, Color attenuationColor, float weight, U32 fresnel) {
    Attenuator a{};
    a.kind = AttenuatorClearcoat;
    a.N = N;
    a.etaI = etaI;
    a.etaT = etaT;
    a.refracts = refracts;
    a.thickness = thickness;
    a.attenuationColor = attenuationColor;
    a.weight = weight;
    a.exitingFresnel = dielectricFresnel(me.bsdf.fresnels, etaT, etaI, 1.0f);  // flipped for exiting
    a.oneMinusFresnel = oneMinusRoughFresnel(me.bsdf.fresnels, fresnel, roughness);
    return a;
}
X2_HD inline void finishLobe(Builder& me, U32 lobe, U32 label) {
    if (lobe == NoIndex) {
        me.overflow = true;
        return;
    }
    me.bsdf.records[lobe].label = encodeLabel(me, label);
    me.bsdf.records[lobe].lightSet = me.lightSet;
    if (!addLobe(me.bsdf, lobe)) me.overflow = true;
}
// Scale of an additive lobe, or of one placed under the previous lobes.
X2_HD inline U32 scaleOrPlace(Builder& me, U32 lobe, float weight, U32 behavior) {
    if (lobe == NoIndex) return lobe;
    Color scale = rgb(weight);
    if (isUnderB(behavior)) {
        scale = scale * me.currentTransmittance;
        me.bsdf.records[lobe].scale = scale;
        return placeUnder(me, lobe);
    }
    me.bsdf.records[lobe].scale = scale;
    return lobe;
}
// Under-placement of a lobe whose weight is in its Fresnel: white times the transmittance.
X2_HD inline U32 placeTransmittance(Builder& me, U32 lobe, U32 behavior) {
    if (lobe == NoIndex || !isUnderB(behavior)) return lobe;
    me.bsdf.records[lobe].scale = rgb(1) * me.currentTransmittance;
    return placeUnder(me, lobe);
}

//----------------------------------------------------------------------------
X2_CALL void addMicrofacetAnisotropicClearcoat(Builder& me, const MicrofacetAnisotropicClearcoat& c, float weight, U32 behavior,
                                                    U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = anisoCookTorranceLobe(me.bsdf, N, c.shadingTangent, c.roughnessU, c.roughnessV);
    const U32 fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior))
        stage(me, clearcoatAttenuator(me, N, (c.roughnessU + c.roughnessV) * 0.5f, me.currentMediumIor, c.eta, c.refracts, c.thickness,
                                      c.attenuationColor, weight, fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetIsotropicClearcoat(Builder& me, const MicrofacetIsotropicClearcoat& c, float weight, U32 behavior,
                                                  U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = c.distribution == Beckmann ? cookTorranceLobe(me.bsdf, KindCookTorrance, N, c.roughness, rgb(0), rgb(0))
                                          : cookTorranceLobe(me.bsdf, KindGGX, N, c.roughness, rgb(0), rgb(0));
    const U32 fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior))
        stage(me, clearcoatAttenuator(me, N, c.roughness, me.currentMediumIor, c.eta, c.refracts, c.thickness, c.attenuationColor,
                                      weight, fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMirrorClearcoat(Builder& me, const MirrorClearcoat& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = mirrorReflectionLobe(me.bsdf, N);
    const U32 fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior))
        stage(me, clearcoatAttenuator(me, N, 0.0f, me.currentMediumIor, c.eta, c.refracts, c.thickness, c.attenuationColor, weight,
                                      fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMirrorBRDF(Builder& me, const MirrorBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = mirrorReflectionLobe(me.bsdf, N);
    const U32 fresnel = applyFresnel(me, c.eta, c.k, c.isConductor, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior)) {
        if (c.isConductor) me.weightAccum += weight;
        else stage(me, simpleAttenuator(me, N, 0.0f, fresnel));
    }
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMirrorBTDF(Builder& me, const MirrorBTDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const ShaderIor ior = shaderIor(me.state, c.eta, me.thinGeo);
    U32 lobe = mirrorTransmissionLobe(me.bsdf, c.N, ior.incident, me.thinGeo ? ior.incident : ior.transmitted, c.tint, c.abbeNumber);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMirrorBSDF(Builder& me, const MirrorBSDF& c, float weight, U32 behavior, U32 reflLabel, U32 transLabel) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float reflWeight = weight * c.reflectionWeight, transWeight = weight * c.transmissionWeight;
    const ShaderIor ior = shaderIor(me.state, c.refractionEta, me.thinGeo);
    const V3 N = adaptNormal(me.state, c.N);
    U32 refl = NoIndex, trans = NoIndex;
    if (reflWeight > 0.f) refl = mirrorReflectionLobe(me.bsdf, N);
    if (transWeight > 0.f)
        trans = mirrorTransmissionLobe(me.bsdf, c.N, ior.incident, me.thinGeo ? ior.incident : ior.transmitted, c.tint, c.abbeNumber);
    U32 fresnel = NoIndex;
    if (refl != NoIndex) {
        fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, reflWeight, refl);
        refl = handleIridescence(me, refl, c.iridescence);
        if (trans != NoIndex) me.bsdf.records[trans].fresnel = oneMinusFresnel(me.bsdf.fresnels, fresnel);
    }
    if (isUnderB(behavior)) {
        const Color scale = rgb(1) * me.currentTransmittance;
        if (refl != NoIndex) {
            me.bsdf.records[refl].scale = scale;
            refl = placeUnder(me, refl);
        }
        if (trans != NoIndex) {
            me.bsdf.records[trans].scale = scale * transWeight;
            trans = placeUnder(me, trans);
        }
    }
    if (isOverB(behavior)) {
        if (fresnel != NoIndex && transWeight < 1.f) stage(me, simpleAttenuator(me, N, 0.0f, fresnel));
        me.weightAccum += transWeight;
    }
    if (refl != NoIndex) finishLobe(me, refl, reflLabel);
    if (trans != NoIndex) finishLobe(me, trans, transLabel);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetAnisotropicBRDF(Builder& me, const MicrofacetAnisotropicBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = anisoCookTorranceLobe(me.bsdf, N, c.shadingTangent, c.roughnessU, c.roughnessV);
    const U32 fresnel = applyFresnel(me, c.eta, c.k, c.isConductor, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior)) {
        if (c.isConductor) me.weightAccum += weight;
        else stage(me, simpleAttenuator(me, N, (c.roughnessU + c.roughnessV) * 0.5f, fresnel));
    }
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addSchlickReflection(Builder& me, V3 normal, Color reflectance, float roughness, bool mirror, U32 label) {
    if (!visible(me, 1.0f, Physical) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, normal);
    U32 lobe = mirror ? mirrorReflectionLobe(me.bsdf, N) : cookTorranceLobe(me.bsdf, KindGGX, N, roughness, rgb(0), rgb(0));
    if (lobe == NoIndex) {
        me.overflow = true;
        return;
    }
    const U32 fresnel = schlickFresnel(me.bsdf.fresnels, reflectance, 1.0f);
    me.bsdf.records[lobe].fresnel = fresnel;
    me.bsdf.records[lobe].scale = rgb(1) * me.currentTransmittance;
    lobe = placeUnder(me, lobe);
    stage(me, simpleAttenuator(me, N, roughness, fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetIsotropicBRDF(Builder& me, const MicrofacetIsotropicBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = cookTorranceLobe(me.bsdf, c.distribution == Beckmann ? KindCookTorrance : KindGGX, N, c.roughness, c.favg, rgb(0));
    const U32 fresnel = applyFresnel(me, c.eta, c.k, c.isConductor, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior)) {
        if (c.isConductor) me.weightAccum += weight;
        else stage(me, simpleAttenuator(me, N, c.roughness, fresnel));
    }
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetAnisotropicBTDF(Builder& me, const MicrofacetAnisotropicBTDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const ShaderIor ior = shaderIor(me.state, c.eta, me.thinGeo);
    float favg, favgInv;
    averageFresnelReflectance(ior.transmitted / ior.incident, favg, favgInv);
    U32 lobe = transmissionCookTorranceLobe(me.bsdf, c.N, (c.roughnessU + c.roughnessV) * 0.5f, ior.incident, ior.transmitted, c.tint,
                                            favg, favgInv, c.abbeNumber);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetIsotropicBTDF(Builder& me, const MicrofacetIsotropicBTDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const ShaderIor ior = shaderIor(me.state, c.eta, me.thinGeo);
    float favg, favgInv;
    averageFresnelReflectance(ior.transmitted / ior.incident, favg, favgInv);
    U32 lobe = transmissionCookTorranceLobe(me.bsdf, c.N, c.roughness, ior.incident, ior.transmitted, c.tint, favg, favgInv,
                                            c.abbeNumber);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetAnisotropicBSDF(Builder& me, const MicrofacetAnisotropicBSDF& c, float weight, U32 behavior,
                                               U32 reflLabel, U32 transLabel) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float reflWeight = weight * c.reflectionWeight, transWeight = weight * c.transmissionWeight;
    const ShaderIor ior = shaderIor(me.state, c.refractionEta, me.thinGeo);
    const V3 N = adaptNormal(me.state, c.N);
    U32 refl = NoIndex, trans = NoIndex;
    if (reflWeight > 0.f) refl = anisoCookTorranceLobe(me.bsdf, N, c.shadingTangent, c.roughnessU, c.roughnessV);
    const float avgTrans = (c.transmissionRoughnessU + c.transmissionRoughnessV) * 0.5f;
    const float avgRefl = (c.roughnessU + c.roughnessV) * 0.5f;
    if (transWeight > 0.0f) {
        float favg, favgInv;
        averageFresnelReflectance(ior.transmitted / ior.incident, favg, favgInv);
        trans = transmissionCookTorranceLobe(me.bsdf, c.N, avgTrans, ior.incident, ior.transmitted, c.tint, favg, favgInv, c.abbeNumber);
    }
    U32 fresnel = NoIndex;
    if (refl != NoIndex) {
        fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, reflWeight, refl);
        refl = handleIridescence(me, refl, c.iridescence);
        if (trans != NoIndex)
            me.bsdf.records[trans].fresnel = c.useTransmissionRoughness ? oneMinusRoughFresnel(me.bsdf.fresnels, fresnel, avgRefl)
                                                                        : oneMinusFresnel(me.bsdf.fresnels, fresnel);
    }
    if (isUnderB(behavior)) {
        const Color scale = rgb(1) * me.currentTransmittance;
        if (refl != NoIndex) {
            me.bsdf.records[refl].scale = scale;
            refl = placeUnder(me, refl);
        }
        if (trans != NoIndex) {
            me.bsdf.records[trans].scale = scale * transWeight;
            trans = placeUnder(me, trans);
        }
    }
    if (isOverB(behavior)) {
        if (fresnel != NoIndex && transWeight < 1.f) stage(me, simpleAttenuator(me, N, avgTrans, fresnel));
        me.weightAccum += transWeight;
    }
    if (refl != NoIndex) finishLobe(me, refl, reflLabel);
    if (trans != NoIndex) finishLobe(me, trans, transLabel);
    accumulateAttenuation(me);
}
X2_CALL void addMicrofacetIsotropicBSDF(Builder& me, const MicrofacetIsotropicBSDF& c, float weight, U32 behavior, U32 reflLabel,
                                             U32 transLabel) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float reflWeight = weight * c.reflectionWeight, transWeight = weight * c.transmissionWeight;
    const bool coupled = !isZero(transWeight);
    const ShaderIor ior = shaderIor(me.state, c.refractionEta, me.thinGeo);
    float favg, favgInv;
    averageFresnelReflectance(ior.transmitted / ior.incident, favg, favgInv);
    const V3 N = adaptNormal(me.state, c.N);
    U32 refl = NoIndex, trans = NoIndex;
    if (reflWeight > 0.f)
        refl = cookTorranceLobe(me.bsdf, c.distribution == Beckmann ? KindCookTorrance : KindGGX, N, c.roughness, rgb(favg), rgb(favgInv),
                                ior.incident, ior.transmitted, coupled);
    if (transWeight > 0.f) {
        if (isOne(ior.ratio) || me.thinGeo)
            trans = mirrorTransmissionLobe(me.bsdf, c.N, ior.incident, me.thinGeo ? ior.incident : ior.transmitted, c.tint, c.abbeNumber);
        else
            trans = transmissionCookTorranceLobe(me.bsdf, c.N, c.transmissionRoughness, ior.incident, ior.transmitted, c.tint, favg,
                                                 favgInv, c.abbeNumber);
    }
    U32 fresnel = NoIndex;
    if (refl != NoIndex) {
        fresnel = applyFresnel(me, rgb(c.eta), rgb(0), false, reflWeight, refl);
        refl = handleIridescence(me, refl, c.iridescence);
        if (trans != NoIndex)
            me.bsdf.records[trans].fresnel = c.useTransmissionRoughness ? oneMinusRoughFresnel(me.bsdf.fresnels, fresnel, c.roughness)
                                                                        : oneMinusFresnel(me.bsdf.fresnels, fresnel);
    }
    if (isUnderB(behavior)) {
        const Color scale = rgb(1) * me.currentTransmittance;
        if (refl != NoIndex) {
            me.bsdf.records[refl].scale = scale;
            refl = placeUnder(me, refl);
        }
        if (trans != NoIndex) {
            me.bsdf.records[trans].scale = scale * transWeight;
            trans = placeUnder(me, trans);
        }
    }
    if (isOverB(behavior)) {
        if (fresnel != NoIndex && transWeight < 1.f) stage(me, simpleAttenuator(me, N, c.transmissionRoughness, fresnel));
        me.weightAccum += transWeight;
    }
    if (refl != NoIndex) finishLobe(me, refl, reflLabel);
    if (trans != NoIndex) finishLobe(me, trans, transLabel);
    accumulateAttenuation(me);
}
X2_HD inline void preventCulling(Builder& me, U32 lobe) {
    if (!me.preventLightCulling || lobe == NoIndex) return;
    me.bsdf.records[lobe].type = AllDiffuse;
    me.bsdf.records[lobe].flags |= FlagSpherical;
}
X2_CALL void addLambertianBRDF(Builder& me, const LambertianBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = lambertLobe(me.bsdf, c.N, c.albedo, true);
    preventCulling(me, lobe);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addLambertianBTDF(Builder& me, const LambertianBTDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = lambertLobe(me.bsdf, c.N, c.tint, false);
    preventCulling(me, lobe);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addOrenNayarBRDF(Builder& me, const OrenNayarBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = orenNayarLobe(me.bsdf, c.N, c.albedo, c.roughness, true);
    preventCulling(me, lobe);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addFlatDiffuseBRDF(Builder& me, const FlatDiffuseBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = flatDiffuseLobe(me.bsdf, c.N, c.albedo, c.roughness, c.terminatorShift, c.flatness, c.flatnessFalloff, true);
    preventCulling(me, lobe);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addFabricBRDF(Builder& me, const FabricBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = fabricLobe(me.bsdf, c.N, c.shadingTangent, c.threadDirection, c.threadElevation, c.roughness, c.threadColor, true);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addVelvetBRDF(Builder& me, const VelvetBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    U32 lobe = velvetLobe(me.bsdf, c.N, c.roughness, c.color);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) {
        float w = weight;
        if (!c.useAbsorbingFibers) w *= maxComponent(c.color);
        Attenuator a{};
        a.kind = AttenuatorVelvet;
        a.N = c.N;
        a.oneMinusFresnel = oneMinusVelvetFresnel(me.bsdf.fresnels, c.roughness, w);
        stage(me, a);
    }
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addEyeCausticBRDF(Builder& me, const EyeCausticBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = eyeCausticLobe(me.bsdf, N, c.irisNormal, c.causticColor, c.exponent, c.eyeMask);
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addGlitterFlakeBRDF(Builder& me, const GlitterFlakeBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 N = adaptNormal(me.state, c.N);
    U32 lobe = glitterGGXLobe(me.bsdf, N, c.flakeN, c.roughness, rgb(0));  // no compensation for glitter
    applyFresnel(me, c.eta, c.k, true, weight, lobe);
    lobe = handleIridescence(me, lobe, c.iridescence);
    lobe = scaleOrPlace(me, lobe, 1.0f, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
// BsdfBuilder_addToonBRDF
X2_CALL void addToonBRDF(Builder& me, const ToonBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    if (me.bsdf.rampCount >= MaxRamps) {
        me.overflow = true;
        return;
    }
    const U32 ramp = me.bsdf.rampCount++;
    rampInit(me.bsdf.ramps[ramp], c.rampNumPoints, c.rampPositions, c.rampColors, c.rampInterpolators, RampRgb);
    U32 lobe = toonLobe(me.bsdf, c.N, c.albedo, c.rampInputScale, ramp, c.extendRamp);
    if (me.modoToon && lobe != NoIndex) {
        // MOONRAY_MODO_TOON_TERMINATOR: as if prevent_light_culling were on, and cut off by its shading normal alone.
        me.bsdf.records[lobe].type = AllDiffuse;
        me.bsdf.records[lobe].flags |= FlagSpherical | FlagModoToon;
    } else {
        preventCulling(me, lobe);
    }
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
// BsdfBuilder_addToonSpecularBRDF: a normal facing away from the viewer is bent toward it and the lobe
// faded, past cos -0.9 left out; a dielectric Fresnel (1.5) weighted by the ramp's average.
X2_CALL void addToonSpecularBRDF(Builder& me, const ToonSpecularBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float cosNO = dot(c.N, me.state.wo);
    V3 adapted = c.N;
    float fade = 1.0f;
    if (cosNO < 0.0f) {
        fade = 1.0f - cosNO / -0.9f;
        if (fade < 0.0f) return;
        adapted = normalize(c.N + (0.001f - cosNO) * me.state.wo);
    }
    if (me.bsdf.rampCount >= MaxRamps) {
        me.overflow = true;
        return;
    }
    const U32 ramp = me.bsdf.rampCount++;
    Color values[RampMaxPoints];
    const U32 n = c.rampNumPoints < RampMaxPoints ? c.rampNumPoints : RampMaxPoints;
    for (U32 i = 0; i < n; ++i) values[i] = rgb(c.rampValues[i]);
    ColorRamp& r = me.bsdf.ramps[ramp];
    rampInit(r, n, c.rampPositions, values, c.rampInterpolators, RampRgb);
    // ToonSpecularBsdfLobe_init: the ramp's average over 64 segments, and its normalization.
    const int segments = 64;
    const float dx = 1.0f / segments, half = 0.5f * dx;
    float rampAvg = 0.0f, totalArea = 0.0f;
    for (int i = 0; i < segments; ++i) {
        const float v = rampEval(r, (i * dx + half) / c.rampInputScale).x;
        rampAvg += v;
        if (isZero(v)) continue;
        const float theta0 = i * dx * HalfPi, theta1 = (i + 1) * dx * HalfPi;
        const float a0 = TwoPi * (1.0f - cosf(theta0)), a1 = TwoPi * (1.0f - cosf(theta1));
        totalArea += (a1 - a0) * v;
    }
    rampAvg /= segments;
    const float normalization = !isZero(totalArea) ? TwoPi / totalArea : 1.0f;
    const U32 fresnel = createFresnel(me, rgb(1.5f), rgb(0), false, weight * fade * rampAvg);
    if (fresnel == NoIndex) me.overflow = true;
    U32 lobe = toonSpecularLobe(me.bsdf, adapted, c.intensity * fade * weight, c.fresnelBlend, c.tint, c.rampInputScale, ramp,
                                c.stretchU, c.stretchV, c.dPds, c.dPdt, c.enableIndirectReflections,
                                c.indirectReflectionsRoughness, c.indirectReflectionsIntensity, fresnel, normalization);
    if (lobe != NoIndex) me.bsdf.records[lobe].fresnel = fresnel;
    lobe = placeTransmittance(me, lobe, behavior);
    if (isOverB(behavior) && fresnel != NoIndex) stage(me, simpleAttenuator(me, adapted, c.indirectReflectionsRoughness, fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
// BsdfBuilder_addNormalizedDiffusion and _addDipoleDiffusion: the Bsdf's Bssrdf (Lambertian where the
// path allows no more subsurface), scaled, under the layers above through a multiple-transmission Fresnel.
struct DiffusionProfile {
    V3 N;
    Color albedo, radius;
};
X2_CALL void addDiffusion(Builder& me, U32 type, const DiffusionProfile& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    if (!me.state.subsurfaceAllowed) {
        addLambertianBRDF(me, LambertianBRDF{c.N, c.albedo, nullptr}, weight, behavior, label);
        return;
    }
    BssrdfRecord& b = me.bsdf.bssrdf;
    if (b.type != BssrdfNone) {
        me.overflow = true;  // one Bssrdf a Bsdf
        return;
    }
    if (type == BssrdfDipole) initDipole(b, c.N, c.albedo, c.radius);
    else initNormalizedDiffusion(b, c.N, c.albedo, c.radius);
    b.label = encodeLabel(me, label);
    b.lightSet = me.lightSet;
    b.fresnel = NoIndex;
    b.scale = rgb(weight);
    if (isUnderB(behavior)) {
        b.scale = b.scale * me.currentTransmittance;
        b.fresnel = multipleTransmissionFresnel(me.bsdf.fresnels);
        if (b.fresnel == NoIndex) me.overflow = true;
        else placeUnderFresnel(me, b.fresnel);
    }
    if (isOverB(behavior)) me.weightAccum += weight;
    accumulateAttenuation(me);
}
// addRandomWalkSubsurface: a VolumeSubsurface (Lambertian where subsurface is not allowed).
struct RandomWalkProfile {
    V3 N;
    Color albedo, radius;
    bool resolveSelfIntersections;
    float creaseAttenuation = 0.0f;
};
X2_CALL void addRandomWalk(Builder& me, const RandomWalkProfile& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    if (!me.state.subsurfaceAllowed) {
        addLambertianBRDF(me, LambertianBRDF{c.N, c.albedo, nullptr}, weight, behavior, label);
        return;
    }
    BssrdfRecord& b = me.bsdf.bssrdf;
    if (b.type != BssrdfNone) {
        me.overflow = true;
        return;
    }
    initRandomWalk(b, c.N, c.albedo, c.radius, c.resolveSelfIntersections);
    b.creaseAttenuation = c.creaseAttenuation;
    b.label = encodeLabel(me, label);
    b.lightSet = me.lightSet;
    b.fresnel = NoIndex;
    b.scale = rgb(weight);
    if (isUnderB(behavior)) {
        b.scale = b.scale * me.currentTransmittance;
        b.fresnel = multipleTransmissionFresnel(me.bsdf.fresnels);
        if (b.fresnel == NoIndex) me.overflow = true;
        else placeUnderFresnel(me, b.fresnel);
    }
    if (isOverB(behavior)) me.weightAccum += weight;
    accumulateAttenuation(me);
}
X2_HD inline void addEmission(Builder& me, Color emission) { me.bsdf.selfEmission = me.bsdf.selfEmission + emission; }
X2_HD inline void setEarlyTermination(Builder& me) { me.bsdf.earlyTermination = true; }
X2_HD inline void setThinGeo(Builder& me) { me.thinGeo = true; }
X2_HD inline void setPreventLightCulling(Builder& me, bool p) { me.preventLightCulling = p; }
X2_HD inline void startAdjacentComponents(Builder& me) { me.inAdjacentBlock = true; }
X2_HD inline void endAdjacentComponents(Builder& me) {
    me.inAdjacentBlock = false;
    accumulateAttenuation(me);
}

//----------------------------------------------------------------------------
// Hair (BsdfBuilder.ispc's hair functions; BsdfComponent.isph HairBSDF, HairRBRDF, HairTRTBRDF, HairTTBTDF,
// HairTRRTBRDF, HairDiffuseBSDF). Hair seems unstable below 0.01 roughness (0.001 for the glint).
constexpr float HairRoughnessMin = 0.01f, HairGlintRoughnessMin = 0.001f;
X2_HD inline float clampHairRoughness(float minAttrRoughness, float minRough, float r) {
    return clampf(r, fmaxf(minAttrRoughness, minRough), 1.0f - Epsilon);
}
struct HairBSDF {
    V3 hairDir;
    float uv[2], ior;
    U32 fresnelType;
    float cuticleLayerThickness;
    bool showR;
    float shiftR, roughnessR;
    Color tintR;
    bool showTT;
    float shiftTT, roughnessTT, aziRoughnessTT;
    Color tintTT;
    float saturationTT;
    bool showTRT;
    float shiftTRT, roughnessTRT;
    Color tintTRT;
    bool showGlint;
    float roughnessGlint, eccentricityGlint, saturationGlint, hairRotation;
    V3 hairNormal;
    bool showTRRT;
    Color hairColor;
};
// A separate hair lobe (HairRBRDF and the others: the R lobe's tint is its tint, sigma from the colour and the
// TT azimuthal roughness for the others).
struct HairLobeBSDF {
    V3 hairDir;
    float uv[2], ior;
    U32 fresnelType;
    float cuticleLayerThickness, shift, roughness, aziRoughness;
    Color hairColor, tint;
    float saturation;
    bool showGlint;
    float roughnessGlint, eccentricityGlint, saturationGlint, hairRotation;
    V3 hairNormal;
};
struct HairDiffuseBSDF {
    V3 hairDir;
    Color reflectionColor, transmissionColor;
};
X2_HD inline U32 newHairRecord(Builder& me) {
    Bsdf& b = me.bsdf;
    if (b.hairCount >= MaxHairRecords) {
        me.overflow = true;
        return NoIndex;
    }
    b.hairs[b.hairCount] = HairRecord{};
    return b.hairCount++;
}
X2_HD inline void hairComponent(HairComponent& c, float shift, float longRoughness, float azimRoughness, Color tint, float saturation) {
    c.show = 1;
    c.sinAlpha = sinf(shift);
    c.cosAlpha = cosf(shift);
    c.longRoughness = longRoughness;
    c.longVariance = hair::longitudinalVar(longRoughness);
    c.azimRoughness = azimRoughness;
    c.azimVariance = hair::azimuthalVar(azimRoughness);
    c.tint = tint;
    c.saturation = saturation;
}
X2_HD inline void hairShared(HairRecord& r, const State& st, V3 hairDir, const float uv[2], float ior, U32 fresnelType, float cuticle,
                             Color hairColor, Color sigmaA, float hairRotation, V3 hairNormal) {
    r.hairDir = hairDir;
    r.uv[0] = uv[0];
    r.uv[1] = uv[1];
    r.h = uv[1] * 2.0f - 1.0f;
    r.mediumIor = st.mediumIor;
    r.ior = ior;
    r.fresnelType = fresnelType;
    r.cuticle = lerpf(0.5f, 1.5f, clampf(cuticle, 0.0f, 1.0f));
    r.hairColor = hairColor;
    r.sigmaA = sigmaA;
    r.hairRotation = hairRotation;
    r.hairNormal = hairNormal;
}
X2_HD inline U32 hairLobeRecord(Builder& me, U32 kind, U32 type, U32 hairRecord) {
    const U32 i = newRecord(me.bsdf);
    if (i == NoIndex) {
        me.overflow = true;
        return i;
    }
    Lobe& l = me.bsdf.records[i];
    l.kind = kind;
    l.type = type;
    l.flags = FlagSpherical | FlagHair | (kind == KindHairDiffuse ? 0u : U32(FlagIgnoresIncomingDifferentials));
    l.hair = HairParams{};
    l.hair.record = hairRecord;
    return i;
}
// addHairBSDF: the one-sampler lobe, its Fresnel the weight's.
X2_CALL void addHairBSDF(Builder& me, const HairBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float minRough = minRoughness(me.state);
    const Color sigmaA = hair::absorptionCoefficients(p.hairColor, p.aziRoughnessTT);
    const U32 rec = newHairRecord(me);
    if (rec == NoIndex) return;
    HairRecord& r = me.bsdf.hairs[rec];
    hairShared(r, me.state, p.hairDir, p.uv, p.ior, p.fresnelType, p.cuticleLayerThickness, p.hairColor, sigmaA, p.hairRotation,
               p.hairNormal);
    // HairOneSamplerBsdfLobe_init: its differentials factor's width starts at zero, so stays the minimum.
    const float rRough = clampHairRoughness(HairRoughnessMin, minRough, p.roughnessR);
    const float ttRough = clampHairRoughness(HairRoughnessMin, minRough, p.roughnessTT);
    const float ttAzim = clampHairRoughness(HairRoughnessMin, minRough, p.aziRoughnessTT);
    const float trtRough = clampHairRoughness(HairRoughnessMin, minRough, p.roughnessTRT);
    if (p.showR) hairComponent(r.c[HairR], p.shiftR, rRough, 0.0f, p.tintR, 1.0f);
    if (p.showTT) hairComponent(r.c[HairTT], p.shiftTT, ttRough, ttAzim, p.tintTT, p.saturationTT);
    if (p.showTRT) {
        hairComponent(r.c[HairTRT], p.shiftTRT, trtRough, 0.0f, p.tintTRT, 1.0f);
        r.showGlint = p.showGlint ? 1u : 0u;
        r.glintRoughness = clampHairRoughness(HairGlintRoughnessMin, minRough, p.roughnessGlint);
        r.glintEccentricity = p.eccentricityGlint;
        r.glintSaturation = p.saturationGlint;
    }
    if (p.showTRRT) {
        const float trrtRough = clampHairRoughness(HairRoughnessMin, minRough, 1.0f);
        r.c[HairTRRT].show = 1;
        r.c[HairTRRT].longRoughness = trrtRough;
        r.c[HairTRRT].longVariance = hair::longitudinalVar(trrtRough);
    }
    r.c[HairR].show = p.showR ? 1u : 0u;
    r.c[HairTT].show = p.showTT ? 1u : 0u;
    r.c[HairTRT].show = p.showTRT ? 1u : 0u;
    U32 lobe = hairLobeRecord(me, KindHairOneSampler, Reflection | Transmission | Glossy, rec);
    if (lobe == NoIndex) return;
    me.bsdf.records[lobe].hair.dDFactor = hair::DDFactorMin;
    me.bsdf.records[lobe].hair.roughU = r.c[HairR].longRoughness;
    me.bsdf.records[lobe].hair.roughV = r.c[HairTT].azimRoughness;
    applyFresnel(me, rgb(p.ior), rgb(0), false, weight, lobe);
    if (isUnderB(behavior)) {
        me.bsdf.records[lobe].scale = rgb(1) * me.currentTransmittance;
        lobe = placeUnder(me, lobe);
    }
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
// A separate hair lobe's record (HairBsdfLobe_init): its component, its differentials factor from the narrower
// of its longitudinal and azimuthal widths.
X2_HD inline U32 separateHairLobe(Builder& me, U32 kind, U32 type, const HairLobeBSDF& p, float shift, float longRough, float azimRough,
                                  Color hairColor, Color sigmaA, float saturation, float hairRotation, V3 hairNormal) {
    const U32 rec = newHairRecord(me);
    if (rec == NoIndex) return NoIndex;
    HairRecord& r = me.bsdf.hairs[rec];
    hairShared(r, me.state, p.hairDir, p.uv, p.ior, p.fresnelType, p.cuticleLayerThickness, hairColor, sigmaA, hairRotation, hairNormal);
    hairComponent(r.c[0], shift, longRough, azimRough, p.tint, saturation);
    const U32 lobe = hairLobeRecord(me, kind, type, rec);
    if (lobe == NoIndex) return lobe;
    HairParams& h = me.bsdf.records[lobe].hair;
    h.dDFactor = hair::DDFactorMin + fminf(hair::roughness2Width(longRough), hair::roughness2Width(azimRough)) * hair::DDFactorSlope;
    h.roughU = longRough;
    h.roughV = kind == KindHairTT ? azimRough : longRough;
    return lobe;
}
X2_CALL void addHairRBRDF(Builder& me, const HairLobeBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float rough = clampHairRoughness(HairRoughnessMin, minRoughness(me.state), p.roughness);
    U32 lobe = separateHairLobe(me, KindHairR, Reflection | Glossy, p, p.shift, rough, 0.75f, rgb(1), rgb(1), 1.0f, 0.0f, V3{0, 0, 0});
    if (lobe == NoIndex) return;
    const U32 fresnel = applyFresnel(me, rgb(p.ior), rgb(0), false, weight, lobe);
    if (isUnderB(behavior)) {
        me.bsdf.records[lobe].scale = rgb(1) * me.currentTransmittance;
        lobe = placeUnder(me, lobe);
    }
    if (isOverB(behavior)) {
        Attenuator a{};
        a.kind = AttenuatorHair;
        a.N = V3{0, 0, 1};
        a.oneMinusFresnel = oneMinusFresnel(me.bsdf.fresnels, fresnel);
        stage(me, a);
    }
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
// TRT, TT and TRRT: weighted lobes with no Fresnel of their own.
X2_HD inline void finishWeightedHair(Builder& me, U32 lobe, float weight, U32 behavior, U32 label) {
    if (lobe == NoIndex) return;
    lobe = scaleOrPlace(me, lobe, weight, behavior);
    if (isOverB(behavior)) me.weightAccum += weight;
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addHairTRTBRDF(Builder& me, const HairLobeBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float minRough = minRoughness(me.state);
    const Color sigmaA = hair::absorptionCoefficients(p.hairColor, p.aziRoughness);
    const U32 lobe = separateHairLobe(me, KindHairTRT, Reflection | Glossy, p, p.shift, clampHairRoughness(HairRoughnessMin, minRough, p.roughness),
                                      0.75f, p.hairColor, sigmaA, 1.0f, p.hairRotation, p.hairNormal);
    if (lobe != NoIndex) {
        HairRecord& r = me.bsdf.hairs[me.bsdf.records[lobe].hair.record];
        r.showGlint = p.showGlint ? 1u : 0u;
        r.glintRoughness = clampHairRoughness(HairGlintRoughnessMin, minRough, p.roughnessGlint);
        r.glintEccentricity = p.eccentricityGlint;
        r.glintSaturation = p.saturationGlint;
    }
    finishWeightedHair(me, lobe, weight, behavior, label);
}
X2_CALL void addHairTTBTDF(Builder& me, const HairLobeBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const float minRough = minRoughness(me.state);
    const Color sigmaA = hair::absorptionCoefficients(p.hairColor, p.aziRoughness);
    const U32 lobe = separateHairLobe(me, KindHairTT, Transmission | Glossy, p, p.shift, clampHairRoughness(HairRoughnessMin, minRough, p.roughness),
                                      clampHairRoughness(HairRoughnessMin, minRough, p.aziRoughness), p.hairColor, sigmaA, p.saturation, 0.0f,
                                      V3{0, 0, 0});
    finishWeightedHair(me, lobe, weight, behavior, label);
}
X2_CALL void addHairTRRTBRDF(Builder& me, const HairLobeBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const Color sigmaA = hair::absorptionCoefficients(p.hairColor, p.aziRoughness);
    const U32 lobe = separateHairLobe(me, KindHairTRRT, Reflection | Glossy, p, 0.0f,
                                      clampHairRoughness(HairRoughnessMin, minRoughness(me.state), p.roughness), 0.75f, p.hairColor, sigmaA,
                                      1.0f, 0.0f, V3{0, 0, 0});
    finishWeightedHair(me, lobe, weight, behavior, label);
}
// BsdfComponent HairToonSpecularBRDF and BsdfBuilder_addHairToonSpecularBRDF: a hair toon specular lobe over its hair R
// lobes (HairRBsdfLobe_init: azimuthal roughness 0.75, white colour and absorption), a dielectric Fresnel of 1.5 at
// the weight, and a simple attenuator over the lobes below.
struct HairToonSpecularBRDF {
    V3 N, hairDir;
    float hairUV[2], ior;
    U32 fresnelType;
    float cuticleLayerThickness, shift, roughness;
    Color tint;
    float intensity;
    U32 rampNumPoints;
    const float* rampPositions;
    const U32* rampInterpolators;
    const float* rampValues;
    bool enableIndirectReflections;
    float indirectReflectionsIntensity, indirectReflectionsRoughness;
};
X2_CALL void addHairToonSpecularBRDF(Builder& me, const HairToonSpecularBRDF& c, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const V3 adaptedNormal = adaptNormal(me.state, c.N);
    const float minRough = minRoughness(me.state);
    const U32 rec = newHairRecord(me);
    if (rec == NoIndex) return;
    HairRecord& r = me.bsdf.hairs[rec];
    hairShared(r, me.state, c.hairDir, c.hairUV, c.ior, c.fresnelType, c.cuticleLayerThickness, rgb(1), rgb(1), 0.0f, V3{0, 0, 0});
    const float directRough = clampHairRoughness(HairRoughnessMin, minRough, c.roughness);
    const float indirectRough = clampHairRoughness(HairRoughnessMin, minRough, c.indirectReflectionsRoughness);
    hairComponent(r.c[0], c.shift, directRough, 0.75f, c.tint, 1.0f);
    hairComponent(r.c[1], c.shift, indirectRough, 0.75f, c.tint, 1.0f);
    if (me.bsdf.rampCount >= MaxRamps) {
        me.overflow = true;
        return;
    }
    const U32 ramp = me.bsdf.rampCount++;
    Color values[RampMaxPoints];
    const U32 n = c.rampNumPoints < RampMaxPoints ? c.rampNumPoints : RampMaxPoints;
    for (U32 i = 0; i < n; ++i) values[i] = rgb(c.rampValues[i]);
    rampInit(me.bsdf.ramps[ramp], n, c.rampPositions, values, c.rampInterpolators, RampRgb);
    U32 lobe = newRecord(me.bsdf);
    if (lobe == NoIndex) {
        me.overflow = true;
        return;
    }
    Lobe& l = me.bsdf.records[lobe];
    l.kind = KindHairToonSpecular;
    l.type = Reflection | Glossy;
    l.frame = frame(c.N);
    l.hairToon = HairToonParams{};
    l.hairToon.record = rec;
    l.hairToon.ramp = ramp;
    l.hairToon.enableIndirect = c.enableIndirectReflections ? 1u : 0u;
    l.hairToon.intensity = c.intensity;
    l.hairToon.indirectIntensity = c.indirectReflectionsIntensity;
    l.hairToon.tint = c.tint;
    l.hairToon.dDFactor =
        hair::DDFactorMin + fminf(hair::roughness2Width(indirectRough), hair::roughness2Width(0.75f)) * hair::DDFactorSlope;
    const U32 fresnel = applyFresnel(me, rgb(1.5f), rgb(0), false, weight, lobe);
    if (isUnderB(behavior)) {
        me.bsdf.records[lobe].scale = rgb(1) * me.currentTransmittance;
        lobe = placeUnder(me, lobe);
    }
    if (isOverB(behavior)) stage(me, simpleAttenuator(me, adaptedNormal, c.roughness, fresnel));
    finishLobe(me, lobe, label);
    accumulateAttenuation(me);
}
X2_CALL void addHairDiffuseBSDF(Builder& me, const HairDiffuseBSDF& p, float weight, U32 behavior, U32 label) {
    if (!visible(me, weight, behavior) || lobeLimit(me)) return;
    const U32 lobe = hairLobeRecord(me, KindHairDiffuse, Reflection | Diffuse, NoIndex);
    if (lobe != NoIndex) {
        HairParams& h = me.bsdf.records[lobe].hair;
        h.dir = p.hairDir;
        h.refl = p.reflectionColor;
        h.trans = p.transmissionColor;
        h.roughU = h.roughV = 1.0f;
    }
    finishWeightedHair(me, lobe, weight, behavior, label);
}
}}}

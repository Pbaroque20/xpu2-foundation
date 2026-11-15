// Private experimental: the host half of the DwaBase family (shade_dwabase.h).
// A DwaBase-family scene object becomes a shade::DwaRecord: each attribute's
// value (or, where the class has no such attribute, the default MoonRay's ISPC
// resolveParameters uses) and the hints and uniform parameters DwaBase::update()
// computes. Included by cpu_scene.h after its helpers. Derived from MoonRay
// Apache-2.0 (moonshine lib/material/dwabase/DwaBase.cc and the material DSOs'
// attribute keys), Copyright 2023-2024 DreamWorks Animation LLC. See
// MOONRAY_LICENSE.txt.
#pragma once
#include <algorithm>
#include <cstring>
namespace xpu2::interactive::cpu {

// DwaBaseAttributeKeys, the ones the family's classes set.
enum DwaKey : unsigned {
    KShowFuzz, KFuzz, KFuzzAlbedo, KFuzzRoughness, KFuzzUseAbsorbingFibers, KFuzzNormal, KFuzzNormalDial,
    KShowSpecular, KSpecular, KSpecularModel, KRefractiveIndex, KMetallic, KMetallicColor, KMetallicEdgeColor, KRoughness,
    KAnisotropy, KShadingTangent,
    KShowTransmission, KTransmission, KTransmissionColor, KUseIndTransIor, KIndTransIor, KUseIndTransRoughness, KIndTransRoughness,
    KUseDispersion, KDispersionAbbe,
    KShowDiffuse, KAlbedo, KDiffuseRoughness, KSubsurface, KScatteringColor, KScatteringRadius, KCreaseAttenuation,
    KSubsurfaceTraceSet, KEnableSubsurfaceInputNormal, KSSSResolveSelfIntersections, KDiffuseTransmission,
    KDiffuseTransmissionColor, KDiffuseTransmissionBlending,
    KShowOuterSpecular, KOuterSpecular, KOuterSpecularModel, KOuterSpecularIor, KOuterSpecularRoughness, KOuterSpecularThickness,
    KOuterSpecularAttenuationColor, KOuterSpecularUseBending, KUseOuterSpecularNormal, KOuterSpecularNormal, KOuterSpecularNormalDial,
    KShowEmission, KEmission, KPresence, KInputNormal, KInputNormalDial, KNormalAAStrategy, KNormalAADial, KThinGeometry,
    KCastsCaustics,
    KShowFabricSpecular, KWarpColor, KWarpRoughness, KUseIndependentWeft, KWeftRoughness, KWeftColor, KUseUVsForThreadDirection,
    KWarpThreadDirection, KWarpThreadCoverage, KWarpThreadElevation, KFabricDiffuseScattering,
    KShowGlitter, KGlitter, KGlitterSeed, KGlitterSpace, KGlitterDensity, KGlitterRandomness, KGlitterLayeringMode,
    KGlitterStyleAFrequency, KGlitterSizeA, KGlitterRoughnessA, KGlitterColorA, KGlitterTextureA, KGlitterStyleBFrequency,
    KGlitterSizeB, KGlitterRoughnessB, KGlitterColorB, KGlitterTextureB, KGlitterOrientationRandomness, KGlitterHueVariation,
    KGlitterSaturationVariation, KGlitterValueVariation, KGlitterJitter, KGlitterCompensateDeformation, KGlitterApproximate,
    KGlitterLodQuality, KGlitterDebugMode,
    KIridescence, KIridescenceApplyTo, KIridescenceColorControl, KIridescencePrimary, KIridescenceSecondary, KIridescenceFlipHue,
    KIridescenceRampMode, KIridescencePositions, KIridescenceColors, KIridescenceInterpolations, KIridescenceThickness,
    KIridescenceExponent, KIridescenceAt0, KIridescenceAt90,
    KDiffuseLightSet, KSpecularLightSet,
    // DwaToonMaterial: toon diffuse, toon specular (the multi attributes' names end in their index 0..9)
    KToonDiffuseModel, KToonTerminatorShift, KToonFlatness, KToonFlatnessFalloff, KToonExtendRamp, KToonRampInputScale,
    KToonPositions, KToonColors, KToonInterpolations, KToonRampColorMult, KToonRampOffset,
    KTSIntensity, KTSFresnelBlend, KTSTint, KTSStretchU, KTSStretchV, KTSUseInputVectors, KTSInputU, KTSInputV,
    KTSRampInputScale, KTSPositions, KTSValues, KTSInterpolations, KTSMult, KTSOffset, KTSEnableInputNormal, KTSInputNormal,
    KTSInputNormalDial, KTSEnableIndirect, KTSIndirectIntensity, KTSIndirectRoughness, KPreventLightCulling,
    // HairMaterial_v3 and HairDiffuseMaterial (moonshine lib/material/dwabase/json/hair.json, hair_diffuse.json)
    KHairColor, KHairFresnelType, KHairCuticle, KHairShowR, KHairROffset, KHairRRoughness, KHairRTint, KHairShowTRT,
    KHairTRTOffset, KHairUseTRTRoughness, KHairTRTRoughness, KHairTRTTint, KHairShowGlint, KHairGlintRoughness,
    KHairGlintEccentricity, KHairGlintMinTwists, KHairGlintMaxTwists, KHairGlintSaturation, KHairShowTT, KHairTTOffset,
    KHairUseTTRoughness, KHairTTRoughness, KHairTTAzimRoughness, KHairTTTint, KHairTTSaturation, KHairUseOptimizedSampling,
    KHairShowTRRT, KHairDiffuseIndependent, KHairDiffuseFront, KHairDiffuseBack, KHairSubsurfaceBlend,
    // HairToonMaterial (moonshine dso/material/npr/HairToon): hair diffuse's weight and three toon specular sets
    KShowHairDiffuse, KHairDiffuseWeight,
#define X2_HT_KEYS(n)                                                                                                             KHT##n##Show, KHT##n##Model, KHT##n##FresnelBlend, KHT##n##Intensity, KHT##n##Tint, KHT##n##Roughness, KHT##n##StretchU,         KHT##n##StretchV, KHT##n##UseInputVectors, KHT##n##InputU, KHT##n##InputV, KHT##n##RampInputScale, KHT##n##Positions,         KHT##n##Values, KHT##n##Interpolations, KHT##n##Mult, KHT##n##Offset, KHT##n##EnableInputNormal, KHT##n##InputNormal,         KHT##n##InputNormalDial, KHT##n##EnableIndirect, KHT##n##IndirectIntensity, KHT##n##IndirectRoughness
    X2_HT_KEYS(1), X2_HT_KEYS(2), X2_HT_KEYS(3),
#undef X2_HT_KEYS
    KCount
};
// The multi attributes (ten each, the index after the name).
constexpr unsigned DwaHairToonKeys = KHT2Show - KHT1Show;  // keys in a HairToonMaterial specular set
// A key's HairToonMaterial specular set (0..2) and its key in the first set, or false.
inline bool dwaHairToonKey(unsigned k, unsigned& set, unsigned& first) {
    if (k < KHT1Show || k >= KHT1Show + 3 * DwaHairToonKeys) return false;
    set = (k - KHT1Show) / DwaHairToonKeys;
    first = KHT1Show + (k - KHT1Show) % DwaHairToonKeys;
    return true;
}
inline bool dwaMultiKey(unsigned k) {
    unsigned set, first;
    if (dwaHairToonKey(k, set, first)) return first == KHT1Mult || first == KHT1Offset;
    return k == KToonRampColorMult || k == KToonRampOffset || k == KTSMult || k == KTSOffset;
}
inline const char* dwaKeyName(unsigned k) {
    static const char* names[KCount] = {
        "show_fuzz", "fuzz", "fuzz_albedo", "fuzz_roughness", "use_absorbing_fuzz_fibers", "fuzz_normal", "fuzz_normal_dial",
        "show_specular", "specular", "specular_model", "refractive_index", "metallic", "metallic_color", "metallic_edge_color",
        "roughness", "anisotropy", "shading_tangent",
        "show_transmission", "transmission", "transmission_color", "use_independent_transmission_refractive_index",
        "independent_transmission_refractive_index", "use_independent_transmission_roughness", "independent_transmission_roughness",
        "use_dispersion", "dispersion_abbe_number",
        "show_diffuse", "albedo", "diffuse_roughness", "bssrdf", "scattering_color", "scattering_radius", "crease_attenuation",
        "sss_trace_set", "enable_sss_input_normal", "resolve_self_intersections", "diffuse_transmission",
        "diffuse_transmission_color", "diffuse_transmission_blending_behavior",
        "show_clearcoat", "clearcoat", "clearcoat_model", "clearcoat_refractive_index", "clearcoat_roughness", "clearcoat_thickness",
        "clearcoat_attenuation_color", "clearcoat_bending", "use_independent_clearcoat_normal", "independent_clearcoat_normal",
        "clearcoat_normal_dial",
        "show_emission", "emission", "presence", "input_normal", "input_normal_dial", "normal_AA_strategy", "normal_AA_dial",
        "thin_geometry", "casts_caustics",
        "show_specular", "warp_color", "warp_roughness", "use_independent_weft_attributes", "weft_roughness", "weft_color",
        "use_UVs_for_thread_direction", "warp_thread_direction", "warp_thread_coverage", "warp_thread_elevation",
        "fabric_diffuse_scattering",
        "show_glitter", "glitter", "glitter_seed", "glitter_space", "glitter_density", "glitter_randomness",
        "glitter_layering_mode", "glitter_style_A_frequency", "glitter_size_A", "glitter_roughness_A", "glitter_color_A",
        "glitter_texture_A", "glitter_style_B_frequency", "glitter_size_B", "glitter_roughness_B", "glitter_color_B",
        "glitter_texture_B", "glitter_texture_orientation_randomness", "glitter_color_hue_variation",
        "glitter_color_saturation_variation", "glitter_color_value_variation", "glitter_jitter",
        "glitter_compensate_reference_space_deformation", "glitter_approximate_for_secondary_rays", "glitter_LOD_quality",
        "glitter_debug_mode",
        "iridescence", "iridescence_apply_to", "iridescence_color_control", "iridescence_primary_color",
        "iridescence_secondary_color", "iridescence_flip_hue_direction", "iridescence_ramp_interpolation_mode",
        "iridescence_positions", "iridescence_colors", "iridescence_interpolations", "iridescence_thickness",
        "iridescence_exponent", "iridescence_at_0_incidence", "iridescence_at_90_incidence",
        "diffuse_lightset", "specular_lightset",
        "diffuse_model", "terminator_shift", "diffuse_flatness", "diffuse_flatness_falloff", "extend_ramp", "ramp_input_scale",
        "positions", "colors", "interpolations", "ramp_color_multiplier", "ramp_position_offset",
        "toon_specular_intensity", "toon_specular_fresnel_blend", "toon_specular_tint", "toon_specular_stretch_u",
        "toon_specular_stretch_v", "toon_specular_use_input_vectors_for_stretch", "toon_specular_input_U", "toon_specular_input_V",
        "toon_specular_ramp_input_scale", "toon_specular_positions", "toon_specular_values", "toon_specular_interpolations",
        "toon_specular_mult", "toon_specular_offset", "toon_specular_enable_input_normal", "toon_specular_input_normal",
        "toon_specular_input_normal_dial", "toon_specular_enable_indirect_reflections",
        "toon_specular_indirect_reflections_intensity", "toon_specular_indirect_reflections_roughness", "prevent_light_culling",
        "hair_color", "fresnel_type", "cuticle_layer_thickness", "show_primary_specular", "primary_specular_offset",
        "primary_specular_roughness", "primary_specular_tint", "show_secondary_specular", "secondary_specular_offset",
        "use_independent_secondary_specular_roughness", "independent_secondary_specular_roughness", "secondary_specular_tint",
        "show_hair_glint", "glint_roughness", "glint_eccentricity", "glint_min_twists", "glint_max_twists", "glint_saturation",
        "show_transmission", "transmission_offset", "use_independent_transmission_roughness", "independent_transmission_roughness",
        "transmission_azimuthal_roughness", "transmission_tint", "direct_transmission_saturation", "use_optimized_sampling",
        "show_multiple_scattering", "use_independent_front_and_back_hair_color", "front_hair_color", "back_hair_color",
        "subsurface_blend", "show_hair_diffuse", "hair_diffuse",
#define X2_HT_NAMES(n)                                                                                                             "specular_" #n "_show", "specular_" #n "_model", "specular_" #n "_fresnel_blend", "specular_" #n "_intensity",                      "specular_" #n "_tint", "specular_" #n "_roughness", "specular_" #n "_stretch_u", "specular_" #n "_stretch_v",                  "specular_" #n "_use_input_vectors_for_stretch", "specular_" #n "_input_U", "specular_" #n "_input_V",                          "specular_" #n "_ramp_input_scale", "specular_" #n "_positions", "specular_" #n "_values",                                      "specular_" #n "_interpolations", "specular_" #n "_mult", "specular_" #n "_offset", "specular_" #n "_enable_input_normal",         "specular_" #n "_input_normal", "specular_" #n "_input_normal_dial", "specular_" #n "_enable_indirect_reflections",             "specular_" #n "_indirect_reflections_intensity", "specular_" #n "_indirect_reflections_roughness"
        X2_HT_NAMES(1), X2_HT_NAMES(2), X2_HT_NAMES(3)
#undef X2_HT_NAMES
    };
    return names[k];
}
// A family class: its model and the keys its collectAttributeKeys() sets.
struct DwaClass {
    shade::DwaModel model;
    std::vector<unsigned> keys;
    const char* fuzzRoughnessName = "fuzz_roughness";
    std::vector<std::pair<unsigned, const char*>> renames;  // keys whose attribute this class names otherwise
};
inline const DwaClass* dwaClass(const std::string& cls) {
    auto range = [](unsigned a, unsigned b) { std::vector<unsigned> v; for (unsigned k = a; k <= b; ++k) v.push_back(k); return v; };
    auto join = [](std::initializer_list<std::vector<unsigned>> parts) {
        std::vector<unsigned> v;
        for (auto& p : parts) v.insert(v.end(), p.begin(), p.end());
        return v;
    };
    const auto fuzz = range(KShowFuzz, KFuzzNormalDial), outer = range(KShowOuterSpecular, KOuterSpecularNormalDial),
               glitter = range(KShowGlitter, KGlitterDebugMode), iridescence = range(KIridescence, KIridescenceAt90),
               transmissionAll = range(KShowTransmission, KDispersionAbbe), diffuseAll = range(KShowDiffuse, KDiffuseTransmissionBlending),
               common = range(KShowEmission, KCastsCaustics), fabric = range(KShowFabricSpecular, KFabricDiffuseScattering);
    const std::vector<unsigned> simpleCommon = {KShowEmission, KEmission, KPresence, KInputNormal, KInputNormalDial, KThinGeometry,
                                                KCastsCaustics};
    const std::vector<unsigned> diffuseSimple = {KShowDiffuse, KAlbedo, KDiffuseRoughness, KDiffuseTransmission,
                                                 KDiffuseTransmissionColor, KDiffuseTransmissionBlending};
    static const std::map<std::string, DwaClass> classes = {
        {"DwaBaseMaterial", {shade::ModelMixed, join({glitter, iridescence, fuzz, range(KShowSpecular, KShadingTangent), transmissionAll,
                                                      diffuseAll, outer, common, {KDiffuseLightSet, KSpecularLightSet}})}},
        {"DwaMetalMaterial", {shade::ModelMetal, join({glitter, iridescence, fuzz,
                                                       {KShowSpecular, KSpecular, KSpecularModel, KMetallicColor, KMetallicEdgeColor,
                                                        KRoughness, KAnisotropy, KShadingTangent},
                                                       outer, common})}},
        {"DwaRefractiveMaterial", {shade::ModelRefractive, join({iridescence, fuzz,
                                                                {KShowSpecular, KSpecular, KSpecularModel, KRefractiveIndex, KRoughness,
                                                                 KAnisotropy, KShadingTangent, KShowTransmission, KTransmissionColor,
                                                                 KUseIndTransIor, KIndTransIor, KUseIndTransRoughness, KIndTransRoughness,
                                                                 KUseDispersion, KDispersionAbbe},
                                                                outer, common})}},
        {"DwaSolidDielectricMaterial", {shade::ModelSolidDielectric, join({glitter, iridescence, fuzz,
                                                                           {KShowSpecular, KSpecular, KSpecularModel, KRefractiveIndex,
                                                                            KRoughness, KAnisotropy, KShadingTangent},
                                                                           diffuseAll, outer, common})}},
        {"DwaFabricMaterial", {shade::ModelFabric, join({glitter, fuzz, diffuseSimple, fabric, simpleCommon})}},
        {"DwaVelvetMaterial_v2", {shade::ModelVelvetV2, join({glitter, fuzz, diffuseSimple, simpleCommon}), "fuzz_coverage"}},
        {"DwaEmissiveMaterial", {shade::ModelMixed, {KShowEmission, KEmission, KPresence}}},
        // DwaSkinMaterial: its clearcoat is "moisture".
        {"DwaSkinMaterial", {shade::ModelSolidDielectric,
                             join({glitter, iridescence, fuzz, {KShowSpecular, KSpecular, KSpecularModel, KRefractiveIndex},
                                   {KShowOuterSpecular, KOuterSpecular, KOuterSpecularModel, KOuterSpecularIor, KOuterSpecularRoughness,
                                    KUseOuterSpecularNormal, KOuterSpecularNormal, KOuterSpecularNormalDial},
                                   {KRoughness, KAnisotropy, KShadingTangent}, diffuseAll,
                                   {KShowEmission, KEmission, KInputNormal, KPresence, KInputNormalDial, KNormalAAStrategy, KNormalAADial,
                                    KThinGeometry, KCastsCaustics}}),
                             "fuzz_roughness",
                             {{KShowOuterSpecular, "show_moisture"}, {KOuterSpecular, "moisture_mask"},
                              {KOuterSpecularModel, "moisture_model"}, {KOuterSpecularIor, "moisture_refractive_index"},
                              {KOuterSpecularRoughness, "moisture_roughness"}, {KUseOuterSpecularNormal, "use_independent_moisture_normal"},
                              {KOuterSpecularNormal, "independent_moisture_normal"}, {KOuterSpecularNormalDial, "moisture_normal_dial"}}}},
        {"DwaToonMaterial", {shade::ModelToon, join({glitter, iridescence, range(KToonDiffuseModel, KTSIndirectRoughness), fuzz,
                                                     range(KShowSpecular, KShadingTangent), transmissionAll, diffuseAll, outer, common,
                                                     {KPreventLightCulling, KDiffuseLightSet, KSpecularLightSet}})}},
        {"HairMaterial_v3", {shade::ModelHair, join({range(KHairColor, KHairShowTRRT),
                                                     {KRefractiveIndex, KPresence, KCastsCaustics, KShowEmission, KEmission,
                                                      KDiffuseLightSet, KSpecularLightSet}})}},
        {"HairToonMaterial", {shade::ModelHairToon,
                              join({{KHairColor, KHairDiffuseIndependent, KHairDiffuseFront, KHairDiffuseBack, KHairSubsurfaceBlend,
                                     KSubsurface, KScatteringColor, KScatteringRadius, KSubsurfaceTraceSet, KEnableSubsurfaceInputNormal,
                                     KInputNormal, KInputNormalDial, KPresence, KShowEmission, KEmission, KShowHairDiffuse,
                                     KHairDiffuseWeight, KDiffuseLightSet, KSpecularLightSet},
                                    range(KHT1Show, KHT3IndirectRoughness)})}},
        {"HairDiffuseMaterial", {shade::ModelHairDiffuse,
                                 {KHairColor, KHairDiffuseIndependent, KHairDiffuseFront, KHairDiffuseBack, KHairSubsurfaceBlend, KSubsurface,
                                  KScatteringColor, KScatteringRadius, KSubsurfaceTraceSet, KEnableSubsurfaceInputNormal, KInputNormal,
                                  KInputNormalDial, KPresence, KCastsCaustics, KShowEmission, KEmission}}},
    };
    auto it = classes.find(cls);
    return it == classes.end() ? nullptr : &it->second;
}

// The record of one DwaBase-family object (DwaBase::update() and resolveUniformParameters()).
// The DwaInputs field a bindable key's attribute shades through, in floats; -1 for none.
inline int dwaInputOffset(unsigned k) {
    using I = shade::DwaInputs;
#define X2_IN(field) int(X2_OFFSETOF(I, field) / sizeof(float))
    switch (k) {
    case KSpecular: return X2_IN(specular);
    case KGlitter: return X2_IN(glitter);
    case KGlitterStyleAFrequency: return X2_IN(glitterStyleAFrequency);
    case KGlitterColorA: return X2_IN(glitterColorA);
    case KGlitterSizeA: return X2_IN(glitterSizeA);
    case KGlitterStyleBFrequency: return X2_IN(glitterStyleBFrequency);
    case KGlitterColorB: return X2_IN(glitterColorB);
    case KGlitterSizeB: return X2_IN(glitterSizeB);
    case KGlitterHueVariation: return X2_IN(glitterHueVariation);
    case KGlitterSaturationVariation: return X2_IN(glitterSaturationVariation);
    case KGlitterValueVariation: return X2_IN(glitterValueVariation);
    case KGlitterDensity: return X2_IN(glitterDensity);
    case KGlitterJitter: return X2_IN(glitterJitter);
    case KRoughness: return X2_IN(roughness);
    case KAnisotropy: return X2_IN(anisotropy);
    case KShadingTangent: return X2_IN(shadingTangent);
    case KMetallic: return X2_IN(metallic);
    case KMetallicColor: return X2_IN(metallicColor);
    case KMetallicEdgeColor: return X2_IN(metallicEdgeColor);
    case KIridescence: return X2_IN(iridescence);
    case KIridescencePrimary: return X2_IN(iridescencePrimary);
    case KIridescenceSecondary: return X2_IN(iridescenceSecondary);
    case KIridescenceThickness: return X2_IN(iridescenceThickness);
    case KIridescenceExponent: return X2_IN(iridescenceExponent);
    case KIridescenceAt0: return X2_IN(iridescenceAt0);
    case KIridescenceAt90: return X2_IN(iridescenceAt90);
    case KTransmission: return X2_IN(transmission);
    case KTransmissionColor: return X2_IN(transmissionColor);
    case KIndTransRoughness: return X2_IN(independentTransmissionRoughness);
    case KFuzz: return X2_IN(fuzz);
    case KFuzzAlbedo: return X2_IN(fuzzAlbedo);
    case KFuzzRoughness: return X2_IN(fuzzRoughness);
    case KFuzzNormalDial: return X2_IN(fuzzNormalDial);
    case KOuterSpecular: return X2_IN(outerSpecular);
    case KOuterSpecularRoughness: return X2_IN(outerSpecularRoughness);
    case KOuterSpecularThickness: return X2_IN(outerSpecularThickness);
    case KOuterSpecularAttenuationColor: return X2_IN(outerSpecularAttenuationColor);
    case KOuterSpecularNormalDial: return X2_IN(outerSpecularNormalDial);
    case KFabricDiffuseScattering: return X2_IN(fabricDiffuseScattering);
    case KWarpColor: return X2_IN(warpColor);
    case KWarpRoughness: return X2_IN(warpRoughness);
    case KWeftColor: return X2_IN(weftColor);
    case KWeftRoughness: return X2_IN(weftRoughness);
    case KWarpThreadDirection: return X2_IN(warpThreadDirection);
    case KWarpThreadElevation: return X2_IN(warpThreadElevation);
    case KAlbedo: return X2_IN(albedo);
    case KDiffuseRoughness: return X2_IN(diffuseRoughness);
    case KScatteringColor: return X2_IN(scatteringColor);
    case KScatteringRadius: return X2_IN(scatteringRadius);
    case KCreaseAttenuation: return X2_IN(creaseAttenuation);
    case KDiffuseTransmission: return X2_IN(diffuseTransmission);
    case KDiffuseTransmissionColor: return X2_IN(diffuseTransmissionColor);
    case KEmission: return X2_IN(emission);
    case KPresence: return X2_IN(presence);
    case KInputNormalDial: return X2_IN(inputNormalDial);
    case KToonTerminatorShift: return X2_IN(toonTerminatorShift);
    case KToonFlatness: return X2_IN(toonFlatness);
    case KToonFlatnessFalloff: return X2_IN(toonFlatnessFalloff);
    case KToonRampInputScale: return X2_IN(toonRampInputScale);
    case KTSIntensity: return X2_IN(toonSpecIntensity);
    case KTSFresnelBlend: return X2_IN(toonSpecFresnelBlend);
    case KTSTint: return X2_IN(toonSpecTint);
    case KTSStretchU: return X2_IN(toonSpecStretchU);
    case KTSStretchV: return X2_IN(toonSpecStretchV);
    case KTSInputU: return X2_IN(toonSpecInputU);
    case KTSInputV: return X2_IN(toonSpecInputV);
    case KTSRampInputScale: return X2_IN(toonSpecRampInputScale);
    case KTSInputNormalDial: return X2_IN(toonSpecInputNormalDial);
    case KTSIndirectIntensity: return X2_IN(toonSpecIndirectIntensity);
    case KTSIndirectRoughness: return X2_IN(toonSpecIndirectRoughness);
    case KHairColor: return X2_IN(hairColor);
    case KHairCuticle: return X2_IN(hairCuticleLayerThickness);
    case KHairROffset: return X2_IN(hairROffset);
    case KHairRRoughness: return X2_IN(hairRRoughness);
    case KHairRTint: return X2_IN(hairRTint);
    case KHairTRTOffset: return X2_IN(hairTRTOffset);
    case KHairTRTRoughness: return X2_IN(hairTRTRoughness);
    case KHairTRTTint: return X2_IN(hairTRTTint);
    case KHairGlintRoughness: return X2_IN(hairGlintRoughness);
    case KHairGlintEccentricity: return X2_IN(hairGlintEccentricity);
    case KHairGlintMinTwists: return X2_IN(hairGlintMinTwists);
    case KHairGlintMaxTwists: return X2_IN(hairGlintMaxTwists);
    case KHairGlintSaturation: return X2_IN(hairGlintSaturation);
    case KHairTTOffset: return X2_IN(hairTTOffset);
    case KHairTTRoughness: return X2_IN(hairTTRoughness);
    case KHairTTAzimRoughness: return X2_IN(hairTTAzimuthalRoughness);
    case KHairTTTint: return X2_IN(hairTTTint);
    case KHairTTSaturation: return X2_IN(hairTTSaturation);
    case KHairDiffuseFront: return X2_IN(hairDiffuseFrontColor);
    case KHairDiffuseBack: return X2_IN(hairDiffuseBackColor);
    case KHairSubsurfaceBlend: return X2_IN(hairSubsurfaceBlend);
    // the multi attributes: their first's offset (the others follow, one value apart)
    case KToonRampColorMult: return X2_IN(toonRampColorMultipliers);
    case KToonRampOffset: return X2_IN(toonRampPositionOffsets);
    case KTSMult: return X2_IN(toonSpecRampMultipliers);
    case KTSOffset: return X2_IN(toonSpecRampOffsets);
    default: break;
    }
    // HairToonMaterial's specular sets: hairToonSpec[set]'s fields.
    unsigned set, first;
    if (dwaHairToonKey(k, set, first)) {
        using H = shade::HairToonSpecInputs;
        const int base = X2_IN(hairToonSpec) + int(set * sizeof(H) / sizeof(float));
#define X2_HT(field) base + int(X2_OFFSETOF(H, field) / sizeof(float))
        switch (first) {
        case KHT1FresnelBlend: return X2_HT(fresnelBlend);
        case KHT1Intensity: return X2_HT(intensity);
        case KHT1Tint: return X2_HT(tint);
        case KHT1Roughness: return X2_HT(roughness);
        case KHT1StretchU: return X2_HT(stretchU);
        case KHT1StretchV: return X2_HT(stretchV);
        case KHT1InputU: return X2_HT(inputU);
        case KHT1InputV: return X2_HT(inputV);
        case KHT1RampInputScale: return X2_HT(rampInputScale);
        case KHT1Mult: return X2_HT(rampMultipliers);
        case KHT1Offset: return X2_HT(rampOffsets);
        case KHT1InputNormalDial: return X2_HT(inputNormalDial);
        case KHT1IndirectIntensity: return X2_HT(indirectIntensity);
        case KHT1IndirectRoughness: return X2_HT(indirectRoughness);
        default: return -1;
        }
#undef X2_HT
    }
    if (k == KHairDiffuseWeight) return X2_IN(hairDiffuseWeight);
    return -1;
#undef X2_IN
}

// maps and bindings: where the object's bound attributes go (none when null).
// A Glitter object's tables (Glitter::Glitter; Worley::Worley with static tables and normals), in the pool: the
// static permutation and feature points, which the first static-table noise builds from MoonRay's shared generator
// (glitter is their only user); the flake normals for the randomness (getGgxNormal); the colour random table
// (the seed + 100); the Poisson point-count bounds.
inline unsigned glitterTables(std::vector<float>& pool, int seed, float randomness) {
    namespace sm = scene_rdl2::math;
    const int n = int(shade::GlitterTableSize);
    scene_rdl2::util::Random noiseRandom(0xbeadceef);  // Noise::sNoiseRandom
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), noiseRandom);
    std::vector<float> t(shade::GlitterTablesFloats);
    std::memcpy(t.data() + shade::GlitterPerm, perm.data(), sizeof(int) * n);
    std::vector<float> rx(n), ry(n);
    for (int i = 0; i < n; ++i) {  // Worley::buildStaticPointTables
        t[shade::GlitterPointsX + i] = noiseRandom.getNextFloat();
        t[shade::GlitterPointsY + i] = noiseRandom.getNextFloat();
        t[shade::GlitterPointsZ + i] = noiseRandom.getNextFloat();
        rx[i] = noiseRandom.getNextFloat();
        ry[i] = noiseRandom.getNextFloat();
    }
    const float r = randomness * randomness;
    for (int i = 0; i < n; ++i) {  // getGgxNormal
        const float theta = sm::atan(r * sm::sqrt(rx[i]) * sm::rsqrt(1.0f - rx[i]));
        const float phi = sm::sTwoPi * ry[i];
        float sinTheta, cosTheta, sinPhi, cosPhi;
        sm::sincos(theta, &sinTheta, &cosTheta);
        sm::sincos(phi, &sinPhi, &cosPhi);
        t[shade::GlitterNormalsX + i] = cosPhi * sinTheta;
        t[shade::GlitterNormalsY + i] = sinPhi * sinTheta;
        t[shade::GlitterNormalsZ + i] = cosTheta;
    }
    scene_rdl2::util::Random rng(seed + 100);
    for (int i = 0; i < int(shade::GlitterRandomSize); ++i) t[shade::GlitterRandom + i] = rng.getNextFloat();
    worleyProbabilities(t.data() + shade::GlitterProbLow, t.data() + shade::GlitterProbHigh);
    const unsigned offset = unsigned(pool.size());
    pool.insert(pool.end(), t.begin(), t.end());
    return offset;
}
// The plan's lobe light sets being gathered (cpu_scene.h sets it as a plan is prepared): a material's light set's
// index among them, ~0u for none.
inline thread_local std::vector<const rdl::SceneObject*>* lobeLightSetTable = nullptr;
inline unsigned lobeLightSetIndex(const rdl::SceneObject* s, const std::string& owner) {
    if (!s) return ~0u;
    need(lobeLightSetTable != nullptr && s->getSceneClass().getName() == "LightSet", owner + ": a lobe light set that is not a LightSet");
    auto& t = *lobeLightSetTable;
    for (size_t i = 0; i < t.size(); ++i)
        if (t[i] == s) return unsigned(i);
    t.push_back(s);
    return unsigned(t.size() - 1);
}
inline shade::DwaRecord admitDwa(const rdl::SceneObject& o, const DwaClass& c, MapProgramBuilder* maps = nullptr,
                                 std::vector<shade::MapBinding>* bindings = nullptr) {
    bool has[KCount] = {};
    for (unsigned k : c.keys) has[k] = true;
    auto name = [&](unsigned k) {
        for (auto& r : c.renames)
            if (r.first == k) return r.second;
        return k == KFuzzRoughness ? c.fuzzRoughnessName : dwaKeyName(k);
    };
    std::set<std::string> handled;
    for (unsigned k : c.keys) handled.insert(name(k));
    // Every attribute of the class may hold any value; none may be bound yet (maps are the next phase).
    for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i) handled.insert((*i)->getName());
    // The bound attributes the port shades through its inputs; others must stay unbound.
    std::set<std::string> boundable;
    for (unsigned k : c.keys)
        if (dwaInputOffset(k) >= 0) {
            if (!dwaMultiKey(k)) boundable.insert(name(k));
            else
                for (int i = 0; i < 10; ++i) boundable.insert(std::string(name(k)) + std::to_string(i));
        }
    attributes(o, handled, boundable);
    auto bound = [&](unsigned k) {
        if (!has[k]) return false;
        const rdl::Attribute* a = o.getSceneClass().getAttribute(name(k));
        return a->isBindable() && o.getBinding(*a) != nullptr;
    };
    if (maps && bindings)
        for (unsigned k : c.keys)
            if (dwaMultiKey(k))
                for (int i = 0; i < 10; ++i) {
                    const rdl::Attribute* a = o.getSceneClass().getAttribute(std::string(name(k)) + std::to_string(i));
                    if (!a->isBindable() || !o.getBinding(*a)) continue;
                    const unsigned stride = a->getType() == rdl::TYPE_RGB ? 3u : 1u;
                    bindings->push_back(shade::MapBinding{unsigned(dwaInputOffset(k)) + stride * unsigned(i), bindRule(a->getType()),
                                                          maps->reg(*o.getBinding(*a))});
                }
            else if (dwaInputOffset(k) >= 0 && bound(k)) {
                const rdl::Attribute* a = o.getSceneClass().getAttribute(name(k));
                bindings->push_back(shade::MapBinding{unsigned(dwaInputOffset(k)), bindRule(a->getType()), maps->reg(*o.getBinding(*a))});
            }
    auto B = [&](unsigned k, bool d) { return has[k] ? get<rdl::Bool>(o, name(k)) : d; };
    auto I = [&](unsigned k, int d) { return has[k] ? get<rdl::Int>(o, name(k)) : d; };
    auto F = [&](unsigned k, float d) {
        if (!has[k]) return d;
        float f = get<rdl::Float>(o, name(k));
        need(std::isfinite(f), o.getName() + ": invalid " + name(k));
        return f;
    };
    auto C = [&](unsigned k, float d) {
        if (!has[k]) return shade::rgb(d);
        auto v = get<rdl::Rgb>(o, name(k));
        V3 r{v.r, v.g, v.b};
        need(surface::finite(r), o.getName() + ": invalid " + name(k));
        return r;
    };
    auto obj = [&](unsigned k) { return has[k] ? get<rdl::SceneObject*>(o, name(k)) : nullptr; };
    // Normal maps: their results' registers in the material's map program.
    auto normalReg = [&](unsigned k) -> unsigned {
        const rdl::SceneObject* m = obj(k);
        if (!m) return shade::MapNoRegister;
        if (!maps) return shade::MapNoRegister;  // a record read for its uniform parameters only
        need(m->isA<rdl::NormalMap>(), o.getName() + ": " + name(k) + " is not a normal map");
        return maps->reg(*m);
    };

    shade::DwaRecord r{};
    shade::DwaUniforms& u = r.u;
    shade::DwaInputs& in = r.in;
    u.model = c.model;
    // diffuse_lightset and specular_lightset: the lobes' light sets (lobeLightSetIndex).
    u.up.diffuseLightSet = lobeLightSetIndex(obj(KDiffuseLightSet), o.getName());
    u.up.specularLightSet = lobeLightSetIndex(obj(KSpecularLightSet), o.getName());
    const bool metal = c.model == shade::ModelMetal, refractive = c.model == shade::ModelRefractive, fabricModel = c.model == shade::ModelFabric;
    unsigned h = 0;
    auto hint = [&](unsigned bit, bool on) { if (on) h |= bit; };
    // Glitter (DwaBase::update and updateGlitter): the Glitter object's uniform parameters and tables.
    u.glitterTables = shade::GlitterNone;
    if (has[KShowGlitter] && B(KShowGlitter, false)) {
        const float g = F(KGlitter, 1.0f);
        hint(shade::HintGlitter, !shade::isZero(g));
        hint(shade::HintGlitterIsOne, !bound(KGlitter) && shade::isOne(g));
        if (!shade::isZero(g)) {
            for (auto t : {std::make_pair(KGlitterTextureA, KGlitterStyleAFrequency), std::make_pair(KGlitterTextureB, KGlitterStyleBFrequency)})
                need(get<rdl::String>(o, name(t.first)).empty() || shade::isZero(F(t.second, 0.0f)),
                     o.getName() + ": glitter flake textures are not shaded yet");
            const int space = I(KGlitterSpace, shade::GlitterReferenceSpace);
            need(space == shade::GlitterObjectSpace || space == shade::GlitterReferenceSpace,
                 o.getName() + ": glitter needs object or reference space");
            const int seed = I(KGlitterSeed, 0);
            need(seed >= 0, o.getName() + ": a negative glitter seed reads outside MoonRay's noise tables");
            u.glitterNoiseSeed = seed % int(shade::GlitterTableSize);
            u.glitterSpace = unsigned(space);
            // Flake randomness is capped at 0.6 (GlitterGGXCookTorrance's normal fallback).
            u.glitterFlakeRandomness = 0.6f * std::min(std::max(F(KGlitterRandomness, 0.5f), 0.0f), 1.0f);
            const float lodQuality = std::min(std::max(F(KGlitterLodQuality, 0.5f), 0.0f), 1.0f);
            u.glitterBlendStart = int(100 + (lodQuality * 800));
            const int blendRange = int(3.0f * float(u.glitterBlendStart));
            u.glitterBlendEnd = std::min(u.glitterBlendStart + blendRange, int(shade::GlitterMaxPoints));
            const int layering = I(KGlitterLayeringMode, 0), debug = I(KGlitterDebugMode, 0);
            need(layering == 0 || layering == 1, o.getName() + ": invalid glitter layering mode");
            need(debug >= 0, o.getName() + ": invalid glitter debug mode");
            u.glitterLayeringMode = unsigned(layering);
            u.glitterDebugMode = unsigned(debug);
            if (maps) u.glitterTables = glitterTables(maps->pool, seed, u.glitterFlakeRandomness);
        }
    }
    hint(shade::HintFuzz, has[KShowFuzz] && B(KShowFuzz, false));
    if (has[KShowOuterSpecular] && B(KShowOuterSpecular, false)) {
        hint(shade::HintOuterSpecular, !shade::isZero(F(KOuterSpecular, 0)));
        hint(shade::HintOuterSpecularAbsorption, has[KOuterSpecularThickness] && !shade::isZero(F(KOuterSpecularThickness, 0)));
    }
    if (has[KShowSpecular] && B(KShowSpecular, false)) {
        const bool spec = has[KSpecular] && !shade::isZero(F(KSpecular, 0));
        hint(shade::HintSpecular, spec);
        if (spec) {
            if (metal) {
                h |= shade::HintMetallic | shade::HintMetallicIsOne;
            } else if (has[KMetallic]) {
                const float m = F(KMetallic, 0);
                hint(shade::HintMetallic, !shade::isZero(m));
                hint(shade::HintMetallicIsOne, !bound(KMetallic) && shade::isOne(m));
            }
            hint(shade::HintAnisotropy, has[KAnisotropy] && !shade::isZero(F(KAnisotropy, 0)));
        }
    }
    hint(shade::HintIridescence, has[KIridescence] && !shade::isZero(F(KIridescence, 0)));
    if (has[KShowTransmission] && B(KShowTransmission, false)) {
        if (refractive) {
            h |= shade::HintTransmission | shade::HintTransmissionIsOne;
        } else if (has[KTransmission]) {
            const float t = F(KTransmission, 0);
            hint(shade::HintTransmission, !shade::isZero(t));
            hint(shade::HintTransmissionIsOne, !bound(KTransmission) && shade::isOne(t));
        }
        hint(shade::HintTransmissionIor, has[KUseIndTransIor] && B(KUseIndTransIor, false));
    }
    if (has[KShowFabricSpecular] && B(KShowFabricSpecular, false) && fabricModel) h |= shade::HintFabric | shade::HintFabricIsOne;
    const bool transmissionIsOne = h & shade::HintTransmissionIsOne, metallicIsOne = h & shade::HintMetallicIsOne;
    if (has[KShowDiffuse] && B(KShowDiffuse, false)) {
        bool diffuse = !transmissionIsOne && !metallicIsOne && has[KAlbedo] && !shade::isBlack(C(KAlbedo, 0));
        if (has[KFabricDiffuseScattering]) diffuse = diffuse && !shade::isZero(F(KFabricDiffuseScattering, 0));
        hint(shade::HintDiffuse, diffuse);
    }
    const bool subsurface = (h & shade::HintDiffuse) && has[KScatteringRadius] && has[KScatteringColor] &&
                            !shade::isZero(F(KScatteringRadius, 0)) && !shade::isBlack(C(KScatteringColor, 0));
    hint(shade::HintSubsurface, subsurface);
    if (subsurface) {
        const int model = I(KSubsurface, 0);
        need(model >= 0 && model <= 2, o.getName() + ": invalid subsurface model");
        // Random walk's crease attenuation is shaded (async_integrator.h randomWalkRadiance, shade::principalCurvatures).
        need(!obj(KSubsurfaceTraceSet), o.getName() + ": subsurface trace sets are not shaded yet");
        // MoonRay's subsurface takes the material's Input Normal at its entry and exit points when this is on
        // (PathIntegratorSubsurface: evalNormalFunc, DwaBase::resolveSubsurfaceNormal); with no normal map there that
        // is the shading normal it takes with it off, so only a mapped input normal is beyond XPU2 yet.
        need(!B(KEnableSubsurfaceInputNormal, false) || !obj(KInputNormal),
             o.getName() + ": a subsurface input normal from a normal map is not shaded yet");
    }
    hint(shade::HintDiffuseTransmission, has[KShowDiffuse] && B(KShowDiffuse, false) && !transmissionIsOne && !metallicIsOne &&
                                             has[KDiffuseTransmissionColor] && !shade::isBlack(C(KDiffuseTransmissionColor, 0)) &&
                                             has[KDiffuseTransmission] && !shade::isZero(F(KDiffuseTransmission, 0)));
    hint(shade::HintEmission, has[KShowEmission] && B(KShowEmission, false) && has[KEmission] && !shade::isBlack(C(KEmission, 0)));
    hint(shade::HintThinGeometry, has[KThinGeometry] && B(KThinGeometry, false));
    // DwaBase::update: a toon material always resolves its toon diffuse; its specular is a toon specular
    // when the specular model says so (Toon; Hair, which DwaToonMaterial's enumeration does not offer).
    const int specularModel = has[KSpecularModel] ? I(KSpecularModel, 1) : 1;
    if (c.model == shade::ModelToon) {
        h |= shade::HintToonDiffuse;
        hint(shade::HintToonSpecular, has[KShowSpecular] && B(KShowSpecular, false) && (specularModel == 2 || specularModel == 3));
    }
    hint(shade::HintPreventLightCulling, has[KPreventLightCulling] && B(KPreventLightCulling, false));
    // DwaBase::update: a hair material always resolves its hair parameters.
    if (c.model == shade::ModelHair) h |= shade::HintHair | shade::HintHairIsOne;
    if (c.model == shade::ModelHairToon) {
        // DwaBase::update: hair diffuse by its weight; each shown specular set.
        if (B(KShowHairDiffuse, true)) {
            const float v = F(KHairDiffuseWeight, 1.0f);
            if (!shade::isZero(v)) {
                h |= shade::HintHairDiffuse;
                if (!bound(KHairDiffuseWeight) && shade::isOne(v)) h |= shade::HintHairDiffuseIsOne;
            }
        }
        for (unsigned n = 0; n < 3; ++n)
            if (B(KHT1Show + n * DwaHairToonKeys, n == 0)) h |= shade::HintHairToonS1 << n;
        need(!(h & shade::HintHairDiffuse) ||
                 !(has[KScatteringRadius] && has[KScatteringColor] && !shade::isZero(F(KScatteringRadius, 0)) &&
                   !shade::isBlack(C(KScatteringColor, 0))),
             o.getName() + ": subsurface hair diffuse is not shaded yet");
    }
    if (c.model == shade::ModelHairDiffuse) {
        h |= shade::HintHairDiffuse | shade::HintHairDiffuseIsOne;
        // Its subsurface share (DwaBase's mRequiresSubsurfaceParams): curves' subsurface scattering is not ported.
        need(!(has[KScatteringRadius] && has[KScatteringColor] && !shade::isZero(F(KScatteringRadius, 0)) &&
               !shade::isBlack(C(KScatteringColor, 0))),
             o.getName() + ": subsurface hair diffuse is not shaded yet");
    }
    u.hints = h;

    // resolveUniformParameters
    u.up.thinGeometry = (h & shade::HintThinGeometry) ? 1 : 0;
    u.up.preventLightCulling = (h & shade::HintPreventLightCulling) ? 1 : 0;
    u.up.specularModel = (h & shade::HintSpecular) ? I(KSpecularModel, 1) : 1;
    u.up.outerSpecularModel = 1;
    u.up.outerSpecularUseBending = 0;
    if (h & shade::HintOuterSpecular) {
        u.up.outerSpecularModel = I(KOuterSpecularModel, 1);
        u.up.outerSpecularUseBending = B(KOuterSpecularUseBending, false) ? 1 : 0;
    }
    // A toon material's toon models (2, 3) shade any remaining dielectric as GGX, as MoonRay's distribution switch does.
    need((u.up.specularModel <= 1 || (c.model == shade::ModelToon && u.up.specularModel <= 3)) && u.up.outerSpecularModel <= 1,
         o.getName() + ": invalid specular model");
    u.up.subsurface = (h & shade::HintSubsurface) ? unsigned(I(KSubsurface, 0)) : unsigned(shade::SubsurfaceNone);

    unsigned s = 0;
    auto set = [&](unsigned bit, bool on) { if (on) s |= bit; };
    set(shade::SetFuzzUseAbsorbingFibers, B(KFuzzUseAbsorbingFibers, false));
    set(shade::SetUseOuterSpecularNormal, B(KUseOuterSpecularNormal, false));
    set(shade::SetUseIndependentTransmissionRoughness, B(KUseIndTransRoughness, false));
    set(shade::SetUseDispersion, B(KUseDispersion, false));
    set(shade::SetUseIndependentWeftAttributes, B(KUseIndependentWeft, false));
    set(shade::SetUseUVsForThreadDirection, B(KUseUVsForThreadDirection, true));
    set(shade::SetIridescenceFlipHue, B(KIridescenceFlipHue, false));
    set(shade::SetSSSResolveSelfIntersections, B(KSSSResolveSelfIntersections, true));
    set(shade::SetCastsCaustics, B(KCastsCaustics, false));
    set(shade::SetHasNormalAAStrategy, has[KNormalAAStrategy]);
    u.settings = s;
    u.normalAAStrategy = unsigned(I(KNormalAAStrategy, 0));
    u.iridescenceApplyTo = unsigned(I(KIridescenceApplyTo, 0));
    u.iridescenceColorControl = unsigned(I(KIridescenceColorControl, 0));
    u.iridescenceRampMode = unsigned(I(KIridescenceRampMode, 0));
    u.diffuseTransmissionBlending = unsigned(I(KDiffuseTransmissionBlending, 1));
    need(u.iridescenceApplyTo <= 1 && u.iridescenceColorControl <= 1 && u.iridescenceRampMode <= 2 &&
             u.diffuseTransmissionBlending <= 1 && u.normalAAStrategy <= 1,
         o.getName() + ": invalid setting");
    u.toonDiffuseModel = shade::ToonDiffuseOrenNayar;
    if (has[KToonDiffuseModel]) {
        u.toonDiffuseModel = unsigned(I(KToonDiffuseModel, 0));
        need(u.toonDiffuseModel <= 1, o.getName() + ": invalid diffuse model");
    }
    set(shade::SetToonExtendRamp, has[KToonExtendRamp] && B(KToonExtendRamp, false));
    set(shade::SetToonSpecUseInputVectors, has[KTSUseInputVectors] && B(KTSUseInputVectors, false));
    set(shade::SetToonSpecEnableInputNormal, has[KTSEnableInputNormal] && B(KTSEnableInputNormal, false));
    set(shade::SetToonSpecEnableIndirect, has[KTSEnableIndirect] && B(KTSEnableIndirect, false));
    set(shade::SetHairShowR, B(KHairShowR, true));
    set(shade::SetHairShowTT, B(KHairShowTT, true));
    set(shade::SetHairShowTRT, B(KHairShowTRT, true));
    set(shade::SetHairShowGlint, B(KHairShowGlint, false));
    set(shade::SetHairShowTRRT, B(KHairShowTRRT, true));
    set(shade::SetHairUseTTRoughness, B(KHairUseTTRoughness, false));
    set(shade::SetHairUseTRTRoughness, B(KHairUseTRTRoughness, false));
    set(shade::SetHairUseOptimizedSampling, B(KHairUseOptimizedSampling, true));
    set(shade::SetHairDiffuseIndependentColors, B(KHairDiffuseIndependent, false));
    set(shade::SetGlitterCompensateDeformation, B(KGlitterCompensateDeformation, true));
    set(shade::SetGlitterApproximateForSecondaryRays, B(KGlitterApproximate, true));
    u.settings = s;
    u.hairFresnelType = unsigned(I(KHairFresnelType, 1));
    need(u.hairFresnelType <= 2, o.getName() + ": invalid fresnel type");
    // The toon ramps (DwaToonMaterial::update), as given: MoonRay's ramps expect sorted points.
    auto ramp = [&](unsigned kp, unsigned kv, unsigned ki, bool colors, unsigned& count, float* positions, V3* outColors, float* values,
                    unsigned* interpolators) {
        count = 0;
        if (!has[kp]) return;
        const auto& points = get<rdl::FloatVector>(o, name(kp));
        const auto& interps = get<rdl::IntVector>(o, name(ki));
        need(points.size() <= shade::DwaMaxToonRampPoints && interps.size() >= points.size(), o.getName() + ": invalid toon ramp");
        if (colors) need(get<rdl::RgbVector>(o, name(kv)).size() >= points.size(), o.getName() + ": invalid toon ramp");
        else need(get<rdl::FloatVector>(o, name(kv)).size() >= points.size(), o.getName() + ": invalid toon ramp");
        count = unsigned(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            need(std::isfinite(points[i]) && interps[i] >= 0 && interps[i] <= 6, o.getName() + ": invalid toon ramp point");
            positions[i] = points[i];
            interpolators[i] = unsigned(interps[i]);
            if (colors) {
                const auto c3 = get<rdl::RgbVector>(o, name(kv))[i];
                outColors[i] = V3{c3.r, c3.g, c3.b};
            } else {
                values[i] = get<rdl::FloatVector>(o, name(kv))[i];
            }
        }
    };
    ramp(KToonPositions, KToonColors, KToonInterpolations, true, u.toonRampCount, u.toonRampPositions, u.toonRampColors, nullptr,
         u.toonRampInterpolators);
    ramp(KTSPositions, KTSValues, KTSInterpolations, false, u.toonSpecRampCount, u.toonSpecRampPositions, nullptr,
         u.toonSpecRampValues, u.toonSpecRampInterpolators);
    // HairToonMaterial's specular sets: their models, ramps, input normals and settings.
    for (unsigned n = 0; n < 3; ++n) {
        const unsigned d = n * DwaHairToonKeys;
        shade::HairToonSpecUniforms& v = u.hairToonSpec[n];
        v = shade::HairToonSpecUniforms{};
        v.normalReg = shade::MapNoRegister;
        u.up.hairToonModel[n] = unsigned(I(KHT1Model + d, shade::ToonSpecularSurface));
        if (!has[KHT1Show + d]) continue;
        ramp(KHT1Positions + d, KHT1Values + d, KHT1Interpolations + d, false, v.rampCount, v.rampPositions, nullptr, v.rampValues,
             v.rampInterpolators);
        if (B(KHT1UseInputVectors + d, false)) v.settings |= shade::HairToonUseInputVectors;
        if (B(KHT1EnableIndirect + d, false)) v.settings |= shade::HairToonEnableIndirect;
        if (B(KHT1EnableInputNormal + d, false)) {
            v.settings |= shade::HairToonEnableInputNormal;
            v.normalReg = normalReg(KHT1InputNormal + d);
        }
    }
    u.toonSpecNormalReg = (u.settings & shade::SetToonSpecEnableInputNormal) ? normalReg(KTSInputNormal) : shade::MapNoRegister;
    u.normalReg = normalReg(KInputNormal);
    u.outerSpecularNormalReg = normalReg(KOuterSpecularNormal);
    u.fuzzNormalReg = normalReg(KFuzzNormal);
    if (u.normalReg != shade::MapNoRegister) u.settings |= shade::SetHasNormalMap;
    if ((h & shade::HintIridescence) && has[KIridescencePositions]) {
        const auto& points = get<rdl::FloatVector>(o, "iridescence_positions");
        const auto& colors = get<rdl::RgbVector>(o, "iridescence_colors");
        const auto& interps = get<rdl::IntVector>(o, "iridescence_interpolations");
        need(points.size() <= shade::DwaMaxIridescenceRampPoints && colors.size() >= points.size() && interps.size() >= points.size(),
             o.getName() + ": invalid iridescence ramp");
        u.iridescenceRampCount = unsigned(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            need(std::isfinite(points[i]) && (i == 0 || points[i] >= points[i - 1]) && interps[i] >= 0 && interps[i] <= 6,
                 o.getName() + ": invalid iridescence ramp");
            u.iridescenceRampPositions[i] = points[i];
            u.iridescenceRampColors[i] = V3{colors[i].r, colors[i].g, colors[i].b};
            u.iridescenceRampInterpolators[i] = unsigned(interps[i]);
        }
    }

    in.refractiveIndex = F(KRefractiveIndex, c.model == shade::ModelHair ? 1.45f : 1.5f);
    in.hairColor = C(KHairColor, 1);
    in.hairCuticleLayerThickness = F(KHairCuticle, 0.1f);
    in.hairROffset = F(KHairROffset, 0);
    in.hairRRoughness = F(KHairRRoughness, 0.5f);
    in.hairRTint = C(KHairRTint, 1);
    in.hairTRTOffset = F(KHairTRTOffset, 0);
    in.hairTRTRoughness = F(KHairTRTRoughness, 0.4f);
    in.hairTRTTint = C(KHairTRTTint, 1);
    in.hairGlintRoughness = F(KHairGlintRoughness, 0.5f);
    in.hairGlintEccentricity = F(KHairGlintEccentricity, 0.85f);
    in.hairGlintMinTwists = F(KHairGlintMinTwists, 1.5f);
    in.hairGlintMaxTwists = F(KHairGlintMaxTwists, 2.5f);
    in.hairGlintSaturation = F(KHairGlintSaturation, 0.5f);
    in.hairTTOffset = F(KHairTTOffset, 0);
    in.hairTTRoughness = F(KHairTTRoughness, 0.1f);
    in.hairTTAzimuthalRoughness = F(KHairTTAzimRoughness, 1.0f);
    in.hairTTTint = C(KHairTTTint, 1);
    in.hairTTSaturation = F(KHairTTSaturation, 1.0f);
    in.hairDiffuseFrontColor = C(KHairDiffuseFront, 1);
    in.hairDiffuseBackColor = C(KHairDiffuseBack, 1);
    in.hairSubsurfaceBlend = F(KHairSubsurfaceBlend, 1.0f);
    in.glitter = F(KGlitter, 0.0f);
    in.glitterStyleAFrequency = F(KGlitterStyleAFrequency, 1.0f);
    in.glitterColorA = C(KGlitterColorA, 1);
    in.glitterSizeA = F(KGlitterSizeA, 1.0f);
    in.glitterRoughnessA = F(KGlitterRoughnessA, 0.14f);
    in.glitterStyleBFrequency = F(KGlitterStyleBFrequency, 1.0f);
    in.glitterColorB = C(KGlitterColorB, 1);
    in.glitterSizeB = F(KGlitterSizeB, 1.0f);
    in.glitterRoughnessB = F(KGlitterRoughnessB, 0.14f);
    in.glitterHueVariation = F(KGlitterHueVariation, 0.0f);
    in.glitterSaturationVariation = F(KGlitterSaturationVariation, 0.0f);
    in.glitterValueVariation = F(KGlitterValueVariation, 0.0f);
    in.glitterDensity = F(KGlitterDensity, 1.0f);
    in.glitterJitter = F(KGlitterJitter, 1.0f);
    in.glitterOrientationRandomness = F(KGlitterOrientationRandomness, 0.15f);
    in.specular = F(KSpecular, 0);
    in.roughness = F(KRoughness, 0);
    in.anisotropy = F(KAnisotropy, 0);
    if (has[KShadingTangent]) {
        auto t = get<rdl::Vec2f>(o, "shading_tangent");
        need(std::isfinite(t.x) && std::isfinite(t.y), o.getName() + ": invalid shading tangent");
        in.shadingTangent[0] = t.x;
        in.shadingTangent[1] = t.y;
    }
    in.metallic = F(KMetallic, 0);
    in.metallicColor = C(KMetallicColor, 0);
    in.metallicEdgeColor = C(KMetallicEdgeColor, 0);
    in.iridescence = F(KIridescence, 0);
    in.iridescencePrimary = C(KIridescencePrimary, 0);
    in.iridescenceSecondary = C(KIridescenceSecondary, 0);
    in.iridescenceThickness = F(KIridescenceThickness, 0);
    in.iridescenceExponent = F(KIridescenceExponent, 0);
    in.iridescenceAt0 = F(KIridescenceAt0, 0);
    in.iridescenceAt90 = F(KIridescenceAt90, 1);
    in.transmission = F(KTransmission, 0);
    in.transmissionColor = C(KTransmissionColor, 0);
    in.independentTransmissionIor = F(KIndTransIor, 1.5f);
    in.independentTransmissionRoughness = F(KIndTransRoughness, 0);
    in.dispersionAbbeNumber = F(KDispersionAbbe, 0);
    in.fuzz = F(KFuzz, 1);
    in.fuzzAlbedo = C(KFuzzAlbedo, 0);
    in.fuzzRoughness = F(KFuzzRoughness, 0);
    in.fuzzNormalDial = F(KFuzzNormalDial, 1);
    in.outerSpecular = F(KOuterSpecular, 0);
    in.outerSpecularIor = F(KOuterSpecularIor, 1.5f);
    in.outerSpecularRoughness = F(KOuterSpecularRoughness, 0);
    in.outerSpecularThickness = F(KOuterSpecularThickness, 0);
    in.outerSpecularAttenuationColor = C(KOuterSpecularAttenuationColor, 0);
    in.outerSpecularNormalDial = F(KOuterSpecularNormalDial, 1);
    in.fabricDiffuseScattering = F(KFabricDiffuseScattering, 0);
    in.warpColor = C(KWarpColor, 0);
    in.warpRoughness = F(KWarpRoughness, 0);
    in.weftColor = C(KWeftColor, 0);
    in.weftRoughness = F(KWeftRoughness, 0);
    if (has[KWarpThreadDirection]) {
        auto d = get<rdl::Vec3f>(o, "warp_thread_direction");
        in.warpThreadDirection = V3{d.x, d.y, d.z};
        need(surface::finite(in.warpThreadDirection), o.getName() + ": invalid thread direction");
    }
    in.warpThreadCoverage = F(KWarpThreadCoverage, 0);
    in.warpThreadElevation = F(KWarpThreadElevation, 0);
    in.albedo = C(KAlbedo, 0);
    in.diffuseRoughness = F(KDiffuseRoughness, 0);
    in.scatteringColor = C(KScatteringColor, 0);
    in.scatteringRadius = F(KScatteringRadius, 0);
    in.creaseAttenuation = F(KCreaseAttenuation, 0);
    in.diffuseTransmission = F(KDiffuseTransmission, 1);
    in.diffuseTransmissionColor = C(KDiffuseTransmissionColor, 0);
    in.emission = C(KEmission, 0);
    in.presence = F(KPresence, 1);
    in.inputNormalDial = F(KInputNormalDial, 1);
    in.normalAADial = F(KNormalAADial, 1);
    in.toonTerminatorShift = F(KToonTerminatorShift, 0);
    in.toonFlatness = F(KToonFlatness, 0);
    in.toonFlatnessFalloff = F(KToonFlatnessFalloff, 0);
    in.toonRampInputScale = F(KToonRampInputScale, 1);
    in.toonSpecIntensity = F(KTSIntensity, 1);
    in.toonSpecFresnelBlend = F(KTSFresnelBlend, 1);
    in.toonSpecTint = C(KTSTint, 1);
    in.toonSpecStretchU = F(KTSStretchU, 1);
    in.toonSpecStretchV = F(KTSStretchV, 1);
    for (unsigned k : {KTSInputU, KTSInputV}) {
        V3 v{0, 0, 0};
        if (has[k]) {
            const auto d = get<rdl::Vec3f>(o, name(k));
            v = V3{d.x, d.y, d.z};
            need(surface::finite(v), o.getName() + ": invalid " + name(k));
        }
        (k == KTSInputU ? in.toonSpecInputU : in.toonSpecInputV) = v;
    }
    in.toonSpecRampInputScale = F(KTSRampInputScale, 1);
    in.toonSpecInputNormalDial = F(KTSInputNormalDial, 1);
    in.toonSpecIndirectIntensity = F(KTSIndirectIntensity, 1);
    in.toonSpecIndirectRoughness = F(KTSIndirectRoughness, 0.5f);
    // HairToonMaterial's hair diffuse weight and specular sets
    in.hairDiffuseWeight = F(KHairDiffuseWeight, 1.0f);
    for (unsigned n = 0; n < 3; ++n) {
        const unsigned d = n * DwaHairToonKeys;
        shade::HairToonSpecInputs& t = in.hairToonSpec[n];
        t.intensity = F(KHT1Intensity + d, 1);
        t.fresnelBlend = F(KHT1FresnelBlend + d, 1);
        t.roughness = F(KHT1Roughness + d, 1);
        t.tint = C(KHT1Tint + d, 1);
        t.stretchU = F(KHT1StretchU + d, 1);
        t.stretchV = F(KHT1StretchV + d, 1);
        for (unsigned k : {KHT1InputU + d, KHT1InputV + d}) {
            V3 v{0, 0, 0};
            if (has[k]) {
                const auto e = get<rdl::Vec3f>(o, name(k));
                v = V3{e.x, e.y, e.z};
                need(surface::finite(v), o.getName() + ": invalid " + name(k));
            }
            (k == KHT1InputU + d ? t.inputU : t.inputV) = v;
        }
        t.rampInputScale = F(KHT1RampInputScale + d, 1);
        t.inputNormalDial = F(KHT1InputNormalDial + d, 1);
        t.indirectIntensity = F(KHT1IndirectIntensity + d, 1);
        t.indirectRoughness = F(KHT1IndirectRoughness + d, 0.5f);
        for (unsigned i = 0; i < shade::DwaMaxToonRampPoints; ++i) {
            const std::string idx = std::to_string(i);
            t.rampMultipliers[i] = has[KHT1Mult + d] ? get<rdl::Float>(o, (std::string(name(KHT1Mult + d)) + idx).c_str()) : 1.0f;
            t.rampOffsets[i] = has[KHT1Offset + d] ? get<rdl::Float>(o, (std::string(name(KHT1Offset + d)) + idx).c_str()) : 0.0f;
            need(std::isfinite(t.rampMultipliers[i]) && std::isfinite(t.rampOffsets[i]), o.getName() + ": invalid specular ramp multiplier");
        }
    }
    // The multi attributes: a class without them has their ISPC defaults (one, zero).
    for (unsigned i = 0; i < shade::DwaMaxToonRampPoints; ++i) {
        const std::string n = std::to_string(i);
        in.toonRampColorMultipliers[i] = shade::rgb(1);
        in.toonRampPositionOffsets[i] = 0;
        in.toonSpecRampMultipliers[i] = 1;
        in.toonSpecRampOffsets[i] = 0;
        if (has[KToonRampColorMult]) {
            const auto c3 = get<rdl::Rgb>(o, (std::string(name(KToonRampColorMult)) + n).c_str());
            in.toonRampColorMultipliers[i] = V3{c3.r, c3.g, c3.b};
        }
        if (has[KToonRampOffset]) in.toonRampPositionOffsets[i] = get<rdl::Float>(o, (std::string(name(KToonRampOffset)) + n).c_str());
        if (has[KTSMult]) in.toonSpecRampMultipliers[i] = get<rdl::Float>(o, (std::string(name(KTSMult)) + n).c_str());
        if (has[KTSOffset]) in.toonSpecRampOffsets[i] = get<rdl::Float>(o, (std::string(name(KTSOffset)) + n).c_str());
        need(surface::finite(in.toonRampColorMultipliers[i]) && std::isfinite(in.toonRampPositionOffsets[i]) &&
                 std::isfinite(in.toonSpecRampMultipliers[i]) && std::isfinite(in.toonSpecRampOffsets[i]),
             o.getName() + ": invalid toon ramp multiplier or offset");
    }
    // Presence (dwaTreePresence): the integrator's presence continuation takes what is not present.
    need(std::isfinite(in.presence), o.getName() + ": invalid presence");
    return r;
}
//----------------------------------------------------------------------------
// The combinators: a material's tree of nodes in the pool (shade_dwalayer.h), with the uniform
// parameters and castsCaustics their update() and getCastsCaustics() compute.
inline bool dwaCombinator(const std::string& cls) {
    return cls == "DwaLayerMaterial" || cls == "DwaMixMaterial" || cls == "DwaTwoSidedMaterial" || cls == "DwaSwitchMaterial" ||
           cls == "DwaColorCorrectMaterial" || cls == "DwaAdjustMaterial";
}
// HairLayerMaterial and HairColorCorrectMaterial: the hair combinators (Blending.cc registerHairLayerable).
inline bool hairCombinator(const std::string& cls) { return cls == "HairLayerMaterial" || cls == "HairColorCorrectMaterial"; }
inline bool dwaLayerable(const rdl::SceneObject* o) {
    if (!o) return false;
    const std::string cls = o->getSceneClass().getName();
    return dwaClass(cls) || dwaCombinator(cls) || hairCombinator(cls);
}
// Blending.cc blendUniformParameters: where the two disagree, the material's fallback.
inline void blendUniformParameters(const shade::DwaUniformParams& u0, const shade::DwaUniformParams& u1, shade::DwaUniformParams& u,
                                   int fbSpecular, int fbOuter, bool fbBending, int fbBssrdf, bool fbThin, bool fbCulling) {
    if (u0.specularModel != u1.specularModel) u.specularModel = unsigned(fbSpecular);
    if (u0.outerSpecularModel != u1.outerSpecularModel) u.outerSpecularModel = unsigned(fbOuter);
    if (u0.outerSpecularUseBending != u1.outerSpecularUseBending) u.outerSpecularUseBending = fbBending ? 1 : 0;
    if (u.subsurface != unsigned(fbBssrdf)) {
        if (u0.subsurface != u1.subsurface) {
            if (u0.subsurface == shade::SubsurfaceNone) u.subsurface = u1.subsurface;
            else if (u1.subsurface == shade::SubsurfaceNone) u.subsurface = u0.subsurface;
            else u.subsurface = unsigned(fbBssrdf);
        } else if (u.subsurface == shade::SubsurfaceNone) {
            u.subsurface = u0.subsurface;
        }
    }
    if (u0.thinGeometry != u1.thinGeometry) u.thinGeometry = fbThin ? 1 : 0;
    if (u0.preventLightCulling != u1.preventLightCulling) u.preventLightCulling = fbCulling ? 1 : 0;
    // Blended parameters start with no light sets (initParameters) and blending gives them none.
    u.diffuseLightSet = u.specularLightSet = ~0u;
}

struct DwaTreeBuilder {
    std::vector<float>& pool;
    MapProgramBuilder maps;
    struct Leaf {
        unsigned record, bindings, bindingCount;
    };
    std::map<const rdl::SceneObject*, Leaf> records;  // a leaf's DwaRecord and bindings, once per object
    unsigned depth = 0, colorCorrections = 0;
    unsigned dwaAncestors = 0;  // DwaBase combinators above the node being built (they drop hair, as MoonRay's do)
    bool glitter = false, refP = false;  // a leaf has glitter tables; one in reference space

    static const rdl::SceneObject* obj(const rdl::SceneObject& o, const char* name) { return get<rdl::SceneObject*>(o, name); }
    static std::vector<const rdl::SceneObject*> mixInputs(const rdl::SceneObject& o) {
        std::vector<const rdl::SceneObject*> v;
        for (int i = 0; i < int(shade::DwaMaxMixInputs); ++i) {
            const rdl::SceneObject* m = obj(o, ("material" + std::to_string(i)).c_str());
            if (!dwaLayerable(m)) {
                need(!m, o.getName() + ": a mix input is not layerable");
                break;
            }
            v.push_back(m);
        }
        return v;
    }
    static const rdl::SceneObject* switchChoice(const rdl::SceneObject& o) {
        const int choice = get<rdl::Int>(o, "choice");
        need(choice >= 0 && choice < 64, o.getName() + ": choice out of range");
        const rdl::SceneObject* m = obj(o, ("material" + std::to_string(choice)).c_str());
        need(!m || dwaLayerable(m), o.getName() + ": the chosen material is not layerable");
        return m;
    }
    // A combinator's own attributes: any value, no map bindings yet.
    static void admitCombinator(const rdl::SceneObject& o) {
        std::set<std::string> all;
        for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i) all.insert((*i)->getName());
        attributes(o, all);
    }
    // getCastsCaustics()
    static bool castsCaustics(const rdl::SceneObject* o) {
        if (!o) return false;
        const std::string cls = o->getSceneClass().getName();
        if (const DwaClass* c = dwaClass(cls)) {
            for (unsigned k : c->keys)
                if (k == KCastsCaustics) return get<rdl::Bool>(*o, "casts_caustics");
            return false;
        }
        if (cls == "DwaLayerMaterial") {
            const rdl::SceneObject *a = obj(*o, "material_A"), *b = obj(*o, "material_B");
            if (a && b) return castsCaustics(a) || castsCaustics(b);
            return castsCaustics(b ? b : a);
        }
        if (cls == "DwaMixMaterial") {
            for (auto* m : mixInputs(*o))
                if (castsCaustics(m)) return true;
            return false;
        }
        if (cls == "DwaTwoSidedMaterial") {
            const rdl::SceneObject *f = obj(*o, "front_material"), *b = obj(*o, "back_material");
            return f && b && (castsCaustics(f) || castsCaustics(b));
        }
        if (cls == "DwaSwitchMaterial") return castsCaustics(switchChoice(*o));
        if (cls == "DwaColorCorrectMaterial") return castsCaustics(obj(*o, "input_material"));
        if (cls == "HairLayerMaterial") {
            const rdl::SceneObject *a = obj(*o, "hair_material_A"), *b = obj(*o, "hair_material_B");
            if (a && b) return castsCaustics(a) || castsCaustics(b);
            return castsCaustics(b ? b : a);
        }
        if (cls == "HairColorCorrectMaterial") return castsCaustics(obj(*o, "input_hair_material"));
        if (cls == "DwaAdjustMaterial") {
            const rdl::SceneObject* in = obj(*o, "input_material");
            if (!in) return false;
            const int mode = get<rdl::Int>(*o, "casts_caustics");
            return mode == 0 ? castsCaustics(in) : mode == 1;
        }
        return false;
    }
    shade::DwaRecord dwaRecord(const rdl::SceneObject& o) { return admitDwa(o, *dwaClass(o.getSceneClass().getName())); }
    // hasGlitter(): a DwaBase's shown, non-zero glitter; a layer's, mix's, switch's, colour correction's or
    // adjustment's inputs' (a two-sided material reports none).
    static bool hasGlitter(const rdl::SceneObject* o) {
        if (!o) return false;
        const std::string cls = o->getSceneClass().getName();
        if (const DwaClass* c = dwaClass(cls)) {
            if (std::find(c->keys.begin(), c->keys.end(), KShowGlitter) == c->keys.end()) return false;
            return get<rdl::Bool>(*o, "show_glitter") && !shade::isZero(get<rdl::Float>(*o, "glitter"));
        }
        if (cls == "DwaLayerMaterial") return hasGlitter(obj(*o, "material_A")) || hasGlitter(obj(*o, "material_B"));
        if (cls == "DwaMixMaterial") {
            for (auto* m : mixInputs(*o))
                if (hasGlitter(m)) return true;
            return false;
        }
        if (cls == "DwaSwitchMaterial") return hasGlitter(switchChoice(*o));
        if (cls == "DwaColorCorrectMaterial" || cls == "DwaAdjustMaterial") return hasGlitter(obj(*o, "input_material"));
        return false;
    }
    // resolveUniformParameters()
    shade::DwaUniformParams uniformParams(const rdl::SceneObject* o) {
        shade::DwaUniformParams u{1, 1, 0, shade::SubsurfaceNone, 0, 0};  // DwaBaseLayerable::initUniformParameters
        if (!o) return u;
        const std::string cls = o->getSceneClass().getName();
        if (dwaClass(cls)) return dwaRecord(*o).u.up;
        auto fallbacks = [&](const shade::DwaUniformParams& u0, const shade::DwaUniformParams& u1, shade::DwaUniformParams& out, bool thin) {
            blendUniformParameters(u0, u1, out, get<rdl::Int>(*o, "fallback_specular_model"), get<rdl::Int>(*o, "fallback_outer_specular_model"),
                                   get<rdl::Bool>(*o, "fallback_clearcoat_use_bending"), get<rdl::Int>(*o, "fallback_bssrdf"), thin,
                                   get<rdl::Bool>(*o, "fallback_prevent_light_culling"));
        };
        if (cls == "DwaLayerMaterial") {
            const rdl::SceneObject *a = obj(*o, "material_A"), *b = obj(*o, "material_B");
            if (a && b) {
                const auto ua = uniformParams(a), ub = uniformParams(b);
                u = ua;
                fallbacks(ua, ub, u, get<rdl::Bool>(*o, "fallback_thin_geometry"));
            } else if (b) {
                u = uniformParams(b);
            }
        } else if (cls == "DwaMixMaterial") {
            const auto inputs = mixInputs(*o);
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto cur = uniformParams(inputs[i]);
                if (i) fallbacks(cur, uniformParams(inputs[i - 1]), u, get<rdl::Bool>(*o, "fallback_thin_geometry"));
                else u = cur;
            }
        } else if (cls == "DwaTwoSidedMaterial") {
            const auto uf = uniformParams(obj(*o, "front_material")), ub = uniformParams(obj(*o, "back_material"));
            u = uf;
            fallbacks(uf, ub, u, true);
            u.thinGeometry = 1;  // a two-sided material is on thin geometry
        } else if (cls == "DwaSwitchMaterial") {
            if (const rdl::SceneObject* m = switchChoice(*o)) u = uniformParams(m);
        } else if (cls == "DwaColorCorrectMaterial") {
            if (const rdl::SceneObject* m = obj(*o, "input_material")) u = uniformParams(m);
        } else if (cls == "HairLayerMaterial") {
            // HairLayerMaterial::resolveUniformParameters: both inputs' (build() requires both), its fallback BSSRDF.
            const auto ua = uniformParams(obj(*o, "hair_material_A")), ub = uniformParams(obj(*o, "hair_material_B"));
            u = ua;
            blendUniformParameters(ua, ub, u, 1, 1, false, get<rdl::Int>(*o, "fallback_bssrdf"), false, false);
        } else if (cls == "HairColorCorrectMaterial") {
            if (const rdl::SceneObject* m = obj(*o, "input_hair_material")) u = uniformParams(m);
        } else if (cls == "DwaAdjustMaterial") {
            if (const rdl::SceneObject* m = obj(*o, "input_material")) {
                u = uniformParams(m);
                const int thin = get<rdl::Int>(*o, "thin_geometry");
                if (thin == 1) u.thinGeometry = 1;
                else if (thin == 2) u.thinGeometry = 0;
            }
        }
        return u;
    }
    unsigned addBindings(const std::vector<shade::MapBinding>& bindings) {
        const unsigned offset = unsigned(pool.size());
        for (auto& b : bindings) {
            const float* f = reinterpret_cast<const float*>(&b);
            pool.insert(pool.end(), f, f + sizeof(b) / sizeof(float));
        }
        return offset;
    }
    Leaf leaf(const rdl::SceneObject& o) {
        auto it = records.find(&o);
        if (it != records.end()) return it->second;
        std::vector<shade::MapBinding> bindings;
        const shade::DwaRecord r = admitDwa(o, *dwaClass(o.getSceneClass().getName()), &maps, &bindings);
        if (r.u.glitterTables != shade::GlitterNone) {
            glitter = true;
            if (r.u.glitterSpace == shade::GlitterReferenceSpace) refP = true;
        }
        Leaf l;
        l.record = unsigned(pool.size());
        const float* f = reinterpret_cast<const float*>(&r);
        pool.insert(pool.end(), f, f + shade::DwaRecordFloats);
        l.bindings = addBindings(bindings);
        l.bindingCount = unsigned(bindings.size());
        records[&o] = l;
        return l;
    }
    // A combinator's bound values (value[] offsets): the listed attributes' maps.
    void nodeBindings(const rdl::SceneObject& o, const std::vector<std::pair<std::string, unsigned>>& attrs, shade::DwaNode& n) {
        std::vector<shade::MapBinding> bindings;
        maps.bindingsOf(o, attrs, bindings);
        n.bindings = addBindings(bindings);
        n.bindingCount = unsigned(bindings.size());
    }
    static void admitCombinatorBound(const rdl::SceneObject& o, const std::set<std::string>& boundable) {
        std::set<std::string> all;
        for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i) all.insert((*i)->getName());
        attributes(o, all, boundable);
    }
    unsigned addNode(const shade::DwaNode& n) {
        const unsigned offset = unsigned(pool.size());
        const float* f = reinterpret_cast<const float*>(&n);
        pool.insert(pool.end(), f, f + sizeof(shade::DwaNode) / sizeof(float));
        return offset;
    }
    static float f1(const rdl::SceneObject& o, const char* name) {
        const float f = get<rdl::Float>(o, name);
        need(std::isfinite(f), o.getName() + ": invalid " + name);
        return f;
    }
    static void c3(const rdl::SceneObject& o, const char* name, float* out) {
        const auto c = get<rdl::Rgb>(o, name);
        need(std::isfinite(c.r) && std::isfinite(c.g) && std::isfinite(c.b), o.getName() + ": invalid " + name);
        out[0] = c.r;
        out[1] = c.g;
        out[2] = c.b;
    }
    // The node of o (DwaNoNode for none), and the parameter slots it needs counting its own.
    unsigned build(const rdl::SceneObject* o, unsigned& slots) {
        slots = 0;
        if (!o) return shade::DwaNoNode;
        need(++depth <= 20, o->getName() + ": material tree too deep");
        const std::string cls = o->getSceneClass().getName();
        shade::DwaNode n{};
        n.child[0] = n.child[1] = shade::DwaNoNode;
        if (dwaClass(cls)) {
            // Hair materials combine only through the hair combinators: MoonRay's DwaBase combinators drop hair.
            const auto model = dwaClass(cls)->model;
            need(dwaAncestors == 0 || (model != shade::ModelHair && model != shade::ModelHairDiffuse && model != shade::ModelHairToon),
                 o->getName() + ": hair in a layer");
            n.kind = shade::NodeLeaf;
            const Leaf l = leaf(*o);
            n.child[0] = l.record;
            n.bindings = l.bindings;
            n.bindingCount = l.bindingCount;
            slots = 1;
        } else if (hairCombinator(cls)) {
            need(dwaAncestors == 0, o->getName() + ": hair in a layer");
            admitCombinatorBound(*o, {"mask", "mix", "hue_shift", "saturation", "gain"});
            if (cls == "HairLayerMaterial") {
                // MoonRay's update() resolves both inputs' uniform parameters: it needs both.
                const rdl::SceneObject *a = obj(*o, "hair_material_A"), *b = obj(*o, "hair_material_B");
                need(a && b && dwaLayerable(a) && dwaLayerable(b), o->getName() + ": a hair layer needs two layerable inputs");
                n.kind = shade::NodeHairLayer;
                n.colorSpace = unsigned(get<rdl::Int>(*o, "blend_color_space"));
                need(n.colorSpace <= 2, o->getName() + ": invalid colour space");
                n.value[0] = f1(*o, "mask");
                nodeBindings(*o, {{"mask", 0}}, n);
                unsigned sb, sa;
                n.child[0] = build(b, sb);
                n.child[1] = build(a, sa);
                n.childCaustics[0] = castsCaustics(b);
                n.childCaustics[1] = castsCaustics(a);
                slots = std::max({3u, 1 + sb, 2 + sa});
            } else {  // HairColorCorrectMaterial: its TMI is not bindable
                const rdl::SceneObject* m = obj(*o, "input_hair_material");
                need(!m || dwaLayerable(m), o->getName() + ": the input is not layerable");
                n.kind = shade::NodeHairColorCorrect;
                n.value[0] = get<rdl::Bool>(*o, "on") ? 1.0f : 0.0f;
                n.value[1] = f1(*o, "mix");
                n.value[2] = f1(*o, "hue_shift");
                n.value[3] = f1(*o, "saturation");
                n.value[4] = f1(*o, "gain");
                n.value[5] = get<rdl::Bool>(*o, "TMI_enabled") ? 1.0f : 0.0f;
                c3(*o, "TMI", n.value + 6);
                nodeBindings(*o, {{"mix", 1}, {"hue_shift", 2}, {"saturation", 3}, {"gain", 4}}, n);
                n.child[0] = build(m, slots);
                n.childCaustics[0] = castsCaustics(m);
            }
        } else {
            need(dwaCombinator(cls), o->getName() + ": not a layerable material");
            admitCombinatorBound(*o, {"mask", "mix", "hue_shift", "saturation", "gain", "TMI", "emission"});
            ++dwaAncestors;
            if (cls == "DwaLayerMaterial") {
                const rdl::SceneObject *a = obj(*o, "material_A"), *b = obj(*o, "material_B");
                need(b != nullptr, o->getName() + ": no material_B");
                need(dwaLayerable(b) && (!a || dwaLayerable(a)), o->getName() + ": an input is not layerable");
                n.kind = shade::NodeLayer;
                // Both inputs' glitter: the layer's own Glitter object from its fallback glitter attributes (not ported).
                need(!(a && hasGlitter(a) && hasGlitter(b)), o->getName() + ": glitter on both inputs of a layer is not shaded yet");
                n.colorSpace = unsigned(get<rdl::Int>(*o, "blend_color_space"));
                need(n.colorSpace <= 2, o->getName() + ": invalid colour space");
                n.value[0] = f1(*o, "mask");
                nodeBindings(*o, {{"mask", 0}}, n);
                unsigned sb, sa;
                n.child[0] = build(b, sb);
                n.child[1] = build(a, sa);
                // B alone in the layer's slot; blended: B one above, A two above.
                slots = a ? std::max({3u, 1 + sb, 2 + sa}) : sb;
            } else if (cls == "DwaMixMaterial") {
                const auto inputs = mixInputs(*o);
                need(!inputs.empty(), o->getName() + ": no inputs");
                for (size_t i = 1; i < inputs.size(); ++i)
                    need(!(hasGlitter(inputs[i - 1]) && hasGlitter(inputs[i])),
                         o->getName() + ": glitter on neighbouring inputs of a mix is not shaded yet");
                n.kind = shade::NodeMix;
                n.colorSpace = unsigned(get<rdl::Int>(*o, "blend_color_space"));
                need(n.colorSpace <= 2, o->getName() + ": invalid colour space");
                const unsigned count = unsigned(inputs.size());
                n.value[0] = f1(*o, "mix");
                nodeBindings(*o, {{"mix", 0}}, n);
                const bool remap = get<rdl::Bool>(*o, "remap_mix_to_inputs");
                n.value[1] = remap ? 1.0f : float(count - 1);
                n.value[2] = remap ? float(int(count) - 1) : 1.0f;
                const int interpolation = get<rdl::Int>(*o, "mix_interpolation");
                need(interpolation >= 0 && interpolation <= 3, o->getName() + ": invalid interpolation");
                n.value[3] = float(interpolation);
                std::vector<unsigned> nodes;
                slots = count > 1 ? 3 : 1;
                for (auto* m : inputs) {
                    unsigned s;
                    nodes.push_back(build(m, s));
                    slots = std::max(slots, (count > 1 ? 2 : 0) + s);
                }
                n.child[0] = unsigned(pool.size());
                n.child[1] = count;
                for (unsigned v : nodes) {
                    float f;
                    std::memcpy(&f, &v, sizeof f);
                    pool.push_back(f);
                }
            } else if (cls == "DwaTwoSidedMaterial") {
                const rdl::SceneObject *f = obj(*o, "front_material"), *b = obj(*o, "back_material");
                need((!f || dwaLayerable(f)) && (!b || dwaLayerable(b)), o->getName() + ": an input is not layerable");
                n.kind = shade::NodeTwoSided;
                unsigned sf, sb;
                n.child[0] = build(f, sf);
                n.child[1] = build(b, sb);
                n.childCaustics[0] = castsCaustics(f);
                n.childCaustics[1] = castsCaustics(b);
                slots = std::max(sf, sb);
            } else if (cls == "DwaSwitchMaterial") {
                const rdl::SceneObject* m = switchChoice(*o);
                n.kind = shade::NodeSwitch;
                n.child[0] = build(m, slots);
                n.childCaustics[0] = castsCaustics(m);
            } else if (cls == "DwaColorCorrectMaterial") {
                const rdl::SceneObject* m = obj(*o, "input_material");
                need(!m || dwaLayerable(m), o->getName() + ": the input is not layerable");
                n.kind = shade::NodeColorCorrect;
                n.value[0] = get<rdl::Bool>(*o, "on") ? 1.0f : 0.0f;
                n.value[1] = f1(*o, "mix");
                n.value[2] = f1(*o, "hue_shift");
                n.value[3] = f1(*o, "saturation");
                n.value[4] = f1(*o, "gain");
                n.value[5] = get<rdl::Bool>(*o, "TMI_enabled") ? 1.0f : 0.0f;
                c3(*o, "TMI", n.value + 6);
                nodeBindings(*o, {{"mix", 1}, {"hue_shift", 2}, {"saturation", 3}, {"gain", 4}, {"TMI", 6}}, n);
                need(++colorCorrections <= shade::DwaMaxColorCorrections, o->getName() + ": too many nested colour corrections");
                n.child[0] = build(m, slots);
                --colorCorrections;
                n.childCaustics[0] = castsCaustics(m);
            } else {  // DwaAdjustMaterial (its primitive-attribute adjustments need attributes XPU2 meshes do not carry)
                const rdl::SceneObject* m = obj(*o, "input_material");
                need(!m || dwaLayerable(m), o->getName() + ": the input is not layerable");
                n.kind = shade::NodeAdjust;
                n.value[0] = get<rdl::Bool>(*o, "on") ? 1.0f : 0.0f;
                n.value[1] = f1(*o, "mix");
                const int mode = get<rdl::Int>(*o, "emission_mode");
                need(mode >= 0 && mode <= 2, o->getName() + ": invalid emission mode");
                n.value[2] = float(mode);
                c3(*o, "emission", n.value + 3);
                nodeBindings(*o, {{"mix", 1}, {"emission", 3}}, n);
                n.value[6] = get<rdl::Bool>(*o, "disable_clearcoat") ? 1.0f : 0.0f;
                n.value[7] = get<rdl::Bool>(*o, "disable_specular") ? 1.0f : 0.0f;
                n.value[8] = get<rdl::Bool>(*o, "disable_diffuse") ? 1.0f : 0.0f;
                n.child[0] = build(m, slots);
                n.childCaustics[0] = castsCaustics(m);
            }
            --dwaAncestors;
        }
        --depth;
        return addNode(n);
    }
};
// The material record of a DwaBase-family material or combinator; slots grows to what its tree needs.
inline shade::MaterialRecord admitDwaTree(const rdl::SceneObject& o, std::vector<float>& pool, unsigned& slots,
                                          TextureContext& textures, bool* glitter = nullptr, bool* randomWalk = nullptr) {
    DwaTreeBuilder b{pool, MapProgramBuilder{pool, textures}};
    shade::DwaTree tree{};
    unsigned needed = 0;
    tree.root = b.build(&o, needed);
    need(needed <= shade::DwaMaxSlots, o.getName() + ": material tree too wide");
    tree.up = b.uniformParams(&o);
    tree.castsCaustics = DwaTreeBuilder::castsCaustics(&o) ? 1 : 0;
    tree.slots = std::max(needed, 1u);
    tree.requiresRefP = b.refP ? 1 : 0;
    slots = std::max(slots, tree.slots);
    shade::MaterialRecord rec;
    rec.kind = shade::MatDwaBase;
    rec.offset = unsigned(pool.size());
    rec.count = shade::DwaTreeFloats;
    const float* f = reinterpret_cast<const float*>(&tree);
    pool.insert(pool.end(), f, f + shade::DwaTreeFloats);
    rec.maps = b.maps.finish();
    if (b.maps.polyVertices) rec.flags |= shade::MatPolyVertices;
    if (glitter && b.glitter) *glitter = true;
    if (randomWalk && tree.up.subsurface == shade::SubsurfaceRandomWalk) *randomWalk = true;
    return rec;
}
}

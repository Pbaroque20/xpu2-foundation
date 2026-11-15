// Private XPU2 shading library, shared CPU/CUDA: MoonRay's DwaBase material
// and its family (DwaBaseMaterial, DwaMetal, DwaRefractive, DwaSolidDielectric,
// DwaFabric, DwaVelvet, DwaEmissive). The host resolves each material's
// attribute values and update() hints into a DwaRecord; the devices run
// DWABASE_resolveParameters and DWABASE_createLobes (moonshine
// lib/material/dwabase: DwaBase.ispc, DwaBaseLayerable.ispc). Derived from
// MoonRay Apache-2.0, Copyright 2023-2024 DreamWorks Animation LLC. See
// MOONRAY_LICENSE.txt.
#pragma once
#include "shade_builder.h"
#include "shade_glitter.h"
#include "shade_colorcorrect.h"
#include "shade_maps.h"
namespace xpu2 { namespace interactive { namespace shade {

constexpr U32 DwaMaxToonRampPoints = 10, DwaMaxIridescenceRampPoints = 10, DwaMaxColorCorrections = 10;

// DwaBase.isph Model: how DwaBase treats 'metallic', 'transmission' and the like.
enum DwaModel : U32 {
    ModelMetal, ModelRefractive, ModelFabric, ModelVelvet, ModelVelvetV2, ModelSolidDielectric, ModelToon, ModelMixed,
    ModelHair, ModelHairDiffuse, ModelHairToon, ModelLayer
};
// DwaBaseParameterHints, one bit each; computed by the host as DwaBase::update() does.
enum DwaHint : U32 {
    HintToonDiffuse = 1u << 0, HintToonSpecular = 1u << 1, HintGlitter = 1u << 2, HintHair = 1u << 3,
    HintHairDiffuse = 1u << 4, HintHairToonS1 = 1u << 5, HintHairToonS2 = 1u << 6, HintHairToonS3 = 1u << 7,
    HintFuzz = 1u << 8, HintOuterSpecular = 1u << 9, HintOuterSpecularAbsorption = 1u << 10, HintSpecular = 1u << 11,
    HintMetallic = 1u << 12, HintAnisotropy = 1u << 13, HintIridescence = 1u << 14, HintTransmission = 1u << 15,
    HintTransmissionIor = 1u << 16, HintDiffuse = 1u << 17, HintSubsurface = 1u << 18, HintDiffuseTransmission = 1u << 19,
    HintFabric = 1u << 20, HintEmission = 1u << 21, HintGlitterIsOne = 1u << 22, HintHairIsOne = 1u << 23,
    HintHairDiffuseIsOne = 1u << 24, HintMetallicIsOne = 1u << 25, HintTransmissionIsOne = 1u << 26, HintFabricIsOne = 1u << 27,
    HintThinGeometry = 1u << 28, HintPreventLightCulling = 1u << 29, HintDisableOptimizedHairSampling = 1u << 30
};
// The material's boolean settings (GETBOOLATTR), one bit each.
enum DwaSetting : U32 {
    SetFuzzUseAbsorbingFibers = 1u << 0, SetUseOuterSpecularNormal = 1u << 1, SetUseIndependentTransmissionRoughness = 1u << 2,
    SetUseDispersion = 1u << 3, SetUseIndependentWeftAttributes = 1u << 4, SetUseUVsForThreadDirection = 1u << 5,
    SetIridescenceFlipHue = 1u << 6, SetSSSResolveSelfIntersections = 1u << 7, SetCastsCaustics = 1u << 8,
    SetHasNormalMap = 1u << 9, SetHasFuzzNormalMap = 1u << 10, SetHasOuterSpecularNormalMap = 1u << 11,
    SetToonExtendRamp = 1u << 12, SetHasNormalAAStrategy = 1u << 13, SetToonSpecUseInputVectors = 1u << 14,
    SetToonSpecEnableInputNormal = 1u << 15, SetToonSpecEnableIndirect = 1u << 16,
    SetHairShowR = 1u << 17, SetHairShowTT = 1u << 18, SetHairShowTRT = 1u << 19, SetHairShowGlint = 1u << 20,
    SetHairShowTRRT = 1u << 21, SetHairUseTTRoughness = 1u << 22, SetHairUseTRTRoughness = 1u << 23,
    SetHairUseOptimizedSampling = 1u << 24, SetHairDiffuseIndependentColors = 1u << 25,
    SetGlitterCompensateDeformation = 1u << 26, SetGlitterApproximateForSecondaryRays = 1u << 27
};
enum ToonDiffuseModel : U32 { ToonDiffuseOrenNayar = 0, ToonDiffuseRamp = 1 };
enum ToonSpecularModel : U32 { ToonSpecularBeckman = 0, ToonSpecularGGX, ToonSpecularSurface, ToonSpecularHair };
enum IridescenceLobeTarget : U32 { IridescencePrimarySpecular = 0, IridescenceOuterSpecular = 1 };
enum DiffuseTransmissionBlending : U32 { DiffuseTransmissionRgb = 0, DiffuseTransmissionMonochromatic = 1 };
enum NormalAAStrategy : U32 { NormalAANone = 0, NormalAAToksvig = 1 };
enum SubsurfaceType : U32 { SubsurfaceNormalizedDiffusion = 0, SubsurfaceDipoleDiffusion = 1, SubsurfaceRandomWalk = 2,
                            SubsurfaceNone = 3 };
// The DwaBase family's lobe labels (labels.json, as the build numbers them in labels.h; DECLARE_DWA_BASE_LABELS).
enum DwaLabels : U32 {
    DwaLabelNone = 0,
    DwaLabelGlitter = 1,
    DwaLabelHair = 2,
    DwaLabelHairR = 3,
    DwaLabelHairTT = 4,
    DwaLabelHairTRT = 5,
    DwaLabelHairTRRT = 6,
    DwaLabelHairDiffuse = 7,
    DwaLabelFuzz = 8,
    DwaLabelOuterSpecular = 9,
    DwaLabelSpecular = 10,
    DwaLabelDiffuse = 11,
    DwaLabelSpecularTransmission = 12,
    DwaLabelDiffuseTransmission = 13,
};

// HairToonMaterial's toon specular sets (specular_1..3): their ramps, input normal's register and settings, once; their
// inputs at the hit.
enum HairToonSetting : U32 { HairToonUseInputVectors = 1, HairToonEnableInputNormal = 2, HairToonEnableIndirect = 4 };
struct HairToonSpecUniforms {
    U32 rampCount;
    float rampPositions[DwaMaxToonRampPoints], rampValues[DwaMaxToonRampPoints];
    U32 rampInterpolators[DwaMaxToonRampPoints];
    U32 normalReg, settings;
};
struct HairToonSpecInputs {
    float intensity, fresnelBlend, roughness;
    Color tint;
    float stretchU, stretchV;
    V3 inputU, inputV;
    float rampInputScale;
    float rampMultipliers[DwaMaxToonRampPoints], rampOffsets[DwaMaxToonRampPoints];
    float inputNormalDial, indirectIntensity, indirectRoughness;
};
// The attribute values a DwaBase-family material shades from, in floats: its own value for each
// attribute it has, else the ISPC default resolveParameters uses. Bound maps multiply into these
// at the hit (MaterialBinding), as MoonRay's evalFloat/evalColor do.
struct DwaInputs {
    float refractiveIndex, specular, roughness, anisotropy, shadingTangent[2];
    float metallic;
    Color metallicColor, metallicEdgeColor;
    float iridescence;
    Color iridescencePrimary, iridescenceSecondary;
    float iridescenceThickness, iridescenceExponent, iridescenceAt0, iridescenceAt90;
    float transmission;
    Color transmissionColor;
    float independentTransmissionIor, independentTransmissionRoughness, dispersionAbbeNumber;
    float fuzz;
    Color fuzzAlbedo;
    float fuzzRoughness, fuzzNormalDial;
    float outerSpecular, outerSpecularIor, outerSpecularRoughness, outerSpecularThickness;
    Color outerSpecularAttenuationColor;
    float outerSpecularNormalDial;
    float fabricDiffuseScattering;
    Color warpColor;
    float warpRoughness;
    Color weftColor;
    float weftRoughness;
    V3 warpThreadDirection;
    float warpThreadCoverage, warpThreadElevation;
    Color albedo;
    float diffuseRoughness;
    Color scatteringColor;
    float scatteringRadius, creaseAttenuation, diffuseTransmission;
    Color diffuseTransmissionColor, emission;
    float presence, inputNormalDial, normalAADial;
    // toon diffuse
    float toonTerminatorShift, toonFlatness, toonFlatnessFalloff, toonRampInputScale;
    Color toonRampColorMultipliers[DwaMaxToonRampPoints];
    float toonRampPositionOffsets[DwaMaxToonRampPoints];
    // toon specular
    float toonSpecIntensity, toonSpecFresnelBlend;
    Color toonSpecTint;
    float toonSpecStretchU, toonSpecStretchV;
    V3 toonSpecInputU, toonSpecInputV;
    float toonSpecRampInputScale;
    float toonSpecRampMultipliers[DwaMaxToonRampPoints], toonSpecRampOffsets[DwaMaxToonRampPoints];
    float toonSpecInputNormalDial, toonSpecIndirectIntensity, toonSpecIndirectRoughness;
    // hair (HairMaterial_v3, HairDiffuseMaterial)
    Color hairColor;
    float hairCuticleLayerThickness, hairROffset, hairRRoughness;
    Color hairRTint;
    float hairTRTOffset, hairTRTRoughness;
    Color hairTRTTint;
    float hairGlintRoughness, hairGlintEccentricity, hairGlintMinTwists, hairGlintMaxTwists, hairGlintSaturation;
    float hairTTOffset, hairTTRoughness, hairTTAzimuthalRoughness;
    Color hairTTTint;
    float hairTTSaturation;
    Color hairDiffuseFrontColor, hairDiffuseBackColor;
    float hairSubsurfaceBlend;
    // HairToonMaterial
    float hairDiffuseWeight;
    HairToonSpecInputs hairToonSpec[3];
    // glitter
    float glitter, glitterStyleAFrequency;
    Color glitterColorA;
    float glitterSizeA, glitterRoughnessA, glitterStyleBFrequency;
    Color glitterColorB;
    float glitterSizeB, glitterRoughnessB;
    float glitterHueVariation, glitterSaturationVariation, glitterValueVariation, glitterDensity, glitterJitter;
    float glitterOrientationRandomness;
};
// DwaBaseUniformParameters: what createLobes reads from the material it shades (for a layer, the
// uniform parameters its update() blended).
struct DwaUniformParams {
    U32 specularModel, outerSpecularModel, outerSpecularUseBending, subsurface, thinGeometry, preventLightCulling;
    // diffuse_lightset and specular_lightset (indices into the plan's lobe light sets, ~0u none): the fuzz, diffuse,
    // subsurface and hair lobes take the first, the rest the second (DwaBaseLayerable). A layered material's are none.
    U32 diffuseLightSet = 0xffffffffu, specularLightSet = 0xffffffffu;
    U32 hairToonModel[3];  // HairToonMaterial's specular_1..3_model (ToonSpecularSurface or ToonSpecularHair)
};
// The settings a DwaBase-family material resolves once (update() and its uniform parameters).
struct DwaUniforms {
    U32 model, hints, settings;
    U32 normalAAStrategy, iridescenceApplyTo, iridescenceColorControl, iridescenceRampMode, diffuseTransmissionBlending;
    DwaUniformParams up;
    U32 toonDiffuseModel;
    U32 iridescenceRampCount;
    float iridescenceRampPositions[DwaMaxIridescenceRampPoints];
    Color iridescenceRampColors[DwaMaxIridescenceRampPoints];
    U32 iridescenceRampInterpolators[DwaMaxIridescenceRampPoints];
    // The map program registers of the normal maps (MapNoRegister for none): input_normal,
    // independent_clearcoat_normal, fuzz_normal.
    U32 normalReg, outerSpecularNormalReg, fuzzNormalReg;
    U32 toonRampCount;
    float toonRampPositions[DwaMaxToonRampPoints];
    Color toonRampColors[DwaMaxToonRampPoints];
    U32 toonRampInterpolators[DwaMaxToonRampPoints];
    U32 toonSpecRampCount;
    float toonSpecRampPositions[DwaMaxToonRampPoints], toonSpecRampValues[DwaMaxToonRampPoints];
    U32 toonSpecRampInterpolators[DwaMaxToonRampPoints];
    U32 toonSpecNormalReg;
    U32 hairFresnelType;
    HairToonSpecUniforms hairToonSpec[3];
    // The Glitter object's (DwaBase::updateGlitter): its tables in the pool (GlitterNone: none) and uniform parameters.
    U32 glitterTables;
    int glitterNoiseSeed;
    U32 glitterSpace, glitterLayeringMode, glitterDebugMode;
    float glitterFlakeRandomness;
    int glitterBlendStart, glitterBlendEnd;
};
// A DwaBase-family material's record in the material pool: uniforms then inputs.
struct DwaRecord {
    DwaUniforms u;
    DwaInputs in;
};
constexpr U32 DwaRecordFloats = sizeof(DwaRecord) / sizeof(float);
static_assert(sizeof(DwaRecord) % sizeof(float) == 0, "DwaRecord is a whole number of floats");

//----------------------------------------------------------------------------
// DwaBaseParameters (DwaBaseLayerable.isph), the fields XPU2 shades.
struct ToonDiffuseParameters {
    float toonDiffuse;
    U32 model;
    float terminatorShift, flatness, flatnessFalloff, rampWeight;
    U32 rampNumPoints;
    float rampPositions[DwaMaxToonRampPoints];
    Color rampColors[DwaMaxToonRampPoints];
    U32 rampInterpolators[DwaMaxToonRampPoints];
    V3 normal;
    bool extendRamp;
    float rampInputScale;
};
struct ToonSpecularParameters {
    float toonSpecular, intensity, fresnelBlend, roughness;
    Color tint;
    V3 normal;
    float rampInputScale;
    U32 rampNumPoints;
    float rampPositions[DwaMaxToonRampPoints], rampValues[DwaMaxToonRampPoints];
    U32 rampInterpolators[DwaMaxToonRampPoints];
    float stretchU, stretchV;
    V3 dPds, dPdt;
    bool enableIndirectReflections;
    float indirectReflectionsIntensity, indirectReflectionsRoughness;
    // the hair model's
    V3 hairDir;
    float hairUV[2], hairIOR;
    U32 hairFresnelType;
    float hairCuticleLayerThickness, hairRShift;
};
struct IridescenceParameters {
    float iridescence;
    U32 applyTo, colorControl;
    Color primaryColor, secondaryColor;
    bool flipHueDirection;
    float thickness, exponent, at0, at90;
    U32 rampInterpolationMode, rampNumPoints;
    float rampPositions[DwaMaxIridescenceRampPoints];
    Color rampColors[DwaMaxIridescenceRampPoints];
    U32 rampInterpolators[DwaMaxIridescenceRampPoints];
};
struct ColorCorrectParameters {
    bool on;
    float mix, hueShift, saturation, gain;
    bool tmiEnabled;
    Color tmi;
};
// HairParameters (DwaBaseLayerable.isph): HairMaterial_v3's and HairDiffuseMaterial's.
struct HairParameters {
    V3 dir;
    Color color;
    bool castsCaustics;
    float uv[2], ior;
    bool showR;
    float rShift, rLongRoughness;
    Color rTint;
    bool showTT;
    float ttShift, ttLongRoughness, ttAzimRoughness, ttSaturation;
    Color ttTint;
    bool showTRT;
    float trtShift, trtLongRoughness;
    Color trtTint;
    bool showGlint;
    float glintRoughness, glintMinTwists, glintMaxTwists, glintEccentricity, glintSaturation;
    bool showTRRT;
    float trrtLongRoughness;
    U32 fresnelType;
    float cuticleLayerThickness;
    bool useOptimizedSampling;
    Color diffuseFrontColor, diffuseBackColor;
    bool diffuseIndependent;
    float subsurfaceBlend;
};
struct DwaParams {
    float glitterMask;
    GlitterParameters glitter;
    float hair, hairDiffuse;
    HairParameters hairs;
    // fuzz
    float fuzz;
    Color fuzzAlbedo;
    float fuzzRoughness;
    bool fuzzUseAbsorbingFibers;
    V3 fuzzNormal;
    float fuzzNormalDial;
    // outer specular
    float outerSpecular, outerSpecularRefractiveIndex, outerSpecularRoughness, outerSpecularThickness;
    Color outerSpecularAttenuationColor;
    V3 outerSpecularNormal;
    float outerSpecularNormalLength, outerSpecularNormalDial;
    // metallic
    float metallic;
    Color metallicColor, metallicEdgeColor;
    // specular
    float specular, refractiveIndex, roughness, anisotropy, shadingTangent[2];
    // fabric
    float fabricSpecular;
    Color warpColor;
    float warpRoughness;
    bool useIndependentWeftAttributes;
    float weftRoughness;
    Color weftColor;
    V3 warpThreadDirection;
    float warpThreadCoverage, warpThreadElevation, fabricAttenuation;
    V3 fabricTangent;
    IridescenceParameters iridescence;
    // transmission
    float transmission;
    Color transmissionColor;
    bool useIndependentTransmissionRefractiveIndex;
    float independentTransmissionRefractiveIndex;
    bool useIndependentTransmissionRoughness;
    float independentTransmissionRoughness, dispersionAbbeNumber;
    ToonDiffuseParameters toonDiffuse;
    ToonSpecularParameters toonSpec;
    ToonSpecularParameters hairToonSpecs[3];
    // diffuse
    Color albedo;
    float diffuseRoughness;
    Color scatteringRadius;
    float creaseAttenuation;
    bool sssResolveSelfIntersections;
    Color diffuseTransmission;
    U32 diffuseTransmissionBlendingBehavior;
    // other
    V3 normal, diffuseNormal;
    float normalLength, normalDial;
    U32 normalAAStrategy;
    float normalAADial;
    Color emission;
    ColorCorrectParameters colorCorrect[DwaMaxColorCorrections];
    U32 numColorCorrections;
    // accent
    float accentSubsurface;
    Color accentSubsurfaceColor;
    float accentSubsurfaceNormalDial;
    V3 accentSubsurfaceNormal;
};

// DWABASELAYERABLE_initParameters (the colour corrections are the caller's).
X2_HD inline void initParameters(DwaParams& p) {
    p.glitterMask = 0.0f;
    initGlitterParameters(p.glitter);
    p.hair = p.hairDiffuse = 0.0f;
    // DWABASELAYERABLE_initHairParameters
    HairParameters& hp = p.hairs;
    hp.dir = V3{0, 0, 0};
    hp.color = rgb(1);
    hp.castsCaustics = false;
    hp.uv[0] = hp.uv[1] = 0.0f;
    hp.ior = 1.45f;
    hp.showR = true;
    hp.rShift = 0.0f;
    hp.rLongRoughness = 0.5f;
    hp.rTint = rgb(1);
    hp.showTT = true;
    hp.ttShift = 0.0f;
    hp.ttLongRoughness = 0.1f;
    hp.ttAzimRoughness = 1.0f;
    hp.ttSaturation = 1.0f;  // not initialised by MoonRay; set wherever it is read
    hp.ttTint = rgb(1);
    hp.showTRT = true;
    hp.trtShift = 0.0f;
    hp.trtLongRoughness = 0.4f;
    hp.trtTint = rgb(1);
    hp.showGlint = false;
    hp.glintRoughness = 0.5f;
    hp.glintMinTwists = 1.5f;
    hp.glintMaxTwists = 2.5f;
    hp.glintEccentricity = 0.85f;
    hp.glintSaturation = 0.5f;
    hp.showTRRT = true;
    hp.trrtLongRoughness = 4.0f * hp.rLongRoughness;
    hp.fresnelType = HairFresnelDielectricCylinder;
    hp.cuticleLayerThickness = 0.1f;
    hp.useOptimizedSampling = true;
    hp.diffuseIndependent = false;
    hp.diffuseFrontColor = rgb(1);
    hp.diffuseBackColor = rgb(1);
    hp.subsurfaceBlend = 1.0f;
    // DWABASELAYERABLE_initToonSpecularParameters
    for (U32 k = 0; k < 4; ++k) {
        ToonSpecularParameters& ts = k == 0 ? p.toonSpec : p.hairToonSpecs[k - 1];
        ts.toonSpecular = 0.0f;
        ts.intensity = 1.0f;
        ts.fresnelBlend = 1.0f;
        ts.roughness = 0.9f;
        ts.tint = rgb(1);
        ts.normal = V3{0, 0, 0};
        ts.rampInputScale = 1.0f;
        ts.rampNumPoints = 1;
        ts.rampPositions[0] = 0.0f;
        ts.rampValues[0] = 1.0f;
        ts.rampInterpolators[0] = RampNone;
        ts.stretchU = ts.stretchV = 1.0f;
        ts.dPds = ts.dPdt = V3{0, 0, 0};
        ts.enableIndirectReflections = false;
        ts.indirectReflectionsIntensity = 0.0f;
        ts.indirectReflectionsRoughness = 0.5f;
        ts.hairDir = V3{0, 0, 0};
        ts.hairUV[0] = ts.hairUV[1] = 0.0f;
        ts.hairIOR = 1.45f;
        ts.hairFresnelType = 1;  // HAIR_FRESNEL_DIELECTRIC_CYLINDER
        ts.hairCuticleLayerThickness = 0.1f;
        ts.hairRShift = 0.0f;
    }
    p.fuzz = 0.0f;
    p.fuzzRoughness = 0.0f;
    p.fuzzAlbedo = rgb(1);
    p.fuzzUseAbsorbingFibers = false;
    p.fuzzNormal = V3{0, 0, 0};
    p.fuzzNormalDial = 1.0f;  // not initialised by MoonRay; read only with a fuzz normal map
    p.outerSpecular = 0.0f;
    p.outerSpecularRefractiveIndex = 1.5f;
    p.outerSpecularRoughness = 0.0f;
    p.outerSpecularThickness = 0.0f;
    p.outerSpecularAttenuationColor = rgb(1);
    p.outerSpecularNormal = V3{0, 0, 0};
    p.outerSpecularNormalLength = 1.0f;
    p.outerSpecularNormalDial = 1.0f;
    p.metallic = 0.0f;
    p.metallicColor = rgb(1);
    p.metallicEdgeColor = rgb(1);
    p.specular = 0.0f;
    p.refractiveIndex = 1.5f;
    p.roughness = 0.0f;
    p.anisotropy = 0.0f;
    p.shadingTangent[0] = p.shadingTangent[1] = 0.0f;
    p.fabricTangent = V3{1, 0, 0};
    p.fabricSpecular = 0.0f;
    p.warpColor = rgb(0);
    p.warpRoughness = 0.0f;
    p.useIndependentWeftAttributes = false;
    p.weftRoughness = 0.0f;
    p.weftColor = rgb(0);
    p.warpThreadDirection = V3{0, 0, 0};
    p.warpThreadCoverage = 0.0f;
    p.warpThreadElevation = 0.0f;
    p.fabricAttenuation = 1.0f;
    IridescenceParameters& i = p.iridescence;
    i.iridescence = 0.0f;
    i.applyTo = IridescencePrimarySpecular;
    i.colorControl = 0;
    i.primaryColor = rgb(1);
    i.secondaryColor = rgb(1);
    i.flipHueDirection = false;
    i.thickness = 0.0f;
    i.exponent = 1.0f;
    i.at0 = 0.0f;
    i.at90 = 0.0f;
    i.rampInterpolationMode = RampRgb;
    i.rampNumPoints = 1;
    i.rampPositions[0] = 0.0f;
    i.rampColors[0] = rgb(1);
    i.rampInterpolators[0] = RampNone;
    p.transmission = 0.0f;
    p.transmissionColor = rgb(0);
    p.useIndependentTransmissionRefractiveIndex = false;
    p.independentTransmissionRefractiveIndex = 1.5f;
    p.useIndependentTransmissionRoughness = false;
    p.independentTransmissionRoughness = 0.0f;
    p.dispersionAbbeNumber = 0.0f;
    ToonDiffuseParameters& t = p.toonDiffuse;
    t.toonDiffuse = 0.0f;
    t.model = ToonDiffuseOrenNayar;
    t.terminatorShift = t.flatness = t.flatnessFalloff = 0.0f;
    t.rampWeight = 0.0f;
    t.rampNumPoints = 1;
    t.rampPositions[0] = 0.0f;
    t.rampColors[0] = rgb(1);
    t.rampInterpolators[0] = RampNone;
    t.normal = V3{0, 0, 0};
    t.extendRamp = false;
    t.rampInputScale = 1.0f;
    p.albedo = rgb(0);
    p.diffuseRoughness = 0.0f;
    p.scatteringRadius = rgb(0);
    p.creaseAttenuation = 0.0f;
    p.diffuseTransmission = rgb(0);
    p.diffuseTransmissionBlendingBehavior = DiffuseTransmissionMonochromatic;
    p.sssResolveSelfIntersections = true;
    p.normal = V3{0, 0, 0};
    p.diffuseNormal = V3{0, 0, 0};
    p.normalDial = 1.0f;
    p.normalLength = 1.0f;
    p.normalAAStrategy = NormalAANone;
    p.normalAADial = 1.0f;
    p.emission = rgb(0);
    p.accentSubsurface = 1.0f;
    p.accentSubsurfaceNormal = V3{0, 0, 0};
    p.accentSubsurfaceNormalDial = 1.0f;
    p.accentSubsurfaceColor = rgb(1);
}
// DWABASELAYERABLE_initColorCorrectParameters
X2_HD inline void initColorCorrectParameters(DwaParams& p) {
    p.numColorCorrections = 0;
    for (U32 i = 0; i < DwaMaxColorCorrections; ++i) {
        ColorCorrectParameters& c = p.colorCorrect[i];
        c.on = false;
        c.mix = 1.0f;
        c.hueShift = 0.0f;
        c.saturation = 1.0f;
        c.gain = 1.0f;
        c.tmiEnabled = false;
        c.tmi = rgb(0);
    }
}

//----------------------------------------------------------------------------
// Colour corrections (DWABASE_applyColorCorrectParameters).
X2_HD inline void applyColorCorrections(const ColorCorrectParameters& cc, Color& c) {
    if (!isZero(cc.hueShift)) applyHueShift(cc.hueShift, c);
    if (!isOne(cc.saturation)) applySaturation(cc.saturation, c.x, c.y, c.z);
    c = c * cc.gain;
    if (cc.tmiEnabled) applyTMI(cc.tmi.x, cc.tmi.y, cc.tmi.z, c.x, c.y, c.z);
}
X2_HD inline void correctClamped(const ColorCorrectParameters& cc, Color& c) {
    const Color original = c;
    applyColorCorrections(cc, c);
    c = lerpOpt(original, c, cc.mix);
    c = Color{fminf(fmaxf(c.x, 0.f), 1.f), fminf(fmaxf(c.y, 0.f), 1.f), fminf(fmaxf(c.z, 0.f), 1.f)};
}
X2_CALL void applyColorCorrectParameters(DwaParams& p) {
    Color* colors[] = {&p.fuzzAlbedo, &p.metallicColor, &p.metallicEdgeColor, &p.warpColor, &p.weftColor, &p.transmissionColor,
                       &p.albedo, &p.diffuseTransmission, &p.toonSpec.tint, &p.accentSubsurfaceColor, &p.scatteringRadius,
                       &p.iridescence.primaryColor, &p.iridescence.secondaryColor, &p.hairs.diffuseFrontColor,
                       &p.hairs.diffuseBackColor};
    for (U32 k = 0; k < p.numColorCorrections; ++k) {
        const ColorCorrectParameters& cc = p.colorCorrect[k];
        if (!cc.on || isZero(cc.mix)) continue;
        for (U32 i = 0; i < sizeof(colors) / sizeof(colors[0]); ++i) correctClamped(cc, *colors[i]);
        if (!isZero(p.toonDiffuse.toonDiffuse) && p.toonDiffuse.model == ToonDiffuseRamp)
            for (U32 i = 0; i < p.toonDiffuse.rampNumPoints; ++i) correctClamped(cc, p.toonDiffuse.rampColors[i]);
        if (!isZero(p.iridescence.iridescence) && p.iridescence.colorControl == 1)
            for (U32 i = 0; i < p.iridescence.rampNumPoints; ++i) correctClamped(cc, p.iridescence.rampColors[i]);
        // Emission is not clamped at one.
        const Color original = p.emission;
        Color c = p.emission;
        applyColorCorrections(cc, c);
        c = lerpOpt(original, c, cc.mix);
        p.emission = Color{fmaxf(c.x, 0.f), fmaxf(c.y, 0.f), fmaxf(c.z, 0.f)};
    }
}

//----------------------------------------------------------------------------
// DWABASE_resolveParameters, from the record's values at this hit.
// castsCaustics is what the caller passes: the material's own for a material shaded alone.
// EvalAttribute.isph evalNormal with a NormalMap: the map's render-space normal value, the dial
// toward the state's normal, and adaptNormal (adaptToonNormal for a toon normal). unadapted gets
// the normal before adapting; lengthN the value's length.
X2_HD inline V3 evalNormalMap(const State& st, bool present, V3 value, float dial, float* lengthN, V3* unadapted, bool toon = false) {
    if (!present || isZero(dial)) {
        if (unadapted) *unadapted = st.N;
        return st.N;
    }
    if (lengthN) *lengthN = length(value);
    if (!finite3(value) || (isZero(value.x) && isZero(value.y) && isZero(value.z))) {
        if (unadapted) *unadapted = st.N;
        return st.N;
    }
    V3 result = normalize(value);
    if (dial != 1.0f) result = normalize(st.N + (result - st.N) * saturate(dial));
    if (unadapted) *unadapted = result;
    return toon ? adaptToonNormal(st, result) : adaptNormal(st, result);
}
X2_HD inline V3 regValue(const Color* regs, U32 reg) { return reg == MapNoRegister ? V3{0, 0, 0} : regs[reg]; }

X2_CALL void resolveDwaBase(const DwaUniforms& u, const DwaInputs& in, const State& st, bool castsCaustics, DwaParams& p,
                            const Color* regs) {
    initParameters(p);
    const U32 h = u.hints;
    const bool caustics = castsCaustics || !st.causticPath;
    // resolveNormalParams
    if (u.normalReg != MapNoRegister) {
        p.normalDial = in.inputNormalDial;
        if (u.settings & SetHasNormalAAStrategy) {
            p.normalAAStrategy = u.normalAAStrategy;
            p.normalAADial = in.normalAADial;
        }
        V3 unadapted;
        p.normal = evalNormalMap(st, true, regs[u.normalReg], p.normalDial,
                                 p.normalAAStrategy == NormalAAToksvig ? &p.normalLength : nullptr, &unadapted);
        p.diffuseNormal = unadapted;
    } else {
        p.normal = st.N;
        p.diffuseNormal = st.N;
    }
    p.refractiveIndex = fmaxf(Epsilon, in.refractiveIndex);

    // DWABASE_resolveGlitterParams
    if (h & HintGlitter) {
        GlitterParameters& g = p.glitter;
        g.tables = u.glitterTables;
        g.noiseSeed = u.glitterNoiseSeed;
        g.space = u.glitterSpace;
        g.layeringMode = u.glitterLayeringMode;
        g.debugMode = u.glitterDebugMode;
        g.flakeRandomness = u.glitterFlakeRandomness;
        g.blendStart = u.glitterBlendStart;
        g.blendEnd = u.glitterBlendEnd;
        p.glitterMask = (h & HintGlitterIsOne) ? 1.0f : saturate(in.glitter);
        g.styleFrequency[0] = in.glitterStyleAFrequency;
        g.styleFrequency[1] = in.glitterStyleBFrequency;
        // A style that is off has no size, so the search's largest size is right.
        if (!isZero(g.styleFrequency[0])) {
            g.flakeColor[0] = in.glitterColorA;
            g.flakeSize[0] = in.glitterSizeA;
            g.flakeRoughness[0] = in.glitterRoughnessA;
        } else {
            g.flakeColor[0] = rgb(0);
            g.flakeSize[0] = 0.f;
            g.flakeRoughness[0] = 0.14f;
        }
        if (!isZero(g.styleFrequency[1])) {
            g.flakeColor[1] = in.glitterColorB;
            g.flakeSize[1] = in.glitterSizeB;
            g.flakeRoughness[1] = in.glitterRoughnessB;
        } else {
            g.flakeColor[1] = rgb(0);
            g.flakeSize[1] = 0.f;
            g.flakeRoughness[1] = 0.14f;
        }
        g.hsvVariation = V3{in.glitterHueVariation, in.glitterSaturationVariation, in.glitterValueVariation};
        g.density = in.glitterDensity;
        g.jitter = in.glitterJitter;
        g.orientationRandomness = in.glitterOrientationRandomness;
        g.compensateDeformation = (u.settings & SetGlitterCompensateDeformation) ? 1 : 0;
        g.approximateForSecRays = (u.settings & SetGlitterApproximateForSecondaryRays) ? 1 : 0;
    }
    // resolveSpecularParams
    if ((h & HintSpecular) && caustics) {
        p.specular = saturate(in.specular);
        p.roughness = saturate(in.roughness);
        if (h & HintAnisotropy) {
            p.anisotropy = clampf(in.anisotropy, -1.0f, 1.0f);
            if (!isZero(p.anisotropy)) {
                p.shadingTangent[0] = in.shadingTangent[0];
                p.shadingTangent[1] = in.shadingTangent[1];
            }
        }
    }
    // resolveIridescenceParams
    if ((h & HintIridescence) && caustics) {
        IridescenceParameters& i = p.iridescence;
        i.iridescence = saturate(in.iridescence);
        i.applyTo = u.iridescenceApplyTo;
        i.colorControl = u.iridescenceColorControl;
        i.primaryColor = clamp(in.iridescencePrimary, 0.0f, 1.0f);
        i.secondaryColor = clamp(in.iridescenceSecondary, 0.0f, 1.0f);
        i.flipHueDirection = (u.settings & SetIridescenceFlipHue) != 0;
        i.thickness = fmaxf(0.0f, in.iridescenceThickness);
        i.exponent = fmaxf(0.0f, in.iridescenceExponent);
        i.at0 = saturate(in.iridescenceAt0);
        i.at90 = saturate(in.iridescenceAt90);
        i.rampInterpolationMode = u.iridescenceRampMode;
        i.rampNumPoints = u.iridescenceRampCount < DwaMaxIridescenceRampPoints ? u.iridescenceRampCount : DwaMaxIridescenceRampPoints;
        for (U32 k = 0; k < i.rampNumPoints; ++k) {
            i.rampInterpolators[k] = u.iridescenceRampInterpolators[k];
            i.rampPositions[k] = u.iridescenceRampPositions[k];
            i.rampColors[k] = u.iridescenceRampColors[k];
        }
    }
    if ((h & HintMetallic) && caustics) {
        p.metallic = (u.model == ModelMetal || (h & HintMetallicIsOne)) ? 1.0f : saturate(in.metallic);
        p.metallicColor = clamp(in.metallicColor, 0.0f, 1.0f);
        p.metallicEdgeColor = clamp(in.metallicEdgeColor, 0.0f, 1.0f);
    }

    // resolveHairParams
    {
        HairParameters& hp = p.hairs;
        if (castsCaustics || !st.causticPath) hp.castsCaustics = true;
        if ((h & HintHair) || (h & HintHairDiffuse)) hp.dir = normalize(st.dPds);
        if ((h & HintHair) && !(h & HintHairDiffuseIsOne)) hp.color = clamp(in.hairColor, 0.0f, 1.0f);
        if (h & HintHair) {
            p.hair = 1.0f;  // HairMaterial_v3 (a hair weight of a layer is refused at admission)
            if (hp.castsCaustics) {
                hp.ttAzimRoughness = clampf(in.hairTTAzimuthalRoughness, 0.01f, 1.0f - Epsilon);
                hp.uv[0] = st.s;
                hp.uv[1] = st.t;
                hp.ior = in.refractiveIndex;
                hp.showR = (u.settings & SetHairShowR) != 0;
                hp.rLongRoughness = in.hairRRoughness;
                if (hp.showR) {
                    hp.rShift = deg2rad(clampf(in.hairROffset, -10.0f, 10.0f));
                    hp.rTint = in.hairRTint;
                }
                hp.showTT = (u.settings & SetHairShowTT) != 0;
                hp.ttLongRoughness = 0.5f * hp.rLongRoughness;
                if (hp.showTT) {
                    if (u.settings & SetHairUseTTRoughness) hp.ttLongRoughness = in.hairTTRoughness;
                    hp.ttShift = deg2rad(clampf(in.hairTTOffset, -10.0f, 10.0f));
                    hp.ttTint = in.hairTTTint;
                    // MOONSHINE-1238: only for HiFi hits
                    const bool hifi = st.pathType == PathPrimary || st.pathType == PathIndirectMirror;
                    hp.ttSaturation = hifi ? in.hairTTSaturation : 1.0f;
                }
                hp.showTRT = (u.settings & SetHairShowTRT) != 0;
                hp.trtLongRoughness = 2.0f * hp.rLongRoughness;
                if (hp.showTRT) {
                    if (u.settings & SetHairUseTRTRoughness) hp.trtLongRoughness = in.hairTRTRoughness;
                    hp.trtShift = deg2rad(clampf(in.hairTRTOffset, -10.0f, 10.0f));
                    hp.trtTint = in.hairTRTTint;
                    hp.showGlint = (u.settings & SetHairShowGlint) != 0;
                    if (hp.showGlint) {
                        hp.glintRoughness = in.hairGlintRoughness;
                        hp.glintMinTwists = in.hairGlintMinTwists;
                        hp.glintMaxTwists = in.hairGlintMaxTwists;
                        hp.glintEccentricity = in.hairGlintEccentricity;
                        hp.glintSaturation = in.hairGlintSaturation;
                    }
                }
                hp.showTRRT = (u.settings & SetHairShowTRRT) != 0;
                hp.trrtLongRoughness = 4.0f * hp.rLongRoughness;
                hp.fresnelType = u.hairFresnelType;
                hp.cuticleLayerThickness = clampf(in.hairCuticleLayerThickness, 0.0f, 1.0f);
                hp.useOptimizedSampling = (h & HintDisableOptimizedHairSampling) ? false : (u.settings & SetHairUseOptimizedSampling) != 0;
            }
        }
        if (h & HintHairDiffuse) {
            // HairDiffuseMaterial; HairToonMaterial's hair_diffuse
            p.hairDiffuse = (h & HintHairDiffuseIsOne) ? 1.0f : saturate(in.hairDiffuseWeight);
            hp.diffuseIndependent = (u.settings & SetHairDiffuseIndependentColors) != 0;
            hp.subsurfaceBlend = saturate(in.hairSubsurfaceBlend);
            if (hp.diffuseIndependent) {
                hp.diffuseFrontColor = in.hairDiffuseFrontColor;
                hp.diffuseBackColor = in.hairDiffuseBackColor;
            } else {
                hp.diffuseFrontColor = in.hairColor;
                hp.diffuseBackColor = hp.diffuseFrontColor;
            }
        }
    }

    // resolveToonParams (the toon diffuse part; toon specular is refused at admission)
    if (h & HintToonDiffuse) {
        ToonDiffuseParameters& t = p.toonDiffuse;
        t.toonDiffuse = 1.0f;
        t.model = u.toonDiffuseModel;
        if (t.model == ToonDiffuseOrenNayar) {
            t.terminatorShift = clampf(in.toonTerminatorShift, -1.0f, 1.0f);
            t.flatness = clampf(in.toonFlatness, 0.0f, 1.0f);
            t.flatnessFalloff = clampf(in.toonFlatnessFalloff, 0.0f, 1.0f);
        }
        // evalToonNormal: unadapted (the toon diffuse lobes take it as it is).
        V3 unadapted;
        evalNormalMap(st, u.normalReg != MapNoRegister, regValue(regs, u.normalReg), p.normalDial, nullptr, &unadapted, true);
        t.normal = unadapted;
        if (t.model == ToonDiffuseRamp) {
            t.rampWeight = 1.0f;
            t.rampNumPoints = u.toonRampCount < DwaMaxToonRampPoints ? u.toonRampCount : DwaMaxToonRampPoints;
            for (U32 k = 0; k < t.rampNumPoints; ++k) {
                t.rampInterpolators[k] = u.toonRampInterpolators[k];
                t.rampPositions[k] = u.toonRampPositions[k] + in.toonRampPositionOffsets[k];
                t.rampColors[k] = in.toonRampColorMultipliers[k] * u.toonRampColors[k];
            }
            t.extendRamp = (u.settings & SetToonExtendRamp) != 0;
            t.rampInputScale = in.toonRampInputScale;
        }
    }

    // resolveToonSpecParams (the surface toon specular; a toon specular follows the specular's weight)
    if (h & HintToonSpecular) {
        ToonSpecularParameters& t = p.toonSpec;
        t.toonSpecular = p.specular;
        t.intensity = in.toonSpecIntensity;
        t.fresnelBlend = in.toonSpecFresnelBlend;
        t.roughness = in.roughness;
        t.tint = clamp(in.toonSpecTint, 0.0f, 1.0f);
        t.rampInputScale = in.toonSpecRampInputScale;
        t.rampNumPoints = u.toonSpecRampCount < DwaMaxToonRampPoints ? u.toonSpecRampCount : DwaMaxToonRampPoints;
        for (U32 k = 0; k < t.rampNumPoints; ++k) {
            t.rampPositions[k] = u.toonSpecRampPositions[k] + in.toonSpecRampOffsets[k];
            t.rampValues[k] = u.toonSpecRampValues[k] * in.toonSpecRampMultipliers[k];
            t.rampInterpolators[k] = u.toonSpecRampInterpolators[k];
        }
        if (u.settings & SetToonSpecEnableInputNormal)
            t.normal = evalNormalMap(st, u.toonSpecNormalReg != MapNoRegister, regValue(regs, u.toonSpecNormalReg),
                                     in.toonSpecInputNormalDial, nullptr, nullptr);
        else
            t.normal = st.N;
        t.stretchU = in.toonSpecStretchU;
        t.stretchV = in.toonSpecStretchV;
        if (u.settings & SetToonSpecUseInputVectors) {  // render-space vectors
            t.dPds = normalize(renderToWorldVector(st.worldToRender, in.toonSpecInputU));
            t.dPdt = normalize(renderToWorldVector(st.worldToRender, in.toonSpecInputV));
        } else {
            t.dPds = normalize(st.dPds);
            t.dPdt = normalize(st.dPdt);
        }
        t.enableIndirectReflections = (u.settings & SetToonSpecEnableIndirect) != 0;
        if (t.enableIndirectReflections) {
            t.indirectReflectionsIntensity = in.toonSpecIndirectIntensity;
            t.indirectReflectionsRoughness = in.toonSpecIndirectRoughness;
        }
    }

    // resolveHairToonParams: each shown specular set, at weight one
    for (U32 n = 0; n < 3; ++n) {
        if (!(h & (HintHairToonS1 << n))) continue;
        ToonSpecularParameters& t = p.hairToonSpecs[n];
        const HairToonSpecInputs& i = in.hairToonSpec[n];
        const HairToonSpecUniforms& v = u.hairToonSpec[n];
        t.toonSpecular = 1.0f;
        t.intensity = i.intensity;
        t.fresnelBlend = i.fresnelBlend;
        t.roughness = i.roughness;
        t.tint = clamp(i.tint, 0.0f, 1.0f);
        t.rampInputScale = i.rampInputScale;
        t.rampNumPoints = v.rampCount < DwaMaxToonRampPoints ? v.rampCount : DwaMaxToonRampPoints;
        for (U32 k = 0; k < t.rampNumPoints; ++k) {
            t.rampPositions[k] = v.rampPositions[k] + i.rampOffsets[k];
            t.rampValues[k] = v.rampValues[k] * i.rampMultipliers[k];
            t.rampInterpolators[k] = v.rampInterpolators[k];
        }
        if (v.settings & HairToonEnableInputNormal)
            t.normal = evalNormalMap(st, v.normalReg != MapNoRegister, regValue(regs, v.normalReg), i.inputNormalDial, nullptr, nullptr);
        else
            t.normal = st.N;
        t.stretchU = i.stretchU;
        t.stretchV = i.stretchV;
        if (v.settings & HairToonUseInputVectors) {  // render-space vectors
            t.dPds = normalize(renderToWorldVector(st.worldToRender, i.inputU));
            t.dPdt = normalize(renderToWorldVector(st.worldToRender, i.inputV));
        } else {
            t.dPds = normalize(st.dPds);
            t.dPdt = normalize(st.dPdt);
        }
        t.enableIndirectReflections = (v.settings & HairToonEnableIndirect) != 0;
        if (t.enableIndirectReflections) {
            t.indirectReflectionsIntensity = i.indirectIntensity;
            t.indirectReflectionsRoughness = i.indirectRoughness;
        }
        if (u.up.hairToonModel[n] == ToonSpecularHair) {
            t.hairDir = normalize(st.dPds);
            t.hairUV[0] = st.s;
            t.hairUV[1] = st.t;
        }
    }

    // resolveFuzzParams
    if ((h & HintFuzz) && caustics) {
        p.fuzz = saturate(in.fuzz);
        p.fuzzAlbedo = clamp(in.fuzzAlbedo, 0.0f, 1.0f);
        p.fuzzRoughness = saturate(in.fuzzRoughness);
        if (u.fuzzNormalReg != MapNoRegister) {
            p.fuzzNormalDial = in.fuzzNormalDial;
            p.fuzzNormal = evalNormalMap(st, true, regs[u.fuzzNormalReg], p.fuzzNormalDial, nullptr, nullptr);
        } else {
            p.fuzzNormal = st.N;
        }
        p.fuzzUseAbsorbingFibers = (u.settings & SetFuzzUseAbsorbingFibers) != 0;
    }

    // resolveOuterSpecularParams
    p.outerSpecularNormal = st.N;
    if ((h & HintOuterSpecular) && caustics) {
        p.outerSpecular = saturate(in.outerSpecular);
        p.outerSpecularRefractiveIndex = fmaxf(Epsilon, in.outerSpecularIor);
        p.outerSpecularRoughness = saturate(in.outerSpecularRoughness);
        if ((u.settings & SetUseOuterSpecularNormal) && u.outerSpecularNormalReg != MapNoRegister) {
            p.outerSpecularNormalDial = in.outerSpecularNormalDial;
            p.outerSpecularNormal = evalNormalMap(st, true, regs[u.outerSpecularNormalReg], p.outerSpecularNormalDial,
                                                  &p.outerSpecularNormalLength, nullptr);
        }
        if (h & HintOuterSpecularAbsorption) {
            p.outerSpecularThickness = fmaxf(0.0f, in.outerSpecularThickness);
            p.outerSpecularAttenuationColor = clamp(in.outerSpecularAttenuationColor, 0.0f, 1.0f);
        }
    }

    // resolveFabricSpecularParams
    if (!(h & HintMetallicIsOne) && !(h & HintTransmissionIsOne) && (h & HintFabric)) {
        p.fabricSpecular = u.model == ModelFabric ? 1.0f : 0.0f;
        const float fabricDiffuseScattering = saturate(in.fabricDiffuseScattering);
        p.fabricSpecular *= (1.0f - fabricDiffuseScattering);
        if (!caustics) {
            p.fabricAttenuation = 1.0f - p.fabricSpecular;
        } else {
            p.warpColor = clamp(in.warpColor, 0.0f, 1.0f);
            p.warpRoughness = saturate(in.warpRoughness);
            p.warpThreadElevation = clampf(in.warpThreadElevation, -89.0f, 89.0f);
            p.useIndependentWeftAttributes = (u.settings & SetUseIndependentWeftAttributes) != 0;
            if (p.useIndependentWeftAttributes) {
                p.weftColor = clamp(in.weftColor, 0.0f, 1.0f);
                p.weftRoughness = saturate(in.weftRoughness);
            } else {
                p.weftColor = p.warpColor;
                p.weftRoughness = p.warpRoughness;
            }
            // Without UVs: render space's X (the camera's, in world space here).
            p.fabricTangent = (u.settings & SetUseUVsForThreadDirection)
                                  ? normalize(st.dPds)
                                  : (st.worldToRender ? normalize(V3{st.worldToRender[0], st.worldToRender[1], st.worldToRender[2]})
                                                      : V3{1, 0, 0});
            p.warpThreadDirection = normalize(in.warpThreadDirection);
            p.warpThreadCoverage = saturate(in.warpThreadCoverage);
        }
    }

    // resolveTransmissionParams
    if (!(h & HintMetallicIsOne) && (h & HintTransmission) && caustics) {
        if (!(h & HintSpecular)) p.roughness = saturate(in.roughness);
        p.transmission = u.model == ModelRefractive ? 1.0f : saturate(in.transmission);
        p.transmissionColor = clamp(in.transmissionColor, 0.0f, 1.0f);
        if (h & HintTransmissionIor) {
            p.useIndependentTransmissionRefractiveIndex = true;
            p.independentTransmissionRefractiveIndex = fmaxf(Epsilon, in.independentTransmissionIor);
        }
        p.useIndependentTransmissionRoughness = (u.settings & SetUseIndependentTransmissionRoughness) != 0;
        p.independentTransmissionRoughness =
            p.useIndependentTransmissionRoughness ? saturate(in.independentTransmissionRoughness) : p.roughness;
        p.dispersionAbbeNumber = (u.settings & SetUseDispersion) ? fmaxf(0.0f, in.dispersionAbbeNumber) : 0.0f;
    }

    if (h & HintDiffuse) p.albedo = clamp(in.albedo, 0.0f, 1.0f);
    if ((h & HintDiffuse) || (h & HintToonDiffuse)) p.diffuseRoughness = saturate(in.diffuseRoughness);
    if (((h & HintSubsurface) || (h & HintHairDiffuse)) && st.subsurfaceAllowed) {
        p.scatteringRadius = clamp(in.scatteringColor, 0.0f, 1.0f) * fmaxf(0.0f, in.scatteringRadius);
        p.creaseAttenuation = fmaxf(0.0f, in.creaseAttenuation);
        p.sssResolveSelfIntersections = (u.settings & SetSSSResolveSelfIntersections) != 0;
    }
    if (h & HintDiffuseTransmission) {
        const float dt = clampf(in.diffuseTransmission, 0.0f, 1.0f);
        p.diffuseTransmission = dt * clamp(in.diffuseTransmissionColor, 0.0f, 1.0f);
        p.diffuseTransmissionBlendingBehavior = u.diffuseTransmissionBlending;
    }
    if (h & HintEmission) p.emission = Color{fmaxf(0.f, in.emission.x), fmaxf(0.f, in.emission.y), fmaxf(0.f, in.emission.z)};

    // resolveAccentParams
    p.accentSubsurface = 1.0f;
    p.accentSubsurfaceNormal = p.diffuseNormal;
    p.accentSubsurfaceNormalDial = p.normalDial;
    p.accentSubsurfaceColor = p.albedo;

    applyColorCorrectParameters(p);
}
// DWABASE_resolvePresence
X2_HD inline float dwaPresence(const DwaInputs& in) { return saturate(in.presence); }

//----------------------------------------------------------------------------
// DWABASE_createLobes
X2_HD inline float computeMicrofacetRoughness(float roughness, U32 strategy, float normalLength, float normalAADial, float normalDial) {
    if (strategy != NormalAAToksvig) return roughness;
    if (isOne(normalLength) || isZero(normalAADial) || isZero(normalDial)) return roughness;
    normalLength = lerpf(1.f, normalLength, normalDial);
    const float varN = clampf(normalLength, 0.01f, 1.0f);
    const float toksvigFactor = fmaxf((1.0f - varN) / varN, 0.f);
    const float roughnessAA = sqrtf(toksvigFactor + roughness * roughness);
    return lerpf(roughness, roughnessAA, normalAADial);
}
X2_HD inline void computeAnisoRoughness(float roughness, float minRoughness, float anisotropy, float& x, float& y) {
    x = anisotropy > 0.0f ? roughness * (1.0f - anisotropy) : roughness;
    y = anisotropy < 0.0f ? roughness * (1.0f - fabsf(anisotropy)) : roughness;
    x = fmaxf(x, minRoughness);
    y = fmaxf(y, minRoughness);
}
X2_HD inline bool isWhite(Color c) { return isOne(c.x) && isOne(c.y) && isOne(c.z); }
X2_HD inline Color diffuseTransmissionAttenuation(const DwaParams& p) {
    if (p.diffuseTransmissionBlendingBehavior == DiffuseTransmissionRgb) return rgb(1) - p.diffuseTransmission;
    return rgb(1) - rgb(fmaxf(fmaxf(p.diffuseTransmission.x, p.diffuseTransmission.y), p.diffuseTransmission.z));
}

X2_CALL void addOuterSpecularLobes(Builder& bb, const DwaParams& p, const DwaUniformParams& u, float minRoughness,
                                   const Iridescence* iri) {
    const bool refracts = u.outerSpecularUseBending && !u.thinGeometry;
    const Color attenuationColor = p.outerSpecularAttenuationColor;
    const bool absorbs = !isZero(p.outerSpecularThickness) && !isWhite(attenuationColor);
    float roughness = computeMicrofacetRoughness(p.outerSpecularRoughness, p.normalAAStrategy, p.outerSpecularNormalLength,
                                                 p.normalAADial, p.outerSpecularNormalDial);
    roughness = fmaxf(roughness, minRoughness);
    if (refracts || absorbs) {
        if (isZero(roughness)) {
            MirrorClearcoat c{p.outerSpecularNormal, p.outerSpecularRefractiveIndex, p.outerSpecularThickness, attenuationColor,
                              refracts, iri};
            addMirrorClearcoat(bb, c, p.outerSpecular, Physical, DwaLabelOuterSpecular);
        } else {
            MicrofacetIsotropicClearcoat c{p.outerSpecularNormal, p.outerSpecularRefractiveIndex, roughness, p.outerSpecularThickness,
                                           attenuationColor, refracts, u.outerSpecularModel, iri};
            addMicrofacetIsotropicClearcoat(bb, c, p.outerSpecular, Physical, DwaLabelOuterSpecular);
        }
    } else {
        if (isZero(roughness)) {
            addMirrorBRDF(bb, mirrorDielectric(p.outerSpecularNormal, p.outerSpecularRefractiveIndex, iri), p.outerSpecular, Physical,
                          DwaLabelOuterSpecular);
        } else {
            addMicrofacetIsotropicBRDF(bb, microfacetDielectric(p.outerSpecularNormal, p.outerSpecularRefractiveIndex, roughness,
                                                                u.outerSpecularModel, iri),
                                       p.outerSpecular, Physical, DwaLabelOuterSpecular);
        }
    }
}

// DWABASE_addHairLobes: hair diffuse (and its subsurface share), then the hair lobes (one-sampler or separate),
// or hair diffuse in their place on a caustic path the material does not cast.
X2_CALL void addHairLobes(const DwaParams& p, const DwaUniformParams& u, const State& st, Builder& bb) {
    const HairParameters& hp = p.hairs;
    if (!isZero(p.hairDiffuse)) {
        setThinGeo(bb);  // hair curves are always thin
        const bool hasSSS = !isBlack(p.scatteringRadius);
        const float sssWeight = hasSSS ? hp.subsurfaceBlend : 0.0f;
        addHairDiffuseBSDF(bb, HairDiffuseBSDF{hp.dir, hp.diffuseFrontColor, hp.diffuseBackColor}, p.hairDiffuse * (1.0f - sssWeight),
                           Physical, DwaLabelHairDiffuse);
        if (hasSSS) {
            const DiffusionProfile d{st.N, hp.diffuseFrontColor, p.scatteringRadius};
            if (u.subsurface == SubsurfaceNormalizedDiffusion)
                addDiffusion(bb, BssrdfNormalizedDiffusion, d, p.hairDiffuse * sssWeight, Physical, DwaLabelDiffuse);
            else if (u.subsurface == SubsurfaceDipoleDiffusion)
                addDiffusion(bb, BssrdfDipole, d, p.hairDiffuse * sssWeight, Physical, DwaLabelDiffuse);
        }
    }
    if (isZero(p.hair)) return;
    setThinGeo(bb);
    if (!hp.castsCaustics) {
        addHairDiffuseBSDF(bb, HairDiffuseBSDF{hp.dir, hp.color, hp.color}, 1.0f, Physical, DwaLabelHair);
        return;
    }
    V3 N{0, 0, 0};
    float hairRotation = 0.0f, glintRoughness = 0.0f;
    if (hp.showGlint) {
        if (!(isZero(hp.glintMinTwists) && isZero(hp.glintMaxTwists))) {
            const V3 T = normalize(st.dPds), B = normalize(st.dPdt);
            N = cross(B, T);
            // createHairRandomPerStrand: scatter_tag, a random number a strand.
            float glintRandom = 0.0f;
            if (!isEqual(hp.glintMinTwists, hp.glintMaxTwists)) {
                float tag[3];
                if (primValue(st, primKey("scatter_tag", PrimFloat), tag) && !isEqual(tag[0], -1.f)) glintRandom = tag[0];
            }
            hairRotation = TwoPi * lerpf(hp.glintMinTwists, hp.glintMaxTwists, glintRandom);
        }
        glintRoughness = 0.125f * hp.glintRoughness * hp.glintRoughness;  // an empirical remapping
    }
    const float eccentricity = clampf(hp.glintEccentricity, 0.85f, 1.0f);
    if (hp.useOptimizedSampling) {
        HairBSDF b{hp.dir, {hp.uv[0], hp.uv[1]}, hp.ior, hp.fresnelType, hp.cuticleLayerThickness, hp.showR, hp.rShift,
                   hp.rLongRoughness, hp.rTint, hp.showTT, hp.ttShift, hp.ttLongRoughness, hp.ttAzimRoughness, hp.ttTint,
                   hp.ttSaturation, hp.showTRT, hp.trtShift, hp.trtLongRoughness, hp.trtTint, hp.showGlint, glintRoughness,
                   eccentricity, hp.glintSaturation, hairRotation, N, hp.showTRRT, hp.color};
        addHairBSDF(bb, b, 1.0f, Physical, DwaLabelHair);
        return;
    }
    auto lobe = [&](float shift, float roughness, float aziRoughness, Color tint, float saturation) {
        return HairLobeBSDF{hp.dir, {hp.uv[0], hp.uv[1]}, hp.ior, hp.fresnelType, hp.cuticleLayerThickness, shift, roughness,
                            aziRoughness, hp.color, tint, saturation, hp.showGlint, glintRoughness, eccentricity, hp.glintSaturation,
                            hairRotation, N};
    };
    if (hp.showR) addHairRBRDF(bb, lobe(hp.rShift, hp.rLongRoughness, 0.0f, rgb(1), 1.0f), 1.0f, Physical, DwaLabelHairR);
    // TT, TRT and TRRT are adjacent: under R, they do not affect one another's energy.
    startAdjacentComponents(bb);
    if (hp.showTT)
        addHairTTBTDF(bb, lobe(hp.ttShift, hp.ttLongRoughness, hp.ttAzimRoughness, hp.ttTint, hp.ttSaturation), 1.0f, Physical,
                      DwaLabelHairTT);
    if (hp.showTRT)
        addHairTRTBRDF(bb, lobe(hp.trtShift, hp.trtLongRoughness, hp.ttAzimRoughness, hp.trtTint, 1.0f), 1.0f, Physical, DwaLabelHairTRT);
    if (hp.showTRRT)
        addHairTRRTBRDF(bb, lobe(0.0f, hp.trrtLongRoughness, hp.ttAzimRoughness, rgb(1), 1.0f), 1.0f, Physical, DwaLabelHairTRRT);
    endAdjacentComponents(bb);
}

// False when the parameters need a lobe XPU2 does not shade yet (toon ramp, subsurface, hair, glitter).
X2_CALL bool createDwaLobes(const DwaParams& p, const DwaUniformParams& u, const State& st, Builder& bb, const float* pool,
                            GlitterScratch* glitter) {
    addEmission(bb, p.emission);
    const V3 N = p.normal;
    if (u.thinGeometry) setThinGeo(bb);
    if (u.preventLightCulling) setPreventLightCulling(bb, true);
    const float minRough = minRoughness(st);

    // Fuzz
    if (!isZero(p.fuzz) && st.entering && !isZero(p.fuzzRoughness)) {
        VelvetBRDF fuzz{p.fuzzNormal, clampf(0.05f + p.fuzzRoughness, 0.0f, 1.0f), p.fuzzAlbedo, p.fuzzUseAbsorbingFibers, nullptr};
        bb.lightSet = u.diffuseLightSet;
        addVelvetBRDF(bb, fuzz, p.fuzz, Physical, DwaLabelFuzz);
    }
    bb.lightSet = u.specularLightSet;

    // Iridescence
    Iridescence iridescence;
    const Iridescence* iri = nullptr;
    const IridescenceParameters& ip = p.iridescence;
    if (!isZero(ip.iridescence)) {
        iridescence.N = N;
        iridescence.strength = ip.iridescence;
        iridescence.thickness = ip.thickness;
        iridescence.exponent = ip.exponent;
        iridescence.at0 = ip.at0;
        iridescence.at90 = ip.at90;
        if (ip.colorControl == 0) {
            iridescence.colorControl = 0;
            iridescence.primary = ip.primaryColor;
            iridescence.secondary = ip.secondaryColor;
            iridescence.flipHue = ip.flipHueDirection;
            iridescence.ramp.count = 0;
        } else {
            iridescence.colorControl = 1;
            iridescence.primary = iridescence.secondary = rgb(0);
            iridescence.flipHue = false;
            rampInit(iridescence.ramp, ip.rampNumPoints, ip.rampPositions, ip.rampColors, ip.rampInterpolators,
                     ip.rampInterpolationMode);
        }
        iri = &iridescence;
    }
    const Iridescence* outerIri = ip.applyTo == IridescenceOuterSpecular ? iri : nullptr;
    const Iridescence* primaryIri = ip.applyTo == IridescenceOuterSpecular ? nullptr : iri;

    // Outer specular (ignored when exiting)
    if (!isZero(p.outerSpecular) && st.entering) addOuterSpecularLobes(bb, p, u, minRough, outerIri);

    if (!isZero(p.glitterMask) && !createGlitterLobes(bb, st, p.glitter, p.glitterMask, pool, glitter, DwaLabelGlitter)) return false;

    // Fabric
    startAdjacentComponents(bb);
    {
        const float warpRoughness = fmaxf(p.warpRoughness, minRough), weftRoughness = fmaxf(p.weftRoughness, minRough);
        if (!isBlack(p.warpColor)) {
            FabricBRDF warp{};
            warp.N = N;
            warp.shadingTangent = p.fabricTangent;
            warp.threadDirection = p.warpThreadDirection;
            warp.threadElevation = p.warpThreadElevation;
            warp.roughness = warpRoughness;
            warp.threadColor = p.warpColor;
            warp.iridescence = nullptr;
            addFabricBRDF(bb, warp, p.warpThreadCoverage * p.fabricSpecular, Physical, DwaLabelSpecular);
        }
        if (!isBlack(p.weftColor)) {
            FabricBRDF weft{};
            weft.N = N;
            weft.shadingTangent = p.fabricTangent;
            weft.threadDirection = V3{-p.warpThreadDirection.y, p.warpThreadDirection.x, p.warpThreadDirection.z};
            weft.threadElevation = 0.f;
            weft.roughness = weftRoughness;
            weft.threadColor = p.weftColor;
            weft.iridescence = nullptr;
            addFabricBRDF(bb, weft, (1.f - p.warpThreadCoverage) * p.fabricSpecular, Physical, DwaLabelSpecular);
        }
    }
    endAdjacentComponents(bb);

    float reflIor = p.refractiveIndex;
    if (p.useIndependentTransmissionRefractiveIndex && !st.entering) reflIor = p.independentTransmissionRefractiveIndex;

    // Hair toon specular 1 to 3: a surface toon specular, or a hair one
    for (U32 n = 0; n < 3; ++n) {
        const ToonSpecularParameters& ts = p.hairToonSpecs[n];
        if (isZero(ts.toonSpecular)) continue;
        if (u.hairToonModel[n] == ToonSpecularSurface) {
            ToonSpecularBRDF t{ts.normal, ts.intensity, ts.fresnelBlend, ts.tint, ts.rampInputScale, ts.rampNumPoints, ts.rampPositions,
                               ts.rampInterpolators, ts.rampValues, ts.stretchU, ts.stretchV, ts.dPds, ts.dPdt,
                               ts.enableIndirectReflections, ts.indirectReflectionsIntensity, ts.indirectReflectionsRoughness};
            addToonSpecularBRDF(bb, t, ts.toonSpecular, Physical, DwaLabelHair);
        } else if (u.hairToonModel[n] == ToonSpecularHair) {
            HairToonSpecularBRDF t{ts.normal, ts.hairDir, {ts.hairUV[0], ts.hairUV[1]}, ts.hairIOR, ts.hairFresnelType,
                                   ts.hairCuticleLayerThickness, ts.hairRShift, ts.roughness, ts.tint, ts.intensity, ts.rampNumPoints,
                                   ts.rampPositions, ts.rampInterpolators, ts.rampValues, ts.enableIndirectReflections,
                                   ts.indirectReflectionsIntensity, ts.indirectReflectionsRoughness};
            addHairToonSpecularBRDF(bb, t, ts.toonSpecular, Physical, DwaLabelHair);
        }
    }

    if (!isZero(p.specular) || !isZero(p.transmission)) {
        const float roughness = computeMicrofacetRoughness(p.roughness, p.normalAAStrategy, p.normalLength, p.normalAADial, p.normalDial);
        const float roughnessClamped = fmaxf(roughness, minRough);
        V3 shadingTangent{0, 0, 0};
        if (!isZero(p.anisotropy)) {
            const V3 T = normalize(st.dPds);
            shadingTangent = localToGlobal(frame(N, T), normalize(V3{p.shadingTangent[0], p.shadingTangent[1], 0.f}));
        }
        // Metallic specular reflection
        const float metallicWeight = p.specular * p.metallic;
        if (!isZero(metallicWeight)) {
            if (roughnessClamped < Epsilon) {
                addMirrorBRDF(bb, mirrorConductor(N, p.metallicColor, p.metallicEdgeColor, primaryIri), metallicWeight, Physical,
                              DwaLabelSpecular);
            } else if (isZero(p.anisotropy)) {
                addMicrofacetIsotropicBRDF(bb, microfacetConductor(N, p.metallicColor, p.metallicEdgeColor, roughnessClamped,
                                                                   u.specularModel, primaryIri),
                                           metallicWeight, Physical, DwaLabelSpecular);
            } else {
                float rx, ry;
                computeAnisoRoughness(roughness, minRough, p.anisotropy, rx, ry);
                addMicrofacetAnisotropicBRDF(bb, microfacetAnisoConductor(N, p.metallicColor, p.metallicEdgeColor, rx, ry, shadingTangent,
                                                                          u.specularModel, primaryIri),
                                             metallicWeight, Physical, DwaLabelSpecular);
            }
        }
        if (isOne(metallicWeight)) return !bb.overflow;

        float refrIor = reflIor;
        if (p.useIndependentTransmissionRefractiveIndex) refrIor = p.independentTransmissionRefractiveIndex;
        // Dispersion only for primary ray hits, so that a ray is not split twice.
        const float abbeNumber = st.pathType != PathPrimary ? 0.0f : p.dispersionAbbeNumber;

        startAdjacentComponents(bb);
        // Toon specular, and its transmission
        const ToonSpecularParameters& ts = p.toonSpec;
        if (!isZero(ts.toonSpecular)) {
            ToonSpecularBRDF t{ts.normal, ts.intensity, ts.fresnelBlend, ts.tint, ts.rampInputScale, ts.rampNumPoints, ts.rampPositions,
                               ts.rampInterpolators, ts.rampValues, ts.stretchU, ts.stretchV, ts.dPds, ts.dPdt,
                               ts.enableIndirectReflections, ts.indirectReflectionsIntensity, ts.indirectReflectionsRoughness};
            addToonSpecularBRDF(bb, t, ts.toonSpecular, Physical, DwaLabelSpecular);
            if (!isZero(p.transmission)) {
                if (roughnessClamped < Epsilon) {
                    MirrorBTDF m{N, refrIor, p.transmissionColor, abbeNumber};
                    addMirrorBTDF(bb, m, p.transmission, Physical, DwaLabelSpecularTransmission);
                } else {
                    MicrofacetIsotropicBTDF m{N, refrIor, roughnessClamped, p.transmissionColor, abbeNumber, u.specularModel};
                    addMicrofacetIsotropicBTDF(bb, m, p.transmission, Physical, DwaLabelSpecularTransmission);
                }
            }
        }
        const float dielectricWeight = p.specular * (1.0f - ts.toonSpecular);
        const float transmissionWeight = p.transmission * (1.0f - ts.toonSpecular);
        float transmissionRoughness = roughness, transmissionRoughnessClamped = roughnessClamped;
        if (p.useIndependentTransmissionRoughness) {
            transmissionRoughness = computeMicrofacetRoughness(p.independentTransmissionRoughness, p.normalAAStrategy, p.normalLength,
                                                               p.normalAADial, p.normalDial);
            transmissionRoughnessClamped = fmaxf(transmissionRoughness, minRough);
        }
        // MOONRAY_MODO_TRANSMISSION_ROUGHNESS: a rough transmission keeps its roughness under a mirror smooth surface.
        if (roughnessClamped < Epsilon &&
            (!bb.modoTransmissionRoughness || isZero(transmissionWeight) || transmissionRoughnessClamped < Epsilon)) {
            MirrorBSDF m{N, reflIor, p.transmissionColor, abbeNumber, refrIor, dielectricWeight, transmissionWeight, primaryIri};
            addMirrorBSDF(bb, m, 1.f, Physical, DwaLabelSpecular, DwaLabelSpecularTransmission);
        } else if (isZero(p.anisotropy)) {
            MicrofacetIsotropicBSDF m{N, reflIor, roughnessClamped, transmissionRoughnessClamped, p.useIndependentTransmissionRoughness,
                                      p.transmissionColor, abbeNumber, refrIor, dielectricWeight, transmissionWeight, u.specularModel,
                                      primaryIri};
            addMicrofacetIsotropicBSDF(bb, m, 1.f, Physical, DwaLabelSpecular, DwaLabelSpecularTransmission);
        } else {
            float rx, ry, tx, ty;
            computeAnisoRoughness(roughness, minRough, p.anisotropy, rx, ry);
            tx = rx;
            ty = ry;
            if (p.useIndependentTransmissionRoughness) computeAnisoRoughness(transmissionRoughness, minRough, p.anisotropy, tx, ty);
            MicrofacetAnisotropicBSDF m{N, reflIor, rx, ry, tx, ty, p.useIndependentTransmissionRoughness, shadingTangent,
                                        p.transmissionColor, abbeNumber, refrIor, dielectricWeight, transmissionWeight,
                                        u.specularModel, primaryIri};
            addMicrofacetAnisotropicBSDF(bb, m, 1.f, Physical, DwaLabelSpecular, DwaLabelSpecularTransmission);
        }
        endAdjacentComponents(bb);
    }

    // Diffuse
    bb.lightSet = u.diffuseLightSet;
    const bool zeroScatterRadius = isBlack(p.scatteringRadius);
    const Color reflectionAlbedo = p.albedo * diffuseTransmissionAttenuation(p);
    const bool blackAlbedo = isBlack(reflectionAlbedo);
    // Toon ramp diffuse: ignored with any subsurface, else before the other diffuse lobes.
    if (!blackAlbedo && zeroScatterRadius && !isZero(p.toonDiffuse.toonDiffuse) && p.toonDiffuse.model == ToonDiffuseRamp) {
        const ToonDiffuseParameters& td = p.toonDiffuse;
        ToonBRDF t{td.normal, reflectionAlbedo, td.rampInputScale, td.rampNumPoints, td.rampPositions, td.rampInterpolators,
                   td.rampColors, td.extendRamp};
        addToonBRDF(bb, t, p.fabricAttenuation * td.toonDiffuse * td.rampWeight, Physical, DwaLabelDiffuse);
    }

    startAdjacentComponents(bb);
    if (!blackAlbedo) {
        if (zeroScatterRadius) {
            const V3 shadingN = u.preventLightCulling ? p.toonDiffuse.normal : p.diffuseNormal;
            const float flatness = p.toonDiffuse.flatness * p.toonDiffuse.toonDiffuse;
            FlatDiffuseBRDF d{shadingN, reflectionAlbedo, p.diffuseRoughness, p.toonDiffuse.terminatorShift, flatness,
                              p.toonDiffuse.flatnessFalloff};
            addFlatDiffuseBRDF(bb, d, p.fabricAttenuation, Physical, DwaLabelDiffuse);
        } else {
            // Subsurface: the material's diffusion profile or random walk.
            const DiffusionProfile d{p.diffuseNormal, reflectionAlbedo, p.scatteringRadius};
            if (u.subsurface == SubsurfaceDipoleDiffusion) addDiffusion(bb, BssrdfDipole, d, p.fabricAttenuation, Physical, DwaLabelDiffuse);
            else if (u.subsurface == SubsurfaceNormalizedDiffusion)
                addDiffusion(bb, BssrdfNormalizedDiffusion, d, p.fabricAttenuation, Physical, DwaLabelDiffuse);
            else if (u.subsurface == SubsurfaceRandomWalk)
                addRandomWalk(bb, RandomWalkProfile{p.diffuseNormal, reflectionAlbedo, p.scatteringRadius, p.sssResolveSelfIntersections,
                                                    p.creaseAttenuation},
                              p.fabricAttenuation, Physical, DwaLabelDiffuse);
            else return false;
        }
    }
    if (!isBlack(p.diffuseTransmission)) {
        LambertianBTDF t{-p.diffuseNormal, p.diffuseTransmission};
        addLambertianBTDF(bb, t, p.fabricAttenuation, Physical, DwaLabelDiffuseTransmission);
    }
    endAdjacentComponents(bb);

    addHairLobes(p, u, st, bb);
    bb.lightSet = 0xffffffffu;
    return !bb.overflow;
}
}}}

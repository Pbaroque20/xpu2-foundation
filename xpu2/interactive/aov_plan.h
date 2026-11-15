// The scene's render outputs as MoonRay's RenderOutputDriver sets them up (rndr/RenderOutputDriver.cc): its files,
// their parts and each part's outputs with their channel names; the AOV schema of the outputs that are AOVs, in that
// order (aov_core.h); the material AOVs' expressions (pbr/core/Aov.cc MaterialAovs, AovMat.ll and AovMat.yy) and the
// light path expressions of light and visibility AOVs (LightAovs, with MoonRay's own lpe::StateMachine from
// librendering_lpe, its transitions tabulated for every state they reach).
//
// The outputs' attributes are read by name, each with MoonRay's default where the runtime's scene library has not
// declared it, so a scene from a newer kit reads as MoonRay reads it.
#pragma once
#include "aov_core.h"
#include "shade_lobes.h"
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/Metadata.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>
#include "StateMachine.h"  // moonray/lib/rendering/lpe
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xpu2::interactive::aov {
namespace rdl = scene_rdl2::rdl2;

// AovSchemaId's range types (AOV_MAX_RANGE_TYPE, 100, apart).
enum RangeSchema : int {
    SchemaPrimFloat = 100,
    SchemaPrimVec2 = 200,
    SchemaPrimVec3 = 300,
    SchemaPrimRgb = 400,
    SchemaMaterialFloat = 500,
    SchemaMaterialVec2 = 600,
    SchemaMaterialVec3 = 700,
    SchemaMaterialRgb = 800,
    SchemaVisibility = 900,
    SchemaLight = 1000,
};
// RenderOutput::Result.
enum Result : int {
    ResultBeauty = 0,
    ResultAlpha = 1,
    ResultDepth = 2,
    ResultStateVariable = 3,
    ResultPrimitiveAttribute = 4,
    ResultHeatMap = 5,
    ResultWireframe = 6,
    ResultMaterialAov = 7,
    ResultLightAov = 8,
    ResultVisibilityAov = 9,
    ResultWeight = 11,
    ResultBeautyAux = 12,
    ResultCryptomatte = 13,
    ResultAlphaAux = 14,
    ResultDisplayFilter = 15,
};
// A light or visibility AOV's bit in the state machine's valid sets: visibility AOVs 0 to 99, light AOVs 100 to 199.
inline U32 lpeBit(int schema) { return U32(schema - SchemaVisibility); }
constexpr U32 LpeBits = 200;

inline U32 typeOf(int schema) {
    if (schema == SchemaBeauty) return TypeBeauty;
    if (schema == SchemaAlpha) return TypeAlpha;
    if (schema >= SchemaLight) return TypeLight;
    if (schema >= SchemaVisibility) return TypeVisibility;
    if (schema >= SchemaMaterialFloat) return TypeMaterial;
    if (schema >= SchemaPrimFloat) return TypePrim;
    if (schema >= int(StateP)) return TypeState;
    return 0;
}
inline int rangeBase(int schema) { return schema - schema % 100; }
// aovNumChannels.
inline U32 channelsOf(int schema) {
    switch (typeOf(schema)) {
    case TypeState:
        switch (schema) {
        case StateSt:
        case StateMotion: return 2;
        case StateDsdx:
        case StateDsdy:
        case StateDtdx:
        case StateDtdy:
        case StateWireframe:
        case StateDepth: return 1;
        default: return 3;
        }
    case TypeAlpha: return 1;
    case TypePrim:
    case TypeMaterial:
        switch (rangeBase(schema)) {
        case SchemaPrimFloat:
        case SchemaMaterialFloat: return 1;
        case SchemaPrimVec2:
        case SchemaMaterialVec2: return 2;
        default: return 3;
        }
    case TypeVisibility: return 2;
    default: return 3;
    }
}

inline std::string trim(const std::string& in, const char* ws = " \t") {
    const size_t b = in.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return in.substr(b, in.find_last_not_of(ws) - b + 1);
}
// Whether the object's class declares the attribute (SceneClass::hasAttribute only reads, though not const).
inline bool hasAttribute(const rdl::SceneObject& o, const char* name) {
    return const_cast<rdl::SceneClass&>(o.getSceneClass()).hasAttribute(name);
}
// An attribute by name, or its default where the runtime does not declare it.
template <class T>
inline T attribute(const rdl::SceneObject& o, const char* name, T fallback) {
    return hasAttribute(o, name) ? o.get<T>(name) : fallback;
}

// The material classes' lobe labels (each class's json "labels", in the order the build numbers them).
inline const std::vector<std::string>* classLabels(const std::string& cls) {
    static const std::vector<std::string> dwa{"glitter", "hair", "hair R", "hair TT", "hair TRT", "hair TRRT", "hair diffuse", "fuzz",
                                              "outer specular", "specular", "diffuse", "specular transmission", "diffuse transmission",
                                              "diffuse accent", "diffuse accent1", "diffuse accent2", "diffuse accent3", "diffuse accent3",
                                              "diffuse accent5", "diffuse accent6", "diffuse accent7", "diffuse accent8", "diffuse accent9",
                                              "specular accent", "specular accent1", "specular accent2", "specular accent3",
                                              "specular accent4", "specular accent5", "specular accent6", "specular accent7",
                                              "specular accent8", "specular accent9"};
    static const std::vector<std::string> usd{"diffuse", "specular", "specular transmission", "outer specular"};
    static const std::vector<std::string> glass{"specular", "transmission", "diffuse"};
    static const std::set<std::string> dwaClasses{"DwaAdjustMaterial", "DwaBaseMaterial", "DwaColorCorrectMaterial", "DwaEmissiveMaterial",
                                                  "DwaFabricMaterial", "DwaLayerMaterial", "DwaMetalMaterial", "DwaMixMaterial",
                                                  "DwaRefractiveMaterial", "DwaSkinMaterial", "DwaSolidDielectricMaterial",
                                                  "DwaSwitchMaterial", "DwaToonMaterial", "DwaTwoSidedMaterial", "DwaVelvetMaterial_v2",
                                                  "HairColorCorrectMaterial", "HairDiffuseMaterial", "HairLayerMaterial",
                                                  "HairToonMaterial", "HairMaterial_v3"};
    if (dwaClasses.count(cls)) return &dwa;
    if (cls == "UsdPreviewSurface") return &usd;
    if (cls == "ModoGlass") return &glass;
    return nullptr;
}

// ParsedMaterialExpression (AovMat.ll's tokens, AovMat.yy's grammar).
struct ParsedExpression {
    enum Selector : int { Reflection = 1, Transmission = 2, Diffuse = 4, Glossy = 8, Mirror = 16, Subsurface = 32 };
    U32 property = PropertyUnknown;
    U32 state = SchemaUnknown;  // a state variable's schema
    bool fresnel = false;
    int selector = 0;
    std::vector<std::vector<std::string>> labels{{}};  // the dot-separated label groups, the lobe labels last
    std::string prim;
    U32 primType = PrimFloat;
    std::string error;
};
inline ParsedExpression parseMaterialExpression(const std::string& expression) {
    ParsedExpression m;
    const std::string e = trim(expression);
    enum Kind { KLabel, KDot, KLobes, KSubsurface, KFresnel, KProperty };
    struct Token {
        Kind kind;
        std::string text;
        U32 property = 0, state = 0, primType = 0;
    };
    std::vector<Token> tokens;
    static const struct {
        const char* word;
        int kind;
        U32 property, state;
    } words[] = {{"SS", KSubsurface, 0, 0},
                 {"fresnel", KFresnel, 0, 0},
                 {"albedo", KProperty, PropertyAlbedo, 0},
                 {"color", KProperty, PropertyColor, 0},
                 {"emission", KProperty, PropertyEmission, 0},
                 {"factor", KProperty, PropertyFactor, 0},
                 {"normal", KProperty, PropertyNormal, 0},
                 {"radius", KProperty, PropertyRadius, 0},
                 {"roughness", KProperty, PropertyRoughness, 0},
                 {"matte", KProperty, PropertyMatte, 0},
                 {"pbr_validity", KProperty, PropertyPbrValidity, 0},
                 {"eye_mask", KProperty, PropertyEyeMask, 0},
                 {"P", KProperty, PropertyState, StateP},
                 {"N", KProperty, PropertyState, StateN},
                 {"Ng", KProperty, PropertyState, StateNg},
                 {"St", KProperty, PropertyState, StateSt},
                 {"dPds", KProperty, PropertyState, StateDpds},
                 {"dPdt", KProperty, PropertyState, StateDpdt},
                 {"dSdx", KProperty, PropertyState, StateDsdx},
                 {"dSdy", KProperty, PropertyState, StateDsdy},
                 {"dTdx", KProperty, PropertyState, StateDtdx},
                 {"dTdy", KProperty, PropertyState, StateDtdy},
                 {"Wp", KProperty, PropertyState, StateWp},
                 {"depth", KProperty, PropertyState, StateDepth},
                 {"motionvec", KProperty, PropertyState, StateMotion}};
    static const struct {
        const char* prefix;
        U32 type;
    } primPrefixes[] = {{"float:", PrimFloat}, {"vec2:", PrimVec2}, {"vec3:", PrimVec3}, {"rgb:", PrimRgb}};
    auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    size_t i = 0;
    while (i < e.size()) {
        if (e[i] == '\'') {
            size_t j = i + 1;
            while (j < e.size() && (std::isalnum(static_cast<unsigned char>(e[j])) || e[j] == ' ' || e[j] == '_' || e[j] == '-')) ++j;
            if (j >= e.size() || e[j] != '\'') {
                m.error = "syntax error";
                return m;
            }
            tokens.push_back({KLabel, trim(e.substr(i, j - i + 1), "' \t")});
            i = j + 1;
            continue;
        }
        if (e[i] == '.') {
            tokens.push_back({KDot, "."});
            ++i;
            continue;
        }
        // flex takes the longest match, the earlier rule on a tie.
        size_t best = 0;
        Token bestToken{KLabel, ""};
        size_t lobes = 0;
        while (i + lobes < e.size() && strchr("RTDGM", e[i + lobes]) && e[i + lobes]) ++lobes;
        if (lobes) {
            best = lobes;
            bestToken = {KLobes, e.substr(i, lobes)};
        }
        for (const auto& w : words) {
            const size_t n = strlen(w.word);
            if (e.compare(i, n, w.word) == 0 && n > best) {
                best = n;
                bestToken = {Kind(w.kind), w.word, w.property, w.state};
            }
        }
        for (const auto& p : primPrefixes) {
            const size_t n = strlen(p.prefix);
            if (e.compare(i, n, p.prefix) != 0) continue;
            size_t j = i + n;
            while (j < e.size() && word(e[j])) ++j;
            if (j > i + n && j - i > best) {
                best = j - i;
                bestToken = {KProperty, e.substr(i + n, j - i - n), PropertyPrim, 0, p.type};
            }
        }
        if (!best) {
            m.error = "syntax error";
            return m;
        }
        tokens.push_back(bestToken);
        i += best;
    }
    auto fail = [&]() {
        m.error = "syntax error";
        return m;
    };
    size_t k = 0;
    // labels DOT: label groups, a dot between groups (a label after a dot starts a new group).
    if (k < tokens.size() && tokens[k].kind == KLabel) {
        while (k < tokens.size()) {
            if (tokens[k].kind == KLabel) {
                m.labels.back().push_back(tokens[k].text);
                ++k;
            } else if (tokens[k].kind == KDot && k + 1 < tokens.size() && tokens[k + 1].kind == KLabel) {
                m.labels.push_back({});
                ++k;
            } else {
                break;
            }
        }
        if (k >= tokens.size() || tokens[k].kind != KDot) return fail();
        ++k;
    }
    // bsdf_selector DOT
    if (k < tokens.size() && (tokens[k].kind == KLobes || tokens[k].kind == KSubsurface)) {
        while (k < tokens.size() && (tokens[k].kind == KLobes || tokens[k].kind == KSubsurface)) {
            if (tokens[k].kind == KSubsurface) m.selector |= ParsedExpression::Subsurface;
            else
                for (char c : tokens[k].text)
                    m.selector |= c == 'R' ? ParsedExpression::Reflection
                                  : c == 'T' ? ParsedExpression::Transmission
                                  : c == 'D' ? ParsedExpression::Diffuse
                                  : c == 'G' ? ParsedExpression::Glossy
                                             : ParsedExpression::Mirror;
            ++k;
        }
        if (k >= tokens.size() || tokens[k].kind != KDot) return fail();
        ++k;
    }
    // FRESNEL DOT
    if (k < tokens.size() && tokens[k].kind == KFresnel) {
        m.fresnel = true;
        ++k;
        if (k >= tokens.size() || tokens[k].kind != KDot) return fail();
        ++k;
    }
    if (k + 1 != tokens.size() || tokens[k].kind != KProperty) return fail();
    m.property = tokens[k].property;
    m.state = tokens[k].state;
    if (m.property == PropertyPrim) {
        m.prim = tokens[k].text;
        m.primType = tokens[k].primType;
    }
    return m;
}

// One render output, as RenderOutputDriver's Entry holds it.
struct Output {
    const rdl::RenderOutput* ro = nullptr;
    std::string name;
    int result = 0;
    std::vector<std::string> channels;
    int schema = SchemaUnknown;  // its AOV schema id (SchemaUnknown: not in the AOV array)
    int entry = -1;              // its entry in the AOV schema
    int displayFilter = -1;      // its display filter's index
    U32 lpePrefixFlags = 0, stateAov = SchemaUnknown;
    U32 filter = FilterAvg;
    bool half = false;
    int compression = 1;
    float compressionLevel = 85.0f;
    std::string part;
    // A Cryptomatte output's category (cryptomatte_id_channel), layers (cryptomatte_depth + 1, halved) and extra data
    // a fragment (positions, P0, normals, beauty, ref_P, ref_N, UV, resume support), reflected and refracted blocks.
    int cryptoCategory = 0, cryptoLayers = 3;
    bool cryptoExtra[8] = {false, false, false, false, false, false, false, false};
    bool cryptoReflected = false, cryptoRefracted = false;
    // Its exr_header_attributes (a Metadata object), copied: the scene it came from does not outlive the frame.
    bool deep = false;  // output_type "deep": written to a deep file (DeepBuffer::write)
    bool metadata = false;
    std::string metadataName;
    std::vector<std::string> metadataNames, metadataTypes, metadataValues;
};
struct Image {
    std::string part;
    std::vector<Output> outputs;
};
struct File {
    std::string name;
    std::vector<Image> images;
    std::string checkpoint, checkpointMultiVersion, resume;
};

// LightAovs::replaceLpeAliases.
inline std::string replaceLpeAliases(const std::string& lpe) {
    const std::string v = trim(lpe);
    if (v == "caustic") return "CD[S]+[<L.>O]";
    if (v == "diffuse") return "CD[<L.>O]";
    if (v == "emission") return "CO";
    if (v == "glossy") return "CG[<L.>O]";
    if (v == "mirror") return "CS[<L.>O]";
    if (v == "reflection") return "C<RS>[DSG]+[<L.>O]";
    if (v == "translucent") return "C<TD>[DSG]+[<L.>O]";
    if (v == "transmission") return "C<TS>[DSG]+[<L.>O]";
    return lpe;
}

struct Plan {
    std::vector<File> files;
    std::vector<const Output*> all;  // every output, in MoonRay's entry order
    // The AOV schema.
    std::vector<Entry> entries;
    std::vector<const Output*> entryOutputs;
    U32 channels = 0, closestCount = 0, types = 0;
    // MaterialAovs: the entries' descriptors (and their schema bases and expressions), their label lists, and the
    // label tables the lists index.
    std::vector<MaterialAov> materials;
    std::vector<int> materialSchemas;
    std::vector<std::string> materialExpressions;
    std::vector<int> labelLists;
    std::vector<std::string> lobeLabels, materialLabels, geomLabels;
    // The primitive attributes the outputs read (name, PrimType), as Entry::prim indexes them.
    std::vector<std::pair<std::string, U32>> prims;
    // LightAovs.
    std::unique_ptr<moonray::lpe::StateMachine> machine = std::make_unique<moonray::lpe::StateMachine>();
    std::unordered_map<std::string, std::string> labelSubstitutions;
    std::vector<std::string> lpeLabelNames;  // the labels the expressions name
    int nextLight = SchemaLight, nextVisibility = SchemaVisibility;
    std::vector<int> lpe;
    std::vector<U32> lpeValid;
    U32 lpeStates = 0, lpeLabels = 0, lpeStride = 0, lpeWords = 0;
    int lpeCamera = -1;
    // What else the outputs need.
    bool wireframe = false, motion = false, weight = false, heatMap = false, beautyAux = false, alphaAux = false, cryptomatte = false,
         deep = false;
    // Cryptomatte's attributes: the first of deep_id_attribute_names, crypto_uv_attribute_name (none: st), its
    // categories (one, or three when deep_id_attribute_names names three: the film's buffers) and multi presence.
    std::string cryptoIdAttribute, cryptoUvAttribute;
    // Deep output's settings (deep_format: 0 OpenEXR 2.0, 1 OpenDCX 2.0 with subpixel masks; the merge's tolerances) and
    // its deep ID attributes; deepRefused: why XPU2 leaves the scene's deep outputs to MoonRay ("" when it renders them).
    int deepFormat = 0;
    float deepCurvature = 45.0f, deepZ = 2.0f;
    std::vector<std::string> deepIds;
    std::string deepRefused;
    U32 cryptoCategories = 1;
    bool cryptoMultiPresence = false;
    std::vector<const rdl::SceneObject*> displayFilters;
    const Output* denoiserAlbedo = nullptr;
    const Output* denoiserNormal = nullptr;
    bool resumable = false, checkpoint = false, resume = false;
    double checkpointMinutes = 15.0;  // checkpoint_interval: how often a checkpoint render saves its files
    std::vector<std::string> warnings;
    // The rezed aperture (display) and region (data) windows, min x, min y, max x, max y (half open).
    int aperture[4] = {0, 0, 0, 0}, region[4] = {0, 0, 0, 0};

    bool hasLpe() const { return nextLight > SchemaLight || nextVisibility > SchemaVisibility; }
    bool empty() const { return entries.empty(); }
    // LightAovs::getLabelId: a light's or a material's label in the state machine, -1 for none.
    int lpeLabel(const std::string& label) const { return label.empty() || !hasLpe() ? -1 : machine->getLabelId(label); }
    static int find(const std::vector<std::string>& v, const std::string& s) {
        auto i = std::find(v.begin(), v.end(), s);
        return s.empty() || i == v.end() ? -1 : int(i - v.begin());
    }
    int materialAovLabel(const std::string& l) const { return find(lobeLabels, l); }
    int materialAovMaterialLabel(const std::string& l) const { return find(materialLabels, l); }
    int geomLabel(const std::string& l) const { return find(geomLabels, l); }
    // Whether materials need their label tables.
    bool labelsNeeded() const { return !materials.empty() || hasLpe(); }
    // A material's label table (Builder::labels): its count, then the encoded labels of its shader's label 0 (the
    // material's own) and of each lobe label of its class (RenderContext's material and light aov label ids).
    std::vector<U32> labelTable(const rdl::SceneObject& material) const {
        const std::string matLabel = attribute<rdl::String>(material, "label", "");
        std::vector<U32> t;
        t.push_back(1);
        t.push_back(shade::encodedLabel(materialAovMaterialLabel(matLabel), lpeLabel(matLabel)));
        if (const auto* labels = classLabels(material.getSceneClass().getName())) {
            const std::string prefix = matLabel.empty() ? "" : matLabel + ".";
            for (const auto& l : *labels) t.push_back(shade::encodedLabel(materialAovLabel(l), lpeLabel(prefix + l)));
        }
        t[0] = U32(t.size() - 1);
        return t;
    }
    Tables tables() const {
        Tables t;
        t.entries = entries.data();
        t.count = U32(entries.size());
        t.channels = channels;
        t.closestCount = closestCount;
        t.types = types;
        t.lpe = lpe.data();
        t.lpeStates = lpeStates;
        t.lpeLabels = lpeLabels;
        t.lpeStride = lpeStride;
        t.lpeCamera = lpeCamera;
        t.lpeBackground = lpeLabel("U:bg_white");
        t.lpeValid = lpeValid.data();
        t.lpeWords = lpeWords;
        t.materials = materials.data();
        t.labelLists = labelLists.data();
        t.materialCount = U32(materials.size());
        for (const auto& m : materials) t.materialSecondary |= m.secondary && m.lpe != None ? 1u : 0u;
        t.crypto = cryptomatte ? 1u : 0u;
        t.cryptoIdKey = cryptoIdAttribute.empty() ? 0u : shade::primKey(cryptoIdAttribute.c_str(), shade::PrimFloat);
        t.cryptoUvKey = cryptoUvAttribute.empty() ? 0u : shade::primKey(cryptoUvAttribute.c_str(), shade::PrimVec2f);
        t.deep = deep && deepRefused.empty() ? 1u : 0u;
        t.deepIdCount = U32(std::min<size_t>(deepIds.size(), 3));
        for (U32 i = 0; i < t.deepIdCount; ++i) t.deepIdKeys[i] = shade::primKey(deepIds[i].c_str(), shade::PrimFloat);
        return t;
    }

    // LightAovs::expandLpeLabels.
    std::string expandLpeLabels(const std::string& lpe) const {
        std::string out = lpe;
        size_t open = out.find_first_of('\'');
        while (open != std::string::npos) {
            const size_t close = out.find_first_of('\'', open + 1);
            if (close == std::string::npos) return lpe;
            const std::string label = trim(out.substr(open + 1, close - open - 1));
            auto it = labelSubstitutions.find(label);
            if (it != labelSubstitutions.end()) {
                out.replace(open, close - open + 1, it->second);
                open = out.find_first_of('\'', open + it->second.length());
            } else {
                open = out.find_first_of('\'', close + 1);
            }
        }
        return out;
    }
    // LightAovs::createEntry (and the labels the expression names, as StateMachine::addExpression scans them).
    int createLpe(const std::string& expression, bool visibility, U32& prefixFlags) {
        prefixFlags = 0;
        std::string lpe = expression;
        const size_t semicolon = lpe.find_first_of(';');
        if (semicolon != std::string::npos) {
            if (trim(lpe.substr(0, semicolon)) == "unoccluded") prefixFlags |= PrefixUnoccluded;
            else warnings.push_back("Unrecognized prefix found: \"" + trim(lpe.substr(0, semicolon)) + "\". Ignoring...");
            lpe = lpe.substr(semicolon + 1);
        }
        lpe = expandLpeLabels(replaceLpeAliases(lpe));
        int id;
        if (visibility) {
            if (nextVisibility >= SchemaLight) return SchemaUnknown;
            id = nextVisibility;
        } else {
            if (nextLight >= SchemaLight + 100) return SchemaUnknown;
            id = nextLight;
        }
        if (machine->addExpression(lpe, id)) return SchemaUnknown;
        for (size_t t0 = lpe.find_first_of('\''); t0 != std::string::npos;) {
            const size_t t1 = lpe.find_first_of('\'', t0 + 1);
            if (t1 == std::string::npos) break;
            const std::string label = trim(lpe.substr(t0 + 1, t1 - t0 - 1));
            if (std::find(lpeLabelNames.begin(), lpeLabelNames.end(), label) == lpeLabelNames.end()) lpeLabelNames.push_back(label);
            t0 = lpe.find_first_of('\'', t1 + 1);
        }
        return visibility ? nextVisibility++ : nextLight++;
    }

    // MaterialAovs::createEntry over parseExpression: its schema id (SchemaUnknown: an error).
    int createMaterialAov(const std::string& name, int lpeSchema, int lpeLabelId, bool secondary, U32& stateAov, int& prim) {
        const U32 lpeBitOf = lpeSchema == SchemaUnknown ? None : lpeBit(lpeSchema);
        stateAov = SchemaUnknown;
        prim = -1;
        for (size_t i = 0; i < materials.size(); ++i)
            if (materialExpressions[i] == name && materials[i].lpe == lpeBitOf) {
                // findEntry: the entry this expression already has
                if (materials[i].property == PropertyState) stateAov = parseMaterialExpression(name).state;
                return materialSchemas[i];
            }
        const ParsedExpression m = parseMaterialExpression(name);
        if (!m.error.empty()) {
            warnings.push_back("Unable to parse material expression \"" + name + "\": " + m.error);
            return SchemaUnknown;
        }
        MaterialAov a{};
        a.property = m.property;
        a.fresnel = m.fresnel ? 1u : 0u;
        a.secondary = secondary ? 1u : 0u;
        a.lpeLabel = lpeLabelId;
        a.lpe = lpeBitOf;
        const std::vector<std::string> lobe = m.labels.back();
        const std::vector<std::string> material = m.labels.size() > 1 ? m.labels[m.labels.size() - 2] : std::vector<std::string>{};
        const std::vector<std::string> geom = m.labels.size() > 2 ? m.labels[m.labels.size() - 3] : std::vector<std::string>{};
        if (!lobe.empty() && m.property == PropertyEmission) return SchemaUnknown;
        // The lobe flags (shading::BsdfLobe::Type, in XPU2's bits): all lobes by default, none for "SS." alone.
        const U32 sides = shade::AllSurfaceSides, kinds = shade::AllLobes;
        if (m.selector == 0) a.lobeFlags = sides | kinds;
        else if (m.selector == ParsedExpression::Subsurface) a.lobeFlags = 0;
        else {
            a.lobeFlags = 0;
            if (m.selector & (ParsedExpression::Reflection | ParsedExpression::Transmission)) {
                if (m.selector & ParsedExpression::Reflection) a.lobeFlags |= shade::Reflection;
                if (m.selector & ParsedExpression::Transmission) a.lobeFlags |= shade::Transmission;
            } else {
                a.lobeFlags |= sides;
            }
            if (m.selector & (ParsedExpression::Diffuse | ParsedExpression::Glossy | ParsedExpression::Mirror)) {
                if (m.selector & ParsedExpression::Diffuse) a.lobeFlags |= shade::Diffuse;
                if (m.selector & ParsedExpression::Glossy) a.lobeFlags |= shade::Glossy;
                if (m.selector & ParsedExpression::Mirror) a.lobeFlags |= shade::Mirror;
            } else {
                a.lobeFlags |= kinds;
            }
        }
        a.subsurface = (m.selector & ParsedExpression::Subsurface) || m.selector == 0 ? 1u : 0u;
        int base = SchemaUnknown;
        switch (m.property) {
        case PropertyAlbedo:
            if (m.fresnel) return SchemaUnknown;
            base = SchemaMaterialRgb;
            break;
        case PropertyColor: base = SchemaMaterialRgb; break;
        case PropertyEmission:
            if (m.selector != 0 || m.fresnel) return SchemaUnknown;
            base = SchemaMaterialRgb;
            break;
        case PropertyFactor:
            if (!m.fresnel) return SchemaUnknown;
            base = SchemaMaterialFloat;
            break;
        case PropertyNormal:
            if (m.fresnel) return SchemaUnknown;
            base = SchemaMaterialVec3;
            break;
        case PropertyRadius:
            if ((m.selector != 0 && m.selector != ParsedExpression::Subsurface) || m.fresnel) return SchemaUnknown;
            base = SchemaMaterialRgb;
            break;
        case PropertyRoughness:
            if ((m.selector & ParsedExpression::Subsurface) && !m.fresnel) return SchemaUnknown;
            base = SchemaMaterialVec2;
            break;
        case PropertyMatte:
            if (m.fresnel) return SchemaUnknown;
            base = SchemaMaterialFloat;
            break;
        case PropertyPbrValidity:
        case PropertyEyeMask:
            if (m.fresnel) return SchemaUnknown;
            base = SchemaMaterialRgb;
            break;
        case PropertyState: {
            stateAov = m.state;
            const U32 n = channelsOf(int(m.state));
            base = n == 1 ? SchemaMaterialFloat : n == 2 ? SchemaMaterialVec2 : SchemaMaterialVec3;
            break;
        }
        case PropertyPrim: {
            base = m.primType == PrimFloat ? SchemaMaterialFloat
                   : m.primType == PrimVec2 ? SchemaMaterialVec2
                   : m.primType == PrimVec3 ? SchemaMaterialVec3
                                            : SchemaMaterialRgb;
            prim = primIndex(m.prim, m.primType);
            break;
        }
        default: return SchemaUnknown;
        }
        auto indices = [&](const std::vector<std::string>& labels, std::vector<std::string>& table, U32& first, U32& count) {
            first = U32(labelLists.size());
            count = U32(labels.size());
            for (const auto& l : labels) {
                int at = find(table, l);
                if (at < 0) {
                    table.push_back(l);
                    at = int(table.size() - 1);
                }
                labelLists.push_back(at);
            }
        };
        indices(lobe, lobeLabels, a.labels, a.labelCount);
        indices(material, materialLabels, a.materialLabels, a.materialLabelCount);
        indices(geom, geomLabels, a.geomLabels, a.geomLabelCount);
        if (materials.size() >= 100) return SchemaUnknown;
        a.pad = prim < 0 ? None : U32(prim);
        materials.push_back(a);
        materialSchemas.push_back(base + int(materials.size() - 1));
        materialExpressions.push_back(name);
        return materialSchemas.back();
    }
    int primIndex(const std::string& name, U32 type) {
        const auto key = std::make_pair(name, type);
        auto it = std::find(prims.begin(), prims.end(), key);
        if (it != prims.end()) return int(it - prims.begin());
        prims.push_back(key);
        return int(prims.size() - 1);
    }
};

// LightAovs::buildLabelSubstitutions: each material label, and each lobe label, to all of its
// material_label.lobe_label combinations among the layer's materials.
inline void buildLabelSubstitutions(Plan& plan, const rdl::SceneObject* layer) {
    if (!layer || !hasAttribute(*layer, "surface_shaders")) return;
    std::set<const rdl::SceneObject*> materials;
    for (const rdl::SceneObject* s : layer->get<rdl::SceneObjectVector>("surface_shaders")) {
        if (!s) continue;
        rdl::ConstSceneObjectSet b;
        s->getBindingTransitiveClosure(b);
        for (const rdl::SceneObject* o : b)
            if (o && o->isA<rdl::Material>()) materials.insert(o);
    }
    std::map<std::string, std::vector<std::string>> mat2lobe, lobe2mat;
    auto add = [](std::vector<std::string>& v, const std::string& s) {
        if (std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
    };
    for (const rdl::SceneObject* m : materials) {
        const std::string matLabel = attribute<rdl::String>(*m, "label", "");
        const auto* lobes = classLabels(m->getSceneClass().getName());
        if (!lobes || matLabel.empty()) continue;
        for (const auto& l : *lobes) {
            add(mat2lobe[matLabel], l);
            add(lobe2mat[l], matLabel);
        }
    }
    for (const auto& e : mat2lobe) {
        std::string target = "['" + e.first + "'";
        for (const auto& l : e.second) target += "'" + e.first + "." + l + "'";
        plan.labelSubstitutions[e.first] = target + "]";
    }
    for (const auto& e : lobe2mat) {
        if (plan.labelSubstitutions.count(e.first)) continue;
        std::string target = "['" + e.first + "'";
        for (const auto& m : e.second) target += "'" + m + "." + e.first + "'";
        plan.labelSubstitutions[e.first] = target + "]";
    }
}

// stateVariableToSchema and defBaseChannelName(StateVariable).
inline int stateSchema(int sv) {
    static const int map[] = {StateP, StateNg, StateN, StateSt, StateDpds, StateDpdt, StateDsdx,
                              StateDsdy, StateDtdx, StateDtdy, StateWp, StateDepth, StateMotion};
    return sv >= 0 && sv < 13 ? map[sv] : SchemaUnknown;
}
inline std::string stateBaseName(int sv) {
    static const char* const names[] = {"P", "Ng", "N", "ST", "dPds", "dPdt", "dSdx", "dSdy", "dTdx", "dTdy", "Wp", "Z"};
    return sv >= 0 && sv < 12 ? names[sv] : "";
}

// The state machine's transitions for every state reachable from the camera's, and the expressions valid in each.
inline void tabulateLpe(Plan& plan) {
    using namespace moonray::lpe;
    if (!plan.hasLpe()) return;
    plan.machine->build();
    int labels = 0;
    for (const auto& name : plan.lpeLabelNames) labels = std::max(labels, plan.machine->getLabelId(name) + 1);
    plan.lpeLabels = U32(labels);
    plan.lpeStride = LpeLabelledKinds * (plan.lpeLabels + 1) + 2;
    plan.lpeCamera = plan.machine->transition(StateMachine::sInitialStateId, EVENT_TYPE_CAMERA, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
    struct Kind {
        EventType ev;
        EventScatteringType evs;
    };
    const Kind kinds[LpeLabelledKinds] = {{EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_DIFFUSE},  {EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_GLOSSY},
                                          {EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_MIRROR},   {EVENT_TYPE_TRANSMISSION, EVENT_SCATTERING_TYPE_DIFFUSE},
                                          {EVENT_TYPE_TRANSMISSION, EVENT_SCATTERING_TYPE_GLOSSY}, {EVENT_TYPE_TRANSMISSION, EVENT_SCATTERING_TYPE_MIRROR},
                                          {EVENT_TYPE_LIGHT, EVENT_SCATTERING_TYPE_NONE},          {EVENT_TYPE_EMISSION, EVENT_SCATTERING_TYPE_NONE},
                                          {EVENT_TYPE_EXTRA, EVENT_SCATTERING_TYPE_NONE},          {EVENT_TYPE_MATERIAL, EVENT_SCATTERING_TYPE_NONE}};
    std::map<int, std::vector<int>> rows;
    std::vector<int> todo;
    if (plan.lpeCamera >= 0) todo.push_back(plan.lpeCamera);
    int maxState = plan.lpeCamera;
    while (!todo.empty()) {
        const int s = todo.back();
        todo.pop_back();
        if (rows.count(s)) continue;
        std::vector<int> row(plan.lpeStride, -1);
        for (U32 k = 0; k < LpeLabelledKinds; ++k)
            for (int l = -1; l < int(plan.lpeLabels); ++l) {
                if ((kinds[k].ev == EVENT_TYPE_EXTRA || kinds[k].ev == EVENT_TYPE_MATERIAL) && l < 0) continue;  // labelled events only
                row[k * (plan.lpeLabels + 1) + U32(l + 1)] = plan.machine->transition(s, kinds[k].ev, kinds[k].evs, l);
            }
        row[LpeLabelledKinds * (plan.lpeLabels + 1)] = plan.machine->transition(s, EVENT_TYPE_TRANSMISSION, EVENT_SCATTERING_TYPE_STRAIGHT, sNoLabel);
        row[LpeLabelledKinds * (plan.lpeLabels + 1) + 1] = plan.machine->transition(s, EVENT_TYPE_VOLUME, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        for (int n : row)
            if (n >= 0 && !rows.count(n)) {
                todo.push_back(n);
                maxState = std::max(maxState, n);
            }
        rows[s] = std::move(row);
    }
    plan.lpeStates = U32(maxState + 1);
    plan.lpe.assign(size_t(plan.lpeStates) * plan.lpeStride, -1);
    for (const auto& r : rows) std::copy(r.second.begin(), r.second.end(), plan.lpe.begin() + size_t(r.first) * plan.lpeStride);
    plan.lpeWords = (LpeBits + 31) / 32;
    plan.lpeValid.assign(size_t(plan.lpeStates) * plan.lpeWords, 0u);
    for (const auto& r : rows)
        for (int id = SchemaVisibility; id < SchemaLight + 100; ++id) {
            if ((id < SchemaLight && id >= plan.nextVisibility) || id >= plan.nextLight) continue;
            if (plan.machine->isValid(r.first, id)) plan.lpeValid[size_t(r.first) * plan.lpeWords + lpeBit(id) / 32] |= 1u << (lpeBit(id) % 32);
        }
}

// The storage of an output (getStorageType) decides its output type, and so its channel suffixes (getOutputType).
enum OutputType { OutputFloat, OutputVec2, OutputVec3, OutputRgb };
inline OutputType outputType(int result, int schema, int primType) {
    switch (result) {
    case ResultBeauty:
    case ResultBeautyAux:
    case ResultDisplayFilter: return OutputRgb;
    case ResultAlpha:
    case ResultAlphaAux:
    case ResultHeatMap:
    case ResultWeight:
    case ResultWireframe:
    case ResultVisibilityAov: return OutputFloat;
    case ResultStateVariable:
    case ResultDepth: {
        const U32 n = channelsOf(schema);
        return n == 1 ? OutputFloat : n == 2 ? OutputVec2 : OutputVec3;
    }
    case ResultPrimitiveAttribute: {
        const U32 n = channelsOf(schema);
        return n == 1 ? OutputFloat : n == 2 ? OutputVec2 : primType == PrimRgb ? OutputRgb : OutputVec3;
    }
    case ResultMaterialAov: {
        const U32 n = channelsOf(schema);
        return n == 1 ? OutputFloat : n == 2 ? OutputVec2 : rangeBase(schema) == SchemaMaterialRgb ? OutputRgb : OutputVec3;
    }
    case ResultLightAov: return OutputRgb;
    default: return OutputFloat;
    }
}

// getChannelNames.
inline std::vector<std::string> channelNames(const rdl::RenderOutput& ro, int result, int schema, bool resumable) {
    std::vector<std::string> names;
    std::string base = attribute<rdl::String>(ro, "channel_name", "");
    if (base.empty()) {
        switch (result) {
        case ResultBeauty:
        case ResultBeautyAux:
        case ResultDisplayFilter: break;
        case ResultAlpha: base = "A"; break;
        case ResultAlphaAux: base = "Aaux"; break;
        case ResultHeatMap: base = "heat"; break;
        case ResultWeight: base = "weight"; break;
        case ResultWireframe: base = "wireframe"; break;
        case ResultStateVariable: base = stateBaseName(attribute<rdl::Int>(ro, "state_variable", 2)); break;
        case ResultDepth: base = stateBaseName(11); break;
        case ResultPrimitiveAttribute: base = attribute<rdl::String>(ro, "primitive_attribute", ""); break;
        case ResultMaterialAov: base = attribute<rdl::String>(ro, "material_aov", ""); break;
        case ResultLightAov: base = attribute<rdl::String>(ro, "lpe", ""); break;
        case ResultVisibilityAov: base = "visiblility_" + attribute<rdl::String>(ro, "visibility_aov", ""); break;
        case ResultCryptomatte: base = "cryptomatte"; break;
        default: base = "unknown";
        }
    }
    if (result == ResultBeauty && base.empty()) return {"R", "G", "B"};
    if (result == ResultBeautyAux && base.empty()) return {"__beautyAux__.R", "__beautyAux__.G", "__beautyAux__.B"};
    if (resumable && result == ResultVisibilityAov) return {base + ".hit", base + ".att", base + ".vis"};
    if (result == ResultCryptomatte) {
        const int layers = (attribute<rdl::Int>(ro, "cryptomatte_depth", 6) + 1) / 2, fragments = layers * 2;
        char name[80];
        auto rgba = [&](const char* prefix, int n, bool alpha) {
            for (const char* c : {".R", ".G", ".B", ".A"}) {
                if (!alpha && (!strcmp(c, ".B") || !strcmp(c, ".A"))) continue;
                snprintf(name, sizeof(name), "%s%02d%s", prefix, n, c);
                names.push_back(name);
            }
        };
        auto block = [&](const std::string& tag) {
            for (int l = 0; l < layers; ++l) rgba(("Cryptomatte" + tag).c_str(), l, true);
            for (int f = 0; f < fragments; ++f) {
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_positions", false)) rgba(("Crypto" + tag + "P").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_p0", false)) rgba(("Crypto" + tag + "P0").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_normals", false)) rgba(("Crypto" + tag + "N").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_beauty", false)) rgba(("Crypto" + tag + "B").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_refp", false)) rgba(("Crypto" + tag + "RefP").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_refn", false)) rgba(("Crypto" + tag + "RefN").c_str(), f, true);
                if (attribute<rdl::Bool>(ro, "cryptomatte_output_uv", false)) rgba(("Crypto" + tag + "UV").c_str(), f, false);
                if (attribute<rdl::Bool>(ro, "cryptomatte_support_resume_render", false)) rgba(("Crypto" + tag + "S").c_str(), f, false);
            }
        };
        block("");
        if (attribute<rdl::Bool>(ro, "cryptomatte_record_reflected", false)) block("Reflect");
        if (attribute<rdl::Bool>(ro, "cryptomatte_record_refracted", false)) block("Refract");
        return names;
    }
    const int mode = attribute<rdl::Int>(ro, "channel_suffix_mode", 0);
    static const char* const rgb[] = {".R", ".G", ".B"};
    static const char* const xyz[] = {".X", ".Y", ".Z"};
    static const char* const uvw[] = {".U", ".V", ".W"};
    const char* const* suffix = mode == 1 ? rgb : mode == 2 ? xyz : mode == 3 ? uvw : nullptr;
    const bool closestDepth = resumable && attribute<rdl::Int>(ro, "math_filter", 0) == FilterClosest;
    switch (outputType(result, schema, attribute<rdl::Int>(ro, "primitive_attribute_type", 0))) {
    case OutputFloat: names.push_back(base + (result == ResultVisibilityAov ? ".vis" : "")); break;
    case OutputVec2:
        names.push_back(base + (suffix ? suffix[0] : ".X"));
        names.push_back(base + (suffix ? suffix[1] : ".Y"));
        break;
    case OutputVec3:
        for (int c = 0; c < 3; ++c) names.push_back(base + (suffix ? suffix[c] : xyz[c]));
        break;
    case OutputRgb:
        for (int c = 0; c < 3; ++c) names.push_back(base + (suffix ? suffix[c] : rgb[c]));
        break;
    }
    if (closestDepth) names.push_back(base + ".depth");
    return names;
}

// RenderOutputDriver::Impl's constructor: the files, parts and outputs; the AOV schema; the LPE tables.
// The render outputs in the order the scene declared them (SceneContext::getAllRenderOutputs of the scene as it was
// read), which orders the files' parts and channels: a copy of the scene made since (the canonical snapshot) holds
// them by name. Set by the session before each plan; empty: the context's own order.
inline std::vector<std::string>& outputOrder() {
    static thread_local std::vector<std::string> order;
    return order;
}
inline std::shared_ptr<Plan> buildPlan(const rdl::SceneContext& context) {
    auto plan = std::make_shared<Plan>();
    Plan& p = *plan;
    const rdl::SceneVariables& sv = context.getSceneVariables();
    buildLabelSubstitutions(p, sv.getLayer());
    p.resume = attribute<rdl::Bool>(sv, "resume_render", false);
    {
        const auto a = sv.getRezedApertureWindow(), r = sv.getRezedRegionWindow();
        p.aperture[0] = a.min().x;
        p.aperture[1] = a.min().y;
        p.aperture[2] = a.max().x;
        p.aperture[3] = a.max().y;
        p.region[0] = r.min().x;
        p.region[1] = r.min().y;
        p.region[2] = r.max().x;
        p.region[3] = r.max().y;
    }
    p.resumable = p.resume || attribute<rdl::Bool>(sv, "resumable_output", false);
    {
        const auto deepIds = attribute<rdl::StringVector>(sv, "deep_id_attribute_names", rdl::StringVector{});
        p.deepIds.assign(deepIds.begin(), deepIds.end());
        p.deepFormat = attribute<rdl::Int>(sv, "deep_format", 0) == 1 ? 1 : 0;
        p.deepCurvature = attribute<rdl::Float>(sv, "deep_curvature_tolerance", 45.0f);
        p.deepZ = attribute<rdl::Float>(sv, "deep_z_tolerance", 2.0f);
        p.cryptoIdAttribute = deepIds.empty() ? std::string() : deepIds[0];
        p.cryptoCategories = U32(std::max<size_t>(1, std::min<size_t>(deepIds.size(), 3)));
        p.cryptoUvAttribute = attribute<rdl::String>(sv, "crypto_uv_attribute_name", "");
        p.cryptoMultiPresence = attribute<rdl::Bool>(sv, "cryptomatte_multi_presence", false);
    }
    p.checkpoint = p.resume || attribute<rdl::Bool>(sv, "checkpoint_active", false);
    p.checkpointMinutes = attribute<rdl::Float>(sv, "checkpoint_interval", 15.0f);
    std::vector<const rdl::RenderOutput*> renderOutputs(context.getAllRenderOutputs().begin(), context.getAllRenderOutputs().end());
    if (!outputOrder().empty()) {
        const auto& order = outputOrder();
        auto rank = [&](const rdl::RenderOutput* ro) {
            const auto at = std::find(order.begin(), order.end(), ro->getName());
            return size_t(at - order.begin());
        };
        std::stable_sort(renderOutputs.begin(), renderOutputs.end(),
                         [&](const rdl::RenderOutput* a, const rdl::RenderOutput* b) { return rank(a) < rank(b); });
    }
    for (const rdl::RenderOutput* ro : renderOutputs) {
        if (!attribute<rdl::Bool>(*ro, "active", true)) continue;
        Output out;
        out.ro = ro;
        out.name = ro->getName();
        out.result = attribute<rdl::Int>(*ro, "result", 0);
        out.half = attribute<rdl::Int>(*ro, "channel_format", 1) == 1;
        out.compression = attribute<rdl::Int>(*ro, "compression", 1);
        out.compressionLevel = attribute<rdl::Float>(*ro, "exr_dwa_compression_level", 85.0f);
        out.part = attribute<rdl::String>(*ro, "file_part", "");
        if (const rdl::SceneObject* m = attribute<rdl::SceneObject*>(*ro, "exr_header_attributes", nullptr))
            if (m->isA<rdl::Metadata>()) {
                const rdl::Metadata* md = m->asA<rdl::Metadata>();
                out.metadata = true;
                out.metadataName = md->getName();
                out.metadataNames = md->getAttributeNames();
                out.metadataTypes = md->getAttributeTypes();
                out.metadataValues = md->getAttributeValues();
            }
        int schema = SchemaUnknown;
        U32 stateAov = SchemaUnknown, prefixFlags = 0;
        int primType = attribute<rdl::Int>(*ro, "primitive_attribute_type", 0);
        switch (out.result) {
        case ResultStateVariable:
        case ResultDepth: {
            const int svar = out.result == ResultDepth ? 11 : attribute<rdl::Int>(*ro, "state_variable", 2);
            schema = stateSchema(svar);
            if (schema == StateMotion) p.motion = true;
            break;
        }
        case ResultPrimitiveAttribute: {
            const std::string name = attribute<rdl::String>(*ro, "primitive_attribute", "");
            if (name.empty() || primType < 0 || primType > 3) break;
            const int index = p.primIndex(name, U32(primType));
            schema = (primType == PrimFloat ? SchemaPrimFloat : primType == PrimVec2 ? SchemaPrimVec2 : primType == PrimVec3 ? SchemaPrimVec3 : SchemaPrimRgb) +
                     (index % 100);
            break;
        }
        case ResultMaterialAov: {
            const std::string expression = attribute<rdl::String>(*ro, "material_aov", "");
            if (expression.empty()) break;
            int lpeSchema = SchemaUnknown, lpeLabelId = -1;
            std::string lpe = attribute<rdl::String>(*ro, "lpe", "");
            if (!lpe.empty()) {
                const std::string label = "M:" + expression;
                lpe += "'" + label + "'";
                lpeSchema = p.createLpe(lpe, false, prefixFlags);
                // The label's id is known once the machine is built: kept by name until then.
                lpeLabelId = -2;
                (void)label;
            }
            int prim = -1;
            schema = p.createMaterialAov(expression, lpeSchema, lpeLabelId, attribute<rdl::Bool>(*ro, "material_aov_secondary_rays", false),
                                         stateAov, prim);
            if (schema != SchemaUnknown && stateAov == StateMotion) p.motion = true;
            break;
        }
        case ResultWireframe: schema = StateWireframe; break;
        case ResultLightAov: {
            const std::string lpe = attribute<rdl::String>(*ro, "lpe", "");
            if (!lpe.empty()) schema = p.createLpe(lpe, false, prefixFlags);
            break;
        }
        case ResultVisibilityAov: {
            const std::string lpe = attribute<rdl::String>(*ro, "visibility_aov", "C[<T.><RS>]*[<R[DG]><TD>][LO]");
            U32 ignored;
            if (!lpe.empty()) schema = p.createLpe(lpe, true, ignored);
            break;
        }
        case ResultBeauty: schema = SchemaBeauty; break;
        case ResultAlpha: schema = SchemaAlpha; break;
        case ResultCryptomatte: {
            p.cryptomatte = true;
            out.cryptoCategory = attribute<rdl::Int>(*ro, "cryptomatte_id_channel", 0);
            out.cryptoLayers = std::min(100, std::max(1, (attribute<rdl::Int>(*ro, "cryptomatte_depth", 6) + 1) / 2));
            static const char* const extra[8] = {"cryptomatte_output_positions", "cryptomatte_output_p0", "cryptomatte_output_normals",
                                                 "cryptomatte_output_beauty", "cryptomatte_output_refp", "cryptomatte_output_refn",
                                                 "cryptomatte_output_uv", "cryptomatte_support_resume_render"};
            for (int k = 0; k < 8; ++k) out.cryptoExtra[k] = attribute<rdl::Bool>(*ro, extra[k], false);
            out.cryptoReflected = attribute<rdl::Bool>(*ro, "cryptomatte_record_reflected", false);
            out.cryptoRefracted = attribute<rdl::Bool>(*ro, "cryptomatte_record_refracted", false);
            break;
        }
        case ResultHeatMap: p.heatMap = true; break;
        case ResultWeight: p.weight = true; break;
        case ResultBeautyAux: p.beautyAux = true; break;
        case ResultAlphaAux: p.alphaAux = true; break;
        case ResultDisplayFilter: break;
        default: break;
        }
        if (schema == SchemaUnknown && out.result != ResultHeatMap && out.result != ResultWeight && out.result != ResultBeautyAux &&
            out.result != ResultAlphaAux && out.result != ResultCryptomatte && out.result != ResultDisplayFilter)
            continue;  // skipped, as MoonRay skips it (with its message)
        if (out.result == ResultDisplayFilter) {
            const rdl::SceneObject* df = attribute<rdl::SceneObject*>(*ro, "display_filter", nullptr);
            if (!df) continue;
        }
        if (attribute<rdl::String>(*ro, "output_type", "flat") == "deep") {
            p.deep = out.deep = true;
            // The beauty and alpha alone, as XPU2 keeps them a segment; a deep AOV is left to MoonRay.
            if (out.result != ResultBeauty && out.result != ResultAlpha && p.deepRefused.empty())
                p.deepRefused = out.name + ": a deep output of other than the beauty or alpha";
        }
        out.schema = schema;
        out.lpePrefixFlags = prefixFlags;
        out.stateAov = stateAov;
        // getAovFilter: only state, primitive attribute and material AOVs take the other filters, and extra light AOVs
        // ('U:' labels) avg, sum, min and max; light AOVs with an 'M:' label any.
        U32 filter = FilterAvg;
        if (schema != SchemaUnknown) {
            filter = U32(attribute<rdl::Int>(*ro, "math_filter", 0));
            const U32 type = typeOf(schema);
            const std::string lpe = attribute<rdl::String>(*ro, "lpe", "");
            const bool extra = type == TypeLight && lpe.find("'U:") != std::string::npos;
            const bool materialLpe = type == TypeLight && lpe.find("'M:") != std::string::npos;
            const bool extraFilter = filter <= FilterMax;
            if (type != TypeState && type != TypePrim && type != TypeMaterial && filter != FilterAvg && filter != FilterConsistent &&
                filter != FilterClosest && !materialLpe && !(extra && extraFilter)) {
                p.warnings.push_back(out.name + ": unsupported math filter");
                filter = FilterAvg;
            }
            if (filter > FilterClosest) filter = FilterAvg;
        }
        out.filter = filter;
        out.channels = channelNames(*ro, out.result, schema, p.resumable);
        // The file and its part.
        const std::string fileName = attribute<rdl::String>(*ro, "file_name", "scene.exr");
        auto file = std::find_if(p.files.begin(), p.files.end(), [&](const File& f) { return f.name == fileName; });
        if (file == p.files.end()) {
            p.files.push_back(File{fileName, {}, "", "", ""});
            file = p.files.end() - 1;
        }
        if (p.checkpoint) {
            file->checkpoint = attribute<rdl::String>(*ro, "checkpoint_file_name", "checkpoint.exr");
            file->checkpointMultiVersion = attribute<rdl::String>(*ro, "checkpoint_multi_version_file_name", "");
        }
        if (p.resume && !attribute<rdl::String>(*ro, "resume_file_name", "").empty()) file->resume = attribute<rdl::String>(*ro, "resume_file_name", "");
        auto image = std::find_if(file->images.begin(), file->images.end(), [&](const Image& i) { return i.part == out.part; });
        if (image == file->images.end()) {
            file->images.push_back(Image{out.part, {}});
            image = file->images.end() - 1;
        }
        // Conflicts (the same channel for the same result, different compression or metadata, a part on one only).
        bool conflict = false;
        for (const Output& other : image->outputs) {
            const rdl::RenderOutput* ro2 = other.ro;
            bool sameChannel = false;
            for (const auto& c0 : out.channels)
                for (const auto& c1 : other.channels) sameChannel |= c0 == c1 && out.result == other.result;
            const auto* m0 = attribute<rdl::SceneObject*>(*ro, "exr_header_attributes", nullptr);
            const auto* m1 = attribute<rdl::SceneObject*>(*ro2, "exr_header_attributes", nullptr);
            if (sameChannel || out.compression != other.compression || out.compressionLevel != other.compressionLevel ||
                (m0 != m1 && m0 && m1) || out.part.empty() != other.part.empty())
                conflict = true;
        }
        if (conflict) {
            p.warnings.push_back(out.name + ": conflicts with another output of its image, skipped");
            continue;
        }
        image->outputs.push_back(out);
    }
    // The entry order: files, their parts, their outputs.
    for (File& f : p.files)
        for (Image& i : f.images)
            for (Output& out : i.outputs) {
                p.all.push_back(&out);
                if (out.result == ResultDisplayFilter) {
                    out.displayFilter = int(p.displayFilters.size());
                    p.displayFilters.push_back(attribute<rdl::SceneObject*>(*out.ro, "display_filter", nullptr));
                }
                if (out.schema == SchemaUnknown) continue;
                Entry e{};
                e.schema = U32(out.schema);
                e.type = typeOf(out.schema);
                e.channels = channelsOf(out.schema);
                e.filter = out.filter;
                e.offset = p.channels;
                e.closest = out.filter == FilterClosest ? p.closestCount++ : None;
                e.lpe = (e.type == TypeLight || e.type == TypeVisibility) ? lpeBit(out.schema) : None;
                e.prefixFlags = out.lpePrefixFlags;
                e.state = e.type == TypeState ? U32(out.schema) : out.stateAov;
                e.material = e.type == TypeMaterial ? U32(out.schema % 100) : None;
                e.prim = None;
                e.primType = 0;
                if (e.type == TypePrim) {
                    const int index = out.schema % 100;
                    e.prim = U32(index);
                    e.primType = p.prims[size_t(index)].second;
                } else if (e.type == TypeMaterial) {
                    const MaterialAov& m = p.materials[e.material];
                    if (m.pad != None) {
                        e.prim = m.pad;
                        e.primType = p.prims[m.pad].second;
                    }
                }
                if (out.schema == StateWireframe) p.wireframe = true;
                out.entry = int(p.entries.size());
                p.entries.push_back(e);
                p.entryOutputs.push_back(&out);
                p.channels += e.channels;
                p.types |= e.type;
                const int di = attribute<rdl::Int>(*out.ro, "denoiser_input", 0);
                if (di == 1 && !p.denoiserAlbedo) p.denoiserAlbedo = &out;
                if (di == 2 && !p.denoiserNormal) p.denoiserNormal = &out;
            }
    tabulateLpe(p);
    // A material AOV's 'M:' label, now that the machine knows its labels.
    for (size_t i = 0; i < p.materials.size(); ++i)
        if (p.materials[i].lpeLabel == -2) p.materials[i].lpeLabel = p.machine->getLabelId("M:" + p.materialExpressions[i]);
    return plan;
}

}  // namespace xpu2::interactive::aov

// Private experimental: the host half of map bindings (shade_maps.h,
// shade_mapnodes.h). A material's bound attributes become MapBindings onto its
// parameter block; the maps they reach (and the maps those are bound to) become
// one map program in dependency order. Included by cpu_scene.h after its
// helpers. Derived from MoonRay Apache-2.0 (the map DSOs' attributes),
// Copyright 2023-2024 DreamWorks Animation LLC. See MOONRAY_LICENSE.txt.
#pragma once
#include "shade_mapnodes.h"
#include <cstring>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <scene_rdl2/render/util/Random.h>
namespace xpu2::interactive::cpu {

struct MapAttribute {
    const char* name;
    shade::MapAttrType type;
};
struct MapClass {
    shade::MapKind kind;
    std::vector<MapAttribute> attributes;
};
inline unsigned mapAttrFloats(shade::MapAttrType t) {
    return t == shade::MapAttrVec4f ? 4 : t == shade::MapAttrRgb || t == shade::MapAttrVec3f ? 3 : t == shade::MapAttrVec2f ? 2 : 1;
}
inline const MapClass* mapClass(const std::string& cls) {
#define X2_MAP_ATTR(T, n) MapAttribute{#n, shade::MapAttr##T},
#define X2_MAP_ENTRY(kind, name, list) {name, MapClass{shade::kind, {list(X2_MAP_ATTR)}}},
    static const std::map<std::string, MapClass> classes = [] {
        std::map<std::string, MapClass> m = {X2_MAPS(X2_MAP_ENTRY)};
        MapClass sc{shade::MapSwitchColor, {{"choice", shade::MapAttrFloat}}}, sf{shade::MapSwitchFloat, {{"choice", shade::MapAttrFloat}}};
        static std::vector<std::string> names;
        names.reserve(64);
        for (int i = 0; i < 64; ++i) names.push_back("input" + std::to_string(i));
        for (int i = 0; i < 64; ++i) {
            sc.attributes.push_back({names[i].c_str(), shade::MapAttrRgb});
            sf.attributes.push_back({names[i].c_str(), shade::MapAttrFloat});
        }
        m.emplace("SwitchColorMap", sc);
        m.emplace("SwitchFloatMap", sf);
        m.emplace("ProjectTriplanarMap_v2", MapClass{shade::MapProjectTriplanar, {}});
        m.emplace("ProjectCameraMap_v2", MapClass{shade::MapProjectCamera, {}});
        m.emplace("ProjectTriplanarNormalMap_v2", MapClass{shade::MapProjectTriplanarNormal, {}});
        m.emplace("ProjectPlanarNormalMap", MapClass{shade::MapProjectPlanarNormal, {}});
        m.emplace("ProjectCameraNormalMap", MapClass{shade::MapProjectCameraNormal, {}});
        m.emplace("RgbToNormalMap", MapClass{shade::MapRgbToNormal, {{"input", shade::MapAttrRgb}}});
        m.emplace("SwitchNormalMap", MapClass{shade::MapSwitchNormal, {{"choice", shade::MapAttrFloat}}});
        m.emplace("CombineNormalMap",
                  MapClass{shade::MapCombineNormal, {{"normal_map_1_dial", shade::MapAttrFloat}, {"normal_map_2_dial", shade::MapAttrFloat}}});
        return m;
    }();
#undef X2_MAP_ENTRY
#undef X2_MAP_ATTR
    auto it = classes.find(cls);
    return it == classes.end() ? nullptr : &it->second;
}
// The rule a bound attribute of this type follows (EvalAttribute.isph).
inline unsigned bindRule(rdl::AttributeType t) {
    switch (t) {
    case rdl::TYPE_FLOAT: return shade::BindFloat;
    case rdl::TYPE_RGB: return shade::BindColor;
    case rdl::TYPE_VEC2F: return shade::BindVec2;
    case rdl::TYPE_VEC3F: return shade::BindVec3;
    case rdl::TYPE_INT: return shade::BindInt;
    default: throw Unsupported("Unsupported bound attribute type");
    }
}
// Writes one attribute's value as floats.
inline void writeAttribute(const rdl::SceneObject& o, const MapAttribute& a, float* out) {
    const std::string why = o.getName() + ": invalid " + a.name;
    switch (a.type) {
    case shade::MapAttrBool: out[0] = get<rdl::Bool>(o, a.name) ? 1.0f : 0.0f; return;
    case shade::MapAttrInt: out[0] = float(get<rdl::Int>(o, a.name)); return;
    case shade::MapAttrFloat: out[0] = get<rdl::Float>(o, a.name); need(std::isfinite(out[0]), why); return;
    case shade::MapAttrRgb: {
        auto c = get<rdl::Rgb>(o, a.name);
        out[0] = c.r;
        out[1] = c.g;
        out[2] = c.b;
        break;
    }
    case shade::MapAttrVec2f: {
        auto v = get<rdl::Vec2f>(o, a.name);
        out[0] = v.x;
        out[1] = v.y;
        need(std::isfinite(out[0]) && std::isfinite(out[1]), why);
        return;
    }
    case shade::MapAttrVec4f: {
        auto v = get<rdl::Vec4f>(o, a.name);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
        need(std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]) && std::isfinite(out[3]), why);
        return;
    }
    case shade::MapAttrVec3f: {
        auto v = get<rdl::Vec3f>(o, a.name);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        break;
    }
    }
    need(std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]), why);
}

// A projection map's projector (moonshine projection::getProjectorXform): its object-to-world, false
// when it has none (projection mode 'projector' without a projector).
inline bool projectorMatrix(const rdl::SceneObject& m, rdl::Mat4d& out) {
    namespace sm = scene_rdl2::math;
    const int mode = get<rdl::Int>(m, "projection_mode");
    if (mode == 0) {
        const rdl::SceneObject* p = get<rdl::SceneObject*>(m, "projector");
        if (!p) return false;
        need(p->isA<rdl::Node>(), m.getName() + ": its projector is not a node");
        out = get<rdl::Mat4d>(*p, "node_xform");
        return true;
    }
    if (mode == 1) {
        out = get<rdl::Mat4d>(m, "projection_matrix");
        return true;
    }
    need(mode == 2, m.getName() + ": invalid projection mode");
    const int order = get<rdl::Int>(m, "TRS_order"), rotationOrder = get<rdl::Int>(m, "rotation_order");
    need(order >= 0 && order <= 5 && rotationOrder >= 0 && rotationOrder <= 5, m.getName() + ": invalid TRS order");
    const auto r = get<rdl::Vec3d>(m, "rotate"), t = get<rdl::Vec3d>(m, "translate"), s = get<rdl::Vec3d>(m, "scale");
    // The axes in each rotation order (xyz, xzy, yxz, yzx, zxy, zyx), applied first to last.
    static const int axes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    sm::Xform3d rot(sm::one);
    for (int k = 0; k < 3; ++k) {
        const int a = axes[rotationOrder][k];
        // getProjectorXform takes the Vec3d attributes as Vec3f.
        const float angle = float(a == 0 ? r.x : a == 1 ? r.y : r.z);
        rot *= rot.rotate(sm::Vec3d(a == 0 ? 1.0 : 0.0, a == 1 ? 1.0 : 0.0, a == 2 ? 1.0 : 0.0), double(sm::deg2rad(angle)));
    }
    const sm::Xform3d S = sm::Xform3d::scale(sm::Vec3d(float(s.x), float(s.y), float(s.z))),
                      T = sm::Xform3d::translate(sm::Vec3d(float(t.x), float(t.y), float(t.z)));
    // Scale Rot Trans, Scale Trans Rot, Rot Scale Trans, Rot Trans Scale, Trans Scale Rot, Trans Rot Scale.
    const sm::Xform3d* steps[6][3] = {{&S, &rot, &T}, {&S, &T, &rot}, {&rot, &S, &T}, {&rot, &T, &S}, {&T, &S, &rot}, {&T, &rot, &S}};
    sm::Xform3d transform(sm::one);
    for (int k = 0; k < 3; ++k) transform *= *steps[order][k];
    out = rdl::Mat4d(transform);
    return true;
}

// Noise::orderedCompose: a rotation in the rotation order, then scale, rotation and translation in the
// transformation order (row vectors: the first applied first).
inline scene_rdl2::math::Xform3f noiseCompose(const scene_rdl2::math::Vec3f& t, const scene_rdl2::math::Vec3f& r,
                                              const scene_rdl2::math::Vec3f& s, int order, int rotationOrder) {
    namespace sm = scene_rdl2::math;
    using X = sm::Xform3f;
    const sm::Vec3f axis[3] = {sm::Vec3f(1.0f, 0.0f, 0.0f), sm::Vec3f(0.0f, 1.0f, 0.0f), sm::Vec3f(0.0f, 0.0f, 1.0f)};
    const float angle[3] = {r.x, r.y, r.z};
    static const int axes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    X rot(sm::one);
    if (rotationOrder >= 0 && rotationOrder <= 5)
        for (int k = 0; k < 3; ++k) rot *= X::rotate(axis[axes[rotationOrder][k]], angle[axes[rotationOrder][k]]);
    const X S = X::scale(s), T = X::translate(t);
    const X* steps[6][3] = {{&S, &rot, &T}, {&S, &T, &rot}, {&rot, &S, &T}, {&rot, &T, &S}, {&T, &S, &rot}, {&T, &rot, &S}};
    X result(sm::one);
    if (order >= 0 && order <= 5)
        for (int k = 0; k < 3; ++k) result = result * *steps[order][k];
    return result;
}
// One noise's tables as Noise and Perlin build them: scene_rdl2's Random seeded with the seed shuffles 0..2047,
// then (Perlin) draws unit gradients in the unit ball by rejection.
inline void noiseTables(std::vector<float>& out, int seed, bool perlin, bool fourD) {
    namespace sm = scene_rdl2::math;
    const int n = shade::NoiseTableSize;
    scene_rdl2::util::Random rng(seed);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int v : perm) {
        float f;
        std::memcpy(&f, &v, sizeof f);
        out.push_back(f);
    }
    if (!perlin) return;
    std::vector<float> g((fourD ? 4 : 3) * n);
    for (int i = 0; i < n; ++i) {
        if (!fourD) {
            sm::Vec3f v;
            do {
                v.x = rng.getNextFloat() * 2 - 1;
                v.y = rng.getNextFloat() * 2 - 1;
                v.z = rng.getNextFloat() * 2 - 1;
            } while (dot(v, v) > 1);
            v = normalize(v);
            g[i] = v.x;
            g[i + n] = v.y;
            g[i + 2 * n] = v.z;
        } else {
            sm::Vec4f v;
            do {
                v.x = rng.getNextFloat() * 2 - 1;
                v.y = rng.getNextFloat() * 2 - 1;
                v.z = rng.getNextFloat() * 2 - 1;
                v.w = rng.getNextFloat() * 2 - 1;
            } while (dot(v, v) > 1);
            v = normalize(v);
            g[i] = v.x;
            g[i + n] = v.y;
            g[i + 2 * n] = v.z;
            g[i + 3 * n] = v.w;
        }
    }
    out.insert(out.end(), g.begin(), g.end());
}

// Worley's tables (Worley::Worley, no normals): Noise's permutation, then the feature points' x, y, z from
// the same generator; the Poisson point-count bounds (initPointProbabilities).
inline void worleyTables(std::vector<float>& out, int seed) {
    const int n = shade::NoiseTableSize;
    scene_rdl2::util::Random rng(seed);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int v : perm) {
        float f;
        std::memcpy(&f, &v, sizeof f);
        out.push_back(f);
    }
    std::vector<float> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i) {
        x[i] = rng.getNextFloat();
        y[i] = rng.getNextFloat();
        z[i] = rng.getNextFloat();
    }
    out.insert(out.end(), x.begin(), x.end());
    out.insert(out.end(), y.begin(), y.end());
    out.insert(out.end(), z.begin(), z.end());
}
inline void worleyProbabilities(float low[10], float high[10]) {
    int factorial = 1;
    for (int m = 1; m <= 9; ++m) {
        factorial *= m;
        const float prob = 1.0f / (powf(3.0f, float(-m)) * expf(3.0f) * float(factorial));
        low[m] = m == 1 ? 0.0f : high[m - 1];
        high[m] = low[m] + prob;
    }
    high[9] = std::numeric_limits<float>::max();
    low[0] = high[0] = 0.0f;
}

// What ImageMap admission needs from the plan: the frozen asset root, the texture system
// and the scene's texture table (each file once).
struct TextureContext {
    fs::path assetRoot;
    std::shared_ptr<OIIO::TextureSystem>* system = nullptr;
    std::vector<std::shared_ptr<texture::File>>* textures = nullptr;
    const std::vector<std::shared_ptr<texture::File>>* previous = nullptr;  // the previous plan's, for reuse
};

// A material's map program, built as its bound attributes are admitted.
struct MapProgramBuilder {
    std::vector<float>& pool;
    TextureContext& textures;
    std::map<const rdl::SceneObject*, unsigned> registers;
    std::set<const rdl::SceneObject*> visiting;
    std::vector<unsigned> nodes;

    // An ImageMap with alpha_only reads its files' alpha (texture::File::alpha) while it is admitted.
    bool alphaLookup = false;
    unsigned textureIndex(const rdl::SceneObject& o, const char* attribute = "texture") {
        const std::string path = get<rdl::String>(o, attribute);
        if (path.empty()) return shade::MapNoTexture;
        need(path.find("<UDIM>") == std::string::npos && path.find("<udim>") == std::string::npos,
             o.getName() + ": UDIM textures are read by ImageMap only");
        return fileIndex(o, path);
    }
    // UdimTexture's tiles: the files beside the pattern whose name has a number where <UDIM> is; as many tiles as
    // the highest number less 1000, a missing one MapNoTexture. Empty when the path is not a UDIM pattern.
    std::vector<unsigned> udimTiles(const rdl::SceneObject& o, bool& eightBit) {
        const std::string path = get<rdl::String>(o, "texture");
        const size_t at = path.find("<UDIM>");
        if (at == std::string::npos) return {};
        const fs::path pattern(path);
        need(pattern.is_absolute(), o.getName() + ": a UDIM pattern must be absolute");
        // Compared by file name: the directory's listing joins with its own separator.
        const std::string file = pattern.filename().string();
        const size_t fileAt = file.find("<UDIM>");
        const std::string prefix = file.substr(0, fileAt);
        std::vector<std::string> names;
        std::error_code e;
        for (auto& entry : fs::directory_iterator(pattern.parent_path(), e))
            if (entry.is_regular_file() && entry.path().filename().string().rfind(prefix, 0) == 0)
                names.push_back(entry.path().filename().string());
        std::sort(names.begin(), names.end());
        int count = 0;
        for (auto& n : names) {
            if (n.size() < fileAt + 4) continue;
            const std::string digits = n.substr(fileAt, 4);
            if (std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
                count = std::max(count, std::stoi(digits) - 1000);
        }
        need(count > 0 && count <= 100, o.getName() + ": no UDIM tiles (or more than 100)");
        std::vector<unsigned> tiles(size_t(count), shade::MapNoTexture);
        for (int i = 0; i < count; ++i) {
            std::string name = file, full = path;
            name.replace(fileAt, 6, std::to_string(1001 + i));
            full.replace(at, 6, std::to_string(1001 + i));
            if (std::find(names.begin(), names.end(), name) == names.end()) continue;
            tiles[size_t(i)] = fileIndex(o, full);
            eightBit = (*textures.textures)[tiles[size_t(i)]]->eightBit;  // UdimTexture: the last tile's format
        }
        return tiles;
    }
    unsigned fileIndex(const rdl::SceneObject& o, const std::string& path) {
        need(!textures.assetRoot.empty() && textures.textures && textures.system, "Textured preparation requires frozen asset admission");
        texture::admittedPath(fs::path(path), textures.assetRoot);
        auto& table = *textures.textures;
        for (size_t i = 0; i < table.size(); ++i)
            if (table[i]->name.string() == path && table[i]->alpha == alphaLookup) return unsigned(i);
        std::shared_ptr<texture::File> file;
        if (textures.previous)
            for (auto& old : *textures.previous)
                if (old->name.string() == path && old->alpha == alphaLookup) file = old;
        if (file) {
            *textures.system = file->system;
        } else {
            if (!*textures.system) *textures.system = texture::system();
            file = std::make_shared<texture::File>(fs::path(path), textures.assetRoot, *textures.system, alphaLookup);
        }
        need(table.size() < 256, "At most 256 textures");
        table.push_back(file);
        return unsigned(table.size() - 1);
    }

    // The more maps (shade_mapnodes_more.h): the attributes they read beside their lists, refusals, required
    // attributes, and prepared data after their attributes.
    std::vector<unsigned> required;  // primitive attributes a map requires (MoonRay's mRequiredAttributes)
    // Each map's registers it reads (bindings and the normal maps its parameters name), and the maps being built.
    std::vector<std::vector<unsigned>> nodeDeps, depStack;
    bool polyVertices = false;       // a map reads the hit polygon's vertices (ToonMap, CurvatureMap, WireframeMap)
    static void moreListed(shade::MapKind kind, const rdl::SceneObject& m, std::set<std::string>& listed) {
        auto all = [&]() {
            for (auto i = m.getSceneClass().beginAttributes(); i != m.getSceneClass().endAttributes(); ++i) listed.insert((*i)->getName());
        };
        switch (kind) {
        case shade::MapDirectional: listed.insert({"object", "input_normal"}); break;
        case shade::MapTransformSpace: listed.insert({"camera", "object"}); break;
        case shade::MapProjectTriplanarUdim: all(); break;
        case shade::MapExtraAov: listed.insert({"label", "post_scatter"}); break;
        case shade::MapList: all(); break;
        case shade::MapDebug: listed.insert("primitive_attribute_name"); break;
        case shade::MapDistortNormal: listed.insert("input_normals"); break;
        default: break;
        }
    }
    // An object's world-to-object as 3x4 rows for xformPoint (MoonRay's matrices are row vectors).
    static void worldToObjectRows(const rdl::Mat4d& nodeXform, float* rows) {
        const auto w = nodeXform.inverse();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) rows[4 * i + j] = float(w[j][i]);
    }
    // A Mat4d (row vectors) as 3x4 rows for xformPoint.
    static void affineRows(const rdl::Mat4d& M, float* rows) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) rows[4 * i + j] = float(M[j][i]);
    }
    // rdl2::Camera::computeProjectionMatrix (interocular 0) for perspective and orthographic cameras; identity for
    // the others (doesSupportProjectionMatrix).
    static scene_rdl2::math::Mat4f cameraToScreen(const rdl::SceneObject& cam, const float window[4]) {
        namespace sm = scene_rdl2::math;
        const std::string cls = cam.getSceneClass().getName();
        if (cls != "PerspectiveCamera" && cls != "OrthographicCamera") return sm::Mat4f(sm::one);
        const bool perspective = cls == "PerspectiveCamera";
        const float minimum = perspective ? 0.00001f : 0.01f;
        const float focal = perspective ? sm::max(get<rdl::Float>(cam, "focal"), 0.00001f) : 1.0f;
        const float halfFilmWidth = sm::max(get<rdl::Float>(cam, "film_width_aperture"), minimum) * 0.5f;
        const float near = sm::max(get<rdl::Float>(cam, "near"), minimum), far = sm::max(get<rdl::Float>(cam, "far"), minimum);
        const float offsetX = get<rdl::Float>(cam, "horizontal_film_offset"), offsetY = get<rdl::Float>(cam, "vertical_film_offset");
        float par = get<rdl::Float>(cam, "pixel_aspect_ratio");
        if (sm::isEqual(par, 0.0f)) par = 1.0f;
        const float factor = perspective ? near / focal : 1.0f;
        const float w0 = (window[0] * halfFilmWidth + offsetX) * factor, w1 = (window[1] * halfFilmWidth + offsetY) * factor / par,
                    w2 = (window[2] * halfFilmWidth + offsetX) * factor, w3 = (window[3] * halfFilmWidth + offsetY) * factor / par;
        sm::Mat4f c2s(sm::zero);
        if (perspective) {
            c2s[0][0] = (2.0f * near) / (w2 - w0);
            c2s[1][1] = (2.0f * near) / (w3 - w1);
            c2s[2][0] = (w2 + w0) / (w2 - w0);
            c2s[2][1] = (w3 + w1) / (w3 - w1);
            c2s[2][2] = -(far + near) / (far - near);
            c2s[2][3] = -1.0f;
            c2s[3][2] = -(2.0f * far * near) / (far - near);
        } else {
            c2s[0][0] = 2.0f / (w2 - w0);
            c2s[1][1] = 2.0f / (w3 - w1);
            c2s[2][2] = -2.0f / (far - near);
            c2s[3][0] = -(w2 + w0) / (w2 - w0);
            c2s[3][1] = -(w3 + w1) / (w3 - w1);
            c2s[3][2] = -(far + near) / (far - near);
            c2s[3][3] = 1.0f;
        }
        return c2s;
    }
    // The active camera's node transform: render to world (MoonRay's render space is its space).
    static rdl::Mat4d renderToWorld(const rdl::SceneObject& m) {
        const auto cameras = m.getSceneClass().getSceneContext()->getActiveCameras();
        need(!cameras.empty(), m.getName() + ": no active camera");
        return get<rdl::Mat4d>(*cameras[0], "node_xform");
    }
    void morePrepared(shade::MapKind kind, const rdl::SceneObject& m, shade::MapNodeHeader& h, std::vector<float>& params,
                      const std::set<std::string>& bound) {
        namespace sm = scene_rdl2::math;
        switch (kind) {
        case shade::MapAxisAngle:
            for (const char* n : {"input_space", "axis_space", "output_space"}) {
                const int s = get<rdl::Int>(m, n);
                need(s == 0 || s == 1 || s == 2 || s == 4, m.getName() + ": screen space is not shaded yet");
            }
            break;
        case shade::MapRandom:
        case shade::MapRandomNormal: {
            // RandomMap's and RandomNormalMap's static tables: scene_rdl2's Random, default seeded.
            scene_rdl2::util::Random rnd;
            const int per = kind == shade::MapRandom ? 3 : 1;
            for (int i = 0; i < 2048 * per; ++i) params.push_back(rnd.getNextFloat());
            break;
        }
        case shade::MapHairColumn: required.push_back(shade::primKey("scatter_tag", shade::PrimFloat)); break;
        case shade::MapToon:
        case shade::MapCurvature:
        case shade::MapWireframe: polyVertices = true; break;
        case shade::MapDeformation: {
            required.push_back(shade::primKey("ref_P", shade::PrimVec3f));
            const auto c = get<rdl::Rgb>(m.getSceneClass().getSceneContext()->getSceneVariables(), "fatal_color");
            params.insert(params.end(), {c.r, c.g, c.b});
            break;
        }
        case shade::MapDirectional: {
            if (get<rdl::Bool>(m, "use_reference_space")) required.push_back(shade::primKey("ref_P", shade::PrimVec3f));
            float rows[12] = {};
            const rdl::SceneObject* object = get<rdl::SceneObject*>(m, "object");
            if (object) {
                need(object->isA<rdl::Node>(), m.getName() + ": its object is not a node");
                worldToObjectRows(get<rdl::Mat4d>(*object, "node_xform"), rows);
                h.object = 1;
            }
            params.insert(params.end(), rows, rows + 12);
            const rdl::SceneObject* input = get<rdl::SceneObject*>(m, "input_normal");
            need(!input || input->isA<rdl::NormalMap>(), m.getName() + ": its input normal is not a normal map");
            params.push_back(input ? float(int(reg(*input))) : -1.0f);
            break;
        }
        case shade::MapTransformSpace: {
            const int from = get<rdl::Int>(m, "from_space"), to = get<rdl::Int>(m, "to_space"), type = get<rdl::Int>(m, "input_type");
            need(from >= 0 && from <= 12 && to >= 0 && to <= 12 && type >= 0 && type <= 2, m.getName() + ": invalid space or type");
            if (to == 5) required.push_back(shade::primKey("ref_P", shade::PrimVec3f));
            // shading::Xform: render to world (the active camera), to the map's camera (else the active one) and to its
            // screen through the map's window (else the square -1..1), and the map's object.
            float window[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
            if (get<rdl::Bool>(m, "use_custom_window_coordinates")) {
                window[0] = get<rdl::Float>(m, "window_x_min");
                window[1] = get<rdl::Float>(m, "window_y_min");
                window[2] = get<rdl::Float>(m, "window_x_max");
                window[3] = get<rdl::Float>(m, "window_y_max");
            }
            const rdl::Mat4d r2w = renderToWorld(m);
            const rdl::SceneObject* camObj = get<rdl::SceneObject*>(m, "camera");
            const rdl::SceneObject& cam = camObj ? *camObj : *m.getSceneClass().getSceneContext()->getActiveCameras()[0];
            need(cam.isA<rdl::Camera>(), m.getName() + ": its camera is not a camera");
            const rdl::Mat4d r2c = r2w * get<rdl::Mat4d>(cam, "node_xform").inverse();
            const sm::Mat4f r2s = sm::toFloat(r2c) * cameraToScreen(cam, window), s2r = r2s.inverse();
            shade::TransformSpacePrep x{};
            affineRows(r2w, x.r2w);
            affineRows(r2w.inverse(), x.w2r);
            affineRows(r2c, x.r2c);
            affineRows(r2c.inverse(), x.c2r);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    x.r2s[4 * i + j] = r2s[i][j];
                    x.s2r[4 * i + j] = s2r[i][j];
                }
            const rdl::SceneObject* object = get<rdl::SceneObject*>(m, "object");
            if (object) {
                need(object->isA<rdl::Node>(), m.getName() + ": its object is not a node");
                worldToObjectRows(get<rdl::Mat4d>(*object, "node_xform"), x.w2o);
                h.object = 1;
            }
            const float* f = reinterpret_cast<const float*>(&x);
            for (size_t i = 0; i < sizeof(x) / sizeof(float); ++i) need(std::isfinite(f[i]), m.getName() + ": a space cannot be inverted");
            params.insert(params.end(), f, f + sizeof(x) / sizeof(float));
            break;
        }
        case shade::MapProjectTriplanarUdim: {
            // The projector (projection::getProjectorXform) as the projection maps', then each face's composeTransform2D.
            if (get<rdl::Bool>(m, "use_reference_space")) required.push_back(shade::primKey("ref_P", shade::PrimVec3f));
            rdl::Mat4d M;
            float rows[24] = {};
            if (projectorMatrix(m, M)) {
                const auto w = M.inverse();
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) rows[4 * i + j] = float(w[j][i]);
                    for (int j = 0; j < 3; ++j) rows[12 + 4 * i + j] = float(M[i][j]);
                }
                for (float v : rows) need(std::isfinite(v), m.getName() + ": its projector cannot be inverted");
                h.object = 1;
            }
            params.insert(params.end(), rows, rows + 24);
            for (const char* a : {"x", "y", "z"}) {
                const std::string p(a);
                const auto t = get<rdl::Vec2f>(m, (p + "_offset").c_str()), s = get<rdl::Vec2f>(m, (p + "_scale").c_str()),
                           c = get<rdl::Vec2f>(m, (p + "_rotation_center").c_str());
                const float r = sm::deg2rad(get<rdl::Float>(m, (p + "_rotation").c_str()));
                sm::Xform3f x(sm::one);
                x = x * sm::Xform3f::translate(sm::Vec3f(-c.x, -c.y, 0));
                x = x * sm::Xform3f::rotate(sm::Vec3f(0.0f, 0.0f, 1.0f), r);
                x = x * sm::Xform3f::translate(sm::Vec3f(c.x, c.y, 0));
                x = x * sm::Xform3f::scale(sm::Vec3f(s.x, s.y, 0));
                x = x * sm::Xform3f::translate(sm::Vec3f(t.x, t.y, 0));
                const float xr[12] = {x.l.vx.x, x.l.vx.y, x.l.vx.z, x.l.vy.x, x.l.vy.y, x.l.vy.z, x.l.vz.x, x.l.vz.y, x.l.vz.z, x.p.x, x.p.y, x.p.z};
                params.insert(params.end(), xr, xr + 12);
            }
            break;
        }
        case shade::MapDebug: {
            const int type = get<rdl::Int>(m, "map_type"), primType = get<rdl::Int>(m, "primitive_attribute_type");
            need(type >= 0 && type <= 6 && primType >= 0 && primType <= 3, m.getName() + ": invalid debug setting");
            need(get<rdl::Int>(m, "input_normal_space") == 0 || get<rdl::Int>(m, "input_normal_space") == 1,
                 m.getName() + ": invalid input normal space");
            // DebugMap's primitive attribute types: float, vec2f, vec3f, rgb.
            static const unsigned prim[4] = {shade::PrimFloat, shade::PrimVec2f, shade::PrimVec3f, shade::PrimRgb};
            const unsigned key = shade::primKey(get<rdl::String>(m, "primitive_attribute_name").c_str(), prim[primType]);
            if (type == 6) required.push_back(key);
            float f;
            std::memcpy(&f, &key, sizeof f);
            params.push_back(f);
            params.push_back(bound.count("input_normal") ? 1.0f : 0.0f);
            break;
        }
        case shade::MapDistortNormal: {
            const int space = get<rdl::Int>(m, "noise_space");
            need(space != 8 && space != 9, m.getName() + ": hair surface coordinates are not shaded yet");
            if (space == int(shade::SpaceReference)) required.push_back(shade::primKey("ref_P", shade::PrimVec3f));
            const rdl::SceneObject* input = get<rdl::SceneObject*>(m, "input_normals");
            need(!input || input->isA<rdl::NormalMap>(), m.getName() + ": its input is not a normal map");
            params.push_back(input ? float(int(reg(*input))) : -1.0f);
            const size_t offsets = params.size();
            params.insert(params.end(), 2, 0.0f);
            const int seed = get<rdl::Int>(m, "seed");
            params[offsets] = float(params.size());
            noiseTables(params, seed, false, false);
            params[offsets + 1] = float(params.size());
            noiseTables(params, seed + 2, false, false);
            break;
        }
        default: break;
        }
    }

    // The bindings of o's bound attributes among those listed, at their offsets in o's parameter block.
    void bindingsOf(const rdl::SceneObject& o, const std::vector<std::pair<std::string, unsigned>>& attrs,
                    std::vector<shade::MapBinding>& out) {
        for (auto& [name, offset] : attrs) {
            const rdl::Attribute* a = nullptr;  // none for a placeholder field (HairColumnMap, ListMap)
            for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i)
                if ((*i)->getName() == name) {
                    a = *i;
                    break;
                }
            if (!a || !a->isBindable()) continue;
            const rdl::SceneObject* m = o.getBinding(*a);
            if (!m) continue;
            out.push_back(shade::MapBinding{offset, bindRule(a->getType()), reg(*m)});
        }
    }
    // The register holding map m's result; admits m and what it is bound to first.
    unsigned reg(const rdl::SceneObject& m) {
        auto it = registers.find(&m);
        unsigned r;
        if (it != registers.end()) {
            r = it->second;
        } else {
            depStack.emplace_back();
            r = regBuild(m);
            if (nodeDeps.size() <= r) nodeDeps.resize(r + 1);
            nodeDeps[r] = std::move(depStack.back());
            depStack.pop_back();
        }
        if (!depStack.empty()) depStack.back().push_back(r);
        return r;
    }
    unsigned regBuild(const rdl::SceneObject& m) {
        auto it = registers.find(&m);
        if (it != registers.end()) return it->second;
        need(m.isA<rdl::Map>() || m.isA<rdl::NormalMap>(), m.getName() + ": a binding that is not a map");
        need(visiting.insert(&m).second, m.getName() + ": map cycle");
        const std::string cls = m.getSceneClass().getName();
        const bool primvarReader = cls.rfind("UsdPrimvarReader_", 0) == 0;
        const MapClass* c = mapClass(primvarReader ? std::string("UsdPrimvarReader") : cls);
        need(c != nullptr, m.getName() + ": map class " + cls + " is not shaded yet");
        // Every attribute of the class may hold any value; the shaded ones are read, the rest must be default.
        std::set<std::string> listed, bound;
        std::vector<std::pair<std::string, unsigned>> offsets;
        unsigned floats = 0;
        for (auto& a : c->attributes) {
            listed.insert(a.name);
            offsets.push_back({a.name, floats});
            floats += mapAttrFloats(a.type);
        }
        shade::MapNodeHeader h{};
        std::vector<unsigned> udim;
        h.kind = c->kind;
        h.texture = shade::MapNoTexture;
        h.wrap = 1;
        if (c->kind == shade::MapImage) {
            listed.insert("texture");
            listed.insert("wrap_around");
            listed.insert("gamma");
            need(get<rdl::Int>(m, "texture_coordinates") == 0 || get<rdl::Int>(m, "texture_coordinates") == 2,
                 m.getName() + ": hair texture coordinates unsupported");
            // alpha_only: the texture's alpha (its own texture of (a, a, a)), which no gamma linearises.
            listed.insert("alpha_only");
            const bool alphaOnly = get<rdl::Bool>(m, "alpha_only");
            alphaLookup = alphaOnly;
            bool udimEightBit = false;
            udim = udimTiles(m, udimEightBit);
            h.udim = unsigned(udim.size());
            h.texture = udim.empty() ? textureIndex(m) : shade::MapNoTexture;
            alphaLookup = false;
            h.wrap = get<rdl::Bool>(m, "wrap_around") ? 1 : 0;
            const int gamma = get<rdl::Int>(m, "gamma");
            need(gamma >= 0 && gamma <= 2, m.getName() + ": invalid gamma mode");
            // As the validated ImageMap subset: linearise 8-bit data when gamma is on or auto.
            h.gamma = (gamma != 0 && h.texture != shade::MapNoTexture && (*textures.textures)[h.texture]->eightBit) ? 1 : 0;
            if (h.udim) h.gamma = (gamma != 0 && udimEightBit) ? 1 : 0;
            if (alphaOnly) h.gamma = 0;
        }
        if (c->kind == shade::MapModoTexture) listed.insert("uv_name");  // its keys after the attributes, below
        if (primvarReader) listed = {"varname", "warn_when_unavailable", "fallback"};
        if (c->kind == shade::MapUsdUVTexture) {
            for (const char* n : {"file", "wrapS", "wrapT", "sourceColorSpace"}) listed.insert(n);
            h.texture = textureIndex(m, "file");
            const int ws = get<rdl::Int>(m, "wrapS"), wt = get<rdl::Int>(m, "wrapT");
            auto wrap = [](int w) { return w == 4 ? 2 : w; };  // useMetadata: periodic
            need(wrap(ws) == wrap(wt) && (wrap(ws) == 1 || wrap(ws) == 2), m.getName() + ": only matching clamp or repeat wraps");
            h.wrap = wrap(ws) == 2 ? 1 : 0;
            const int space = get<rdl::Int>(m, "sourceColorSpace");
            need(space == 0 || space == 1 || space == 3, m.getName() + ": invalid source colour space");
            need(get<rdl::Int>(m, "output_mode") != 4, m.getName() + ": alpha output unsupported");
            h.gamma = (space != 0 && h.texture != shade::MapNoTexture && (*textures.textures)[h.texture]->eightBit) ? 1 : 0;
        }
        if (c->kind == shade::MapImageNormal) {
            listed.insert("tangent_space_normal_texture");
            h.texture = textureIndex(m, "tangent_space_normal_texture");
            h.wrap = get<rdl::Bool>(m, "wrap_around") ? 1 : 0;
            h.gamma = 0;  // normals are read linear (TEXTURE_GAMMA_OFF)
        }
        for (auto i = m.getSceneClass().beginAttributes(); i != m.getSceneClass().endAttributes(); ++i)
            if ((*i)->isBindable() && m.getBinding(**i)) bound.insert((*i)->getName());
        if (c->kind == shade::MapRamp)
            for (const char* n : {"positions", "colors", "interpolations", "color_space", "object", "camera"}) listed.insert(n);
        if (c->kind == shade::MapGradient) listed.insert("object");
        if (c->kind == shade::MapAttribute) listed.insert("primitive_attribute_name");
        if (c->kind == shade::MapNormalToRgb) listed.insert("input");
        const bool worley = c->kind == shade::MapWorley2 || c->kind == shade::MapWorley3;
        if (c->kind == shade::MapNoise || worley) {
            listed.insert("object_space");
            listed.insert("camera");
            const int space = get<rdl::Int>(m, "space");
            need(space == 0 || space == 1 || space == 2 || space == 4 || space == 6 || space == 7,
                 m.getName() + ": screen, reference and hair spaces are not shaded yet");
            need(space != 1 || !get<rdl::SceneObject*>(m, "camera"), m.getName() + ": a custom camera space is not shaded yet");
            need(!bound.count("translation") && !bound.count("rotation") && !bound.count("scale"),
                 m.getName() + ": bound noise transforms are not shaded yet");
            if (!worley)
                for (const char* n : {"noise_type", "distortion_noise_type"})
                    need(get<rdl::Int>(m, n) == 0 || get<rdl::Int>(m, n) == 1, m.getName() + ": invalid noise type");
            else
                need(get<rdl::Int>(m, "output_mode") >= 0 && get<rdl::Int>(m, "output_mode") <= 4 &&
                         get<rdl::Int>(m, "distance_method") >= 0 && get<rdl::Int>(m, "distance_method") <= 5 &&
                         get<rdl::Int>(m, "cell_id") >= 0 && get<rdl::Int>(m, "cell_id") <= 3,
                     m.getName() + ": invalid Worley setting");
        }
        if (c->kind == shade::MapSwitchNormal)
            for (int i = 0; i < 64; ++i) listed.insert("input" + std::to_string(i));
        if (c->kind == shade::MapCombineNormal) listed.insert({"input_1", "input_2"});
        if (c->kind == shade::MapProjectTriplanar || c->kind == shade::MapProjectTriplanarNormal || c->kind == shade::MapProjectCamera ||
            c->kind == shade::MapProjectPlanarNormal || c->kind == shade::MapProjectCameraNormal)
            for (auto i = m.getSceneClass().beginAttributes(); i != m.getSceneClass().endAttributes(); ++i) listed.insert((*i)->getName());
        const bool projection =
            c->kind == shade::MapProjectPlanar || c->kind == shade::MapProjectSpherical || c->kind == shade::MapProjectCylindrical;
        if (projection) {
            for (const char* n : {"projection_mode", "projector", "projection_matrix", "TRS_order", "rotation_order", "translate", "rotate", "scale"})
                listed.insert(n);
            need(!get<rdl::Bool>(m, "use_reference_space"), m.getName() + ": reference space (ref_P) is not shaded yet");
        }
        moreListed(c->kind, m, listed);
        attributes(m, listed, bound);
        std::vector<float> params(floats, 0.0f);
        if (primvarReader) {
            // UsdPrimvarReader_<type>: varname's special values, else a primitive attribute of the type.
            const std::string type = cls.substr(std::strlen("UsdPrimvarReader_")), name = get<rdl::String>(m, "varname");
            unsigned prim = shade::PrimFloat, components = 1;
            int mode = 0;
            if (type == "float") {
                prim = shade::PrimFloat;
            } else if (type == "float2") {
                prim = shade::PrimVec2f;
                components = 2;
                if (name == "st") mode = 2;
                need(name != "surface_st" && name != "closest_surface_uv", m.getName() + ": hair coordinates unsupported");
            } else if (type == "float3" || type == "point" || type == "vector") {
                prim = shade::PrimVec3f;
                components = 3;
                if (name == "P" && type != "vector") mode = 1;
                if (name == "dpds" && type != "point") mode = 5;
                if (name == "dpdt" && type != "point") mode = 6;
                need(name != "dnds" && name != "dndt" && name != "surface_P", m.getName() + ": " + name + " is not shaded yet");
            } else if (type == "int") {
                prim = shade::PrimInt;
                need(name != "id", m.getName() + ": ids are not shaded yet");
            } else {
                throw Unsupported(m.getName() + ": map class " + cls + " is not shaded yet");
            }
            params[0] = float(mode);
            const unsigned key = shade::primKey(name.c_str(), prim);
            std::memcpy(&params[1], &key, sizeof key);
            params[2] = float(components);
            if (components == 1) {
                params[3] = type == "int" ? float(get<rdl::Int>(m, "fallback")) : get<rdl::Float>(m, "fallback");
            } else if (components == 2) {
                const auto f = get<rdl::Vec2f>(m, "fallback");
                params[3] = f.x;
                params[4] = f.y;
            } else {
                const auto f = get<rdl::Vec3f>(m, "fallback");
                params[3] = f.x;
                params[4] = f.y;
                params[5] = f.z;
            }
        } else if (c->kind == shade::MapNormalToRgb) {
            const rdl::SceneObject* input = get<rdl::SceneObject*>(m, "input");
            need(!input || input->isA<rdl::NormalMap>(), m.getName() + ": its input is not a normal map");
            params[0] = input ? float(int(reg(*input))) : -1.0f;
        } else if (c->kind != shade::MapHairColumn && c->kind != shade::MapList) {
            for (size_t k = 0; k < c->attributes.size(); ++k) writeAttribute(m, c->attributes[k], params.data() + offsets[k].second);
        }
        morePrepared(c->kind, m, h, params, bound);
        // Prepared data after the attributes: a ramp's control, then the 'object' space's matrix.
        if (c->kind == shade::MapRamp) {
            const auto& positions = get<rdl::FloatVector>(m, "positions");
            const auto& colors = get<rdl::RgbVector>(m, "colors");
            const auto& interps = get<rdl::IntVector>(m, "interpolations");
            need(colors.size() >= positions.size() && interps.size() >= positions.size(), m.getName() + ": invalid ramp");
            const int type = get<rdl::Int>(m, "ramp_type"), colorSpace = get<rdl::Int>(m, "color_space");
            need(type >= 0 && type <= 8 && colorSpace >= 0 && colorSpace <= 2 && get<rdl::Int>(m, "wrap_type") >= 0 &&
                     get<rdl::Int>(m, "wrap_type") <= 1, m.getName() + ": invalid ramp setting");
            // ColorRampControl::init: at most RAMP_MAX_POINTS, sorted by position.
            std::vector<std::tuple<float, shade::Color, unsigned>> entries;
            for (size_t i = 0; i < positions.size() && i < shade::RampMaxPoints; ++i) {
                need(std::isfinite(positions[i]) && interps[i] >= 0 && interps[i] <= 6, m.getName() + ": invalid ramp point");
                entries.emplace_back(positions[i], shade::Color{colors[i].r, colors[i].g, colors[i].b}, unsigned(interps[i]));
            }
            std::stable_sort(entries.begin(), entries.end(), [](auto& x, auto& y) { return std::get<0>(x) < std::get<0>(y); });
            float in[shade::RampMaxPoints];
            shade::Color outs[shade::RampMaxPoints];
            unsigned modes[shade::RampMaxPoints];
            for (size_t i = 0; i < entries.size(); ++i) {
                in[i] = std::get<0>(entries[i]);
                outs[i] = std::get<1>(entries[i]);
                modes[i] = std::get<2>(entries[i]);
            }
            shade::ColorRamp ramp{};
            shade::rampInit(ramp, unsigned(entries.size()), in, outs, modes, unsigned(colorSpace), false);
            const float* rf = reinterpret_cast<const float*>(&ramp);
            params.insert(params.end(), rf, rf + sizeof(ramp) / sizeof(float));
        }
        if (c->kind == shade::MapLOD) {
            need(get<rdl::Int>(m, "mode") == 1, m.getName() + ": feature-width LOD needs ray differentials (not shaded yet)");
            need(get<rdl::Float>(m, "start") > shade::Epsilon && get<rdl::Float>(m, "stop") - get<rdl::Float>(m, "start") > shade::Epsilon,
                 m.getName() + ": invalid LOD range");
        }
        if (c->kind == shade::MapModoTexture) {
            // A named UV set (uv_name, modes 12 and 13): the keys of its primitive attribute and of modo_primary_uv
            // after the attributes, and whether it is named (ModoTextureMap::update's optional attributes, which a
            // mesh without them leaves the map's neutral result).
            const std::string uv = get<rdl::String>(m, "uv_name");
            const unsigned keys[2] = {uv.empty() ? 0u : shade::primKey(uv.c_str(), shade::PrimVec2f),
                                      shade::primKey("modo_primary_uv", shade::PrimVec2f)};
            float f[3];
            std::memcpy(&f[0], &keys[0], sizeof f[0]);
            std::memcpy(&f[1], &keys[1], sizeof f[1]);
            f[2] = uv.empty() ? 0.0f : 1.0f;
            params.insert(params.end(), f, f + 3);
        }
        if (c->kind == shade::MapAttribute) {
            const int type = get<rdl::Int>(m, "map_type"), primType = get<rdl::Int>(m, "primitive_attribute_type");
            need(type == 0 || (type >= 1 && type <= 6) || type == 9 || type == 20,
                 m.getName() + ": this attribute map type is not shaded yet");
            need(primType >= 0 && primType <= 4, m.getName() + ": invalid primitive attribute type");
            const unsigned key = shade::primKey(get<rdl::String>(m, "primitive_attribute_name").c_str(), unsigned(primType));
            float f;
            std::memcpy(&f, &key, sizeof f);
            params.push_back(f);
        }
        if (c->kind == shade::MapGradient || c->kind == shade::MapRamp) {
            const int space = get<rdl::Int>(m, "space");
            const bool coords = c->kind == shade::MapRamp && get<rdl::Int>(m, "texture_coordinates") == 1;
            need(coords || (space >= 0 && space <= 6 && space != shade::SpaceScreen && space != shade::SpaceReference),
                 m.getName() + ": screen and reference spaces are not shaded yet");
            if (c->kind == shade::MapRamp) need(!get<rdl::SceneObject*>(m, "camera"), m.getName() + ": a custom camera space is not shaded yet");
            const rdl::SceneObject* object = get<rdl::SceneObject*>(m, "object");
            if (object && space == shade::SpaceObject && !coords) {
                need(object->isA<rdl::Node>(), m.getName() + ": its object is not a node");
                const auto w = get<rdl::Mat4d>(*object, "node_xform").inverse();
                // MoonRay's matrices are row vectors: p' = p * M. As 3x4 rows for xformPoint.
                const float rows[12] = {float(w[0][0]), float(w[1][0]), float(w[2][0]), float(w[3][0]),
                                        float(w[0][1]), float(w[1][1]), float(w[2][1]), float(w[3][1]),
                                        float(w[0][2]), float(w[1][2]), float(w[2][2]), float(w[3][2])};
                params.insert(params.end(), rows, rows + 12);
                h.object = 1;
            }
        }
        for (unsigned tile : udim) params.push_back(tile == shade::MapNoTexture ? -1.0f : float(tile));  // a missing tile -1
        if (projection) {
            // The projector's world-to-object (as 3x4 rows for xformPoint), then its normal transform: a
            // world normal times the object-to-world's transpose (row vectors), so rows of its upper 3x3.
            rdl::Mat4d M;
            float rows[24] = {};
            if (projectorMatrix(m, M)) {
                const auto w = M.inverse();
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) rows[4 * i + j] = float(w[j][i]);
                    for (int j = 0; j < 3; ++j) rows[12 + 4 * i + j] = float(M[i][j]);
                }
                for (float v : rows) need(std::isfinite(v), m.getName() + ": its projector cannot be inverted");
                h.object = 1;
            }
            params.insert(params.end(), rows, rows + 24);
        }
        if (c->kind == shade::MapProjectTriplanar || c->kind == shade::MapProjectTriplanarNormal) params = triplanarParams(m, h);
        // SwitchNormalMap's 64 inputs and CombineNormalMap's two: the registers of the normal maps they name (-1: none).
        if (c->kind == shade::MapSwitchNormal || c->kind == shade::MapCombineNormal) {
            const int n = c->kind == shade::MapSwitchNormal ? 64 : 2;
            std::vector<float> regs;
            for (int i = 0; i < n; ++i) {
                const std::string name = c->kind == shade::MapSwitchNormal ? "input" + std::to_string(i) : "input_" + std::to_string(i + 1);
                const rdl::SceneObject* input = get<rdl::SceneObject*>(m, name);
                float r = -1.0f;
                if (input && input->isA<rdl::NormalMap>()) r = float(reg(*input));
                else need(!input || c->kind == shade::MapCombineNormal, m.getName() + ": " + name + " is not a normal map");
                regs.push_back(r);
            }
            // CombineNormalMap's verifyInputs: both must be normal maps, else neither is read.
            if (c->kind == shade::MapCombineNormal && (regs[0] < 0.0f || regs[1] < 0.0f)) regs[0] = regs[1] = -1.0f;
            params.insert(params.end(), regs.begin(), regs.end());
            need(get<rdl::Float>(m, c->kind == shade::MapSwitchNormal ? "choice" : "normal_map_1_dial") >= 0.0f ||
                     c->kind == shade::MapCombineNormal || bound.count("choice"),
                 m.getName() + ": negative choice");
        }
        if (c->kind == shade::MapProjectCamera || c->kind == shade::MapProjectCameraNormal) params = projectCameraParams(m, h);
        if (c->kind == shade::MapUVTransform) {
            // UVTransformMap::update's rotation: about the axis in world, object and reference space, else about the
            // centre in the plane. Screen and reference spaces are not shaded yet.
            namespace sm = scene_rdl2::math;
            const int space = get<rdl::Int>(m, "space");
            need(space == 0 || space == 1 || space == 2 || space == 4 || space == 6, m.getName() + ": screen and reference spaces are not shaded yet");
            const float theta = sm::deg2rad(get<rdl::Float>(m, "rotation_angle"));
            sm::Xform3f x(sm::one);
            if (!sm::isZero(theta)) {
                if (space == 2 || space == 4) {
                    const auto a = get<rdl::Vec3f>(m, "rotation_axis");
                    x = x.rotate(sm::Vec3f(a.x, a.y, a.z), theta);
                } else {
                    const auto c = get<rdl::Vec2f>(m, "rotation_center");
                    const sm::Vec3f c3(c.x, c.y, 0.f);
                    sm::Xform3f t1(sm::one), r(sm::one), t2(sm::one);
                    t1 = t1.translate(c3);
                    r = r.rotate(sm::Vec3f(0.f, 0.f, 1.f), theta);
                    t2 = t2.translate(-c3);
                    x = t2 * (r * t1);
                }
            }
            const float rows[12] = {x.l.vx.x, x.l.vx.y, x.l.vx.z, x.l.vy.x, x.l.vy.y, x.l.vy.z, x.l.vz.x, x.l.vz.y, x.l.vz.z, x.p.x, x.p.y, x.p.z};
            for (float v : rows) need(std::isfinite(v), m.getName() + ": invalid rotation");
            params.insert(params.end(), rows, rows + 12);
        }
        if (c->kind == shade::MapProjectPlanarNormal) {
            // ProjectPlanarNormalMap::update: its texture linear, wrapped or clamped; transition width 0.5; P and N only.
            need(!get<rdl::Bool>(m, "use_reference_space"), m.getName() + ": reference space (ref_P) is not shaded yet");
            const int encoding = get<rdl::Int>(m, "normal_encoding");
            need(encoding == 0 || encoding == 1, m.getName() + ": invalid normal encoding");
            h.texture = textureIndex(m);
            need(h.texture != shade::MapNoTexture, m.getName() + ": a planar normal map without a texture");
            h.wrap = get<rdl::Bool>(m, "wrap_around") ? 1 : 0;
            h.gamma = 0;
            shade::MapProjectPlanarNormalParams q{};
            float twp = 1.0f - 0.5f;
            twp = twp * twp * twp * twp * twp;
            twp = 0.025f + (twp * 0.975f);
            q.transitionWidthPower = twp * 100.0f;
            q.normalEncoding = float(encoding);
            q.wrap = float(h.wrap);
            rdl::Mat4d M;
            if (projectorMatrix(m, M)) {
                const auto w = M.inverse();
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) q.toObject[4 * i + j] = float(w[j][i]);
                    for (int j = 0; j < 3; ++j) q.normalToObject[4 * i + j] = float(M[i][j]);
                }
                for (float v : q.toObject) need(std::isfinite(v), m.getName() + ": its projector cannot be inverted");
                h.object = 1;
            }
            const float* f = reinterpret_cast<const float*>(&q);
            params.assign(f, f + sizeof(q) / sizeof(float));
        }
        if (worley) {
            namespace sm = scene_rdl2::math;
            // NoiseWorleyMap::update: the transform, the object's world-to-object, the Worley tables.
            const auto t = get<rdl::Vec3f>(m, "translation"), r = get<rdl::Vec3f>(m, "rotation"), s = get<rdl::Vec3f>(m, "scale");
            const sm::Xform3f x = noiseCompose(sm::Vec3f(t.x, t.y, t.z), sm::Vec3f(r.x, r.y, r.z) * sm::sPi / 180.f,
                                               sm::Vec3f(s.x, s.y, s.z), get<rdl::Int>(m, "transformation_order"),
                                               get<rdl::Int>(m, "rotation_order"));
            const float rows[12] = {x.l.vx.x, x.l.vy.x, x.l.vz.x, x.p.x, x.l.vx.y, x.l.vy.y, x.l.vz.y, x.p.y,
                                    x.l.vx.z, x.l.vy.z, x.l.vz.z, x.p.z};
            params.insert(params.end(), rows, rows + 12);
            float object[12] = {};
            const rdl::SceneObject* o = get<rdl::SceneObject*>(m, "object_space");
            if (o && get<rdl::Int>(m, "space") == int(shade::SpaceObject)) {
                need(o->isA<rdl::Node>(), m.getName() + ": its object is not a node");
                const auto w = get<rdl::Mat4d>(*o, "node_xform").inverse();
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j) object[4 * i + j] = float(w[j][i]);
                h.object = 1;
            }
            params.insert(params.end(), object, object + 12);
            float low[10], high[10];
            worleyProbabilities(low, high);
            params.insert(params.end(), low, low + 10);
            params.insert(params.end(), high, high + 10);
            params.push_back(float(params.size() + 1));  // the tables follow this word
            worleyTables(params, get<rdl::Int>(m, "seed"));
            need(params.size() < (1u << 24), m.getName() + ": Worley tables too large");
        }
        if (c->kind == shade::MapNoise) {
            namespace sm = scene_rdl2::math;
            // NoiseMap_v2::update: the transform (rotation in radians), the object's world-to-object, the noises.
            const auto t = get<rdl::Vec3f>(m, "translation"), r = get<rdl::Vec3f>(m, "rotation"), s = get<rdl::Vec3f>(m, "scale");
            const sm::Xform3f x = noiseCompose(sm::Vec3f(t.x, t.y, t.z), sm::Vec3f(r.x, r.y, r.z) * sm::sPi / 180.f,
                                               sm::Vec3f(s.x, s.y, s.z), get<rdl::Int>(m, "transformation_order"),
                                               get<rdl::Int>(m, "rotation_order"));
            const float rows[12] = {x.l.vx.x, x.l.vy.x, x.l.vz.x, x.p.x, x.l.vx.y, x.l.vy.y, x.l.vz.y, x.p.y,
                                    x.l.vx.z, x.l.vy.z, x.l.vz.z, x.p.z};
            params.insert(params.end(), rows, rows + 12);
            float object[12] = {};
            const rdl::SceneObject* o = get<rdl::SceneObject*>(m, "object_space");
            if (o && get<rdl::Int>(m, "space") == int(shade::SpaceObject)) {
                need(o->isA<rdl::Node>(), m.getName() + ": its object is not a node");
                const auto w = get<rdl::Mat4d>(*o, "node_xform").inverse();
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j) object[4 * i + j] = float(w[j][i]);
                h.object = 1;
            }
            params.insert(params.end(), object, object + 12);
            const size_t offsets = params.size();
            params.insert(params.end(), 4, 0.0f);
            const int seed = get<rdl::Int>(m, "seed");
            const bool fourD = get<rdl::Bool>(m, "use_4D_noise"), perlin = get<rdl::Int>(m, "noise_type") == 0;
            auto add = [&](int slot, int noiseSeed, bool isPerlin) {
                params[offsets + slot] = float(params.size());
                noiseTables(params, noiseSeed, isPerlin, fourD);
            };
            add(0, seed, perlin);
            // As NoiseMap_v2: the green and blue noises take seeds + 2 and + 3.
            if (get<rdl::Bool>(m, "color")) {
                add(1, seed + 2, perlin);
                add(2, seed + 3, perlin);
            }
            if (get<rdl::Float>(m, "distortion") != 0.0f || bound.count("distortion"))
                add(3, seed, get<rdl::Int>(m, "distortion_noise_type") == 0);
            need(params.size() < (1u << 24), m.getName() + ": noise tables too large");
        }
        if (c->kind == shade::MapSwitchColor || c->kind == shade::MapSwitchFloat)
            need(get<rdl::Float>(m, "choice") >= 0 || bound.count("choice"), m.getName() + ": negative choice");
        std::vector<shade::MapBinding> bindings;
        if (primvarReader) bindingsOf(m, {{"fallback", 3}}, bindings);
        else if (c->kind != shade::MapNormalToRgb) bindingsOf(m, offsets, bindings);
        h.paramCount = unsigned(params.size());
        h.bindingCount = unsigned(bindings.size());
        need(nodes.size() < shade::MapMaxRegisters, m.getName() + ": too many maps in one material");
        h.reg = unsigned(nodes.size());
        const unsigned offset = unsigned(pool.size());
        const float* hf = reinterpret_cast<const float*>(&h);
        pool.insert(pool.end(), hf, hf + sizeof(h) / sizeof(float));
        pool.insert(pool.end(), params.begin(), params.end());
        for (auto& b : bindings) {
            const float* bf = reinterpret_cast<const float*>(&b);
            pool.insert(pool.end(), bf, bf + sizeof(b) / sizeof(float));
        }
        nodes.push_back(offset);
        visiting.erase(&m);
        registers[&m] = h.reg;
        return h.reg;
    }
    // ProjectTriplanarMap_v2's prepared data (projection::updateTriplanarMap, TriplanarTexture's composeTransform2D
    // with its random seed, calculateTransitionWidthPower, fillTriplanarTextureIndices). Positions from P and N only.
    std::vector<float> triplanarParams(const rdl::SceneObject& m, shade::MapNodeHeader& h) {
        namespace sm = scene_rdl2::math;
        need(get<rdl::Int>(m, "input_position_source") == 0, m.getName() + ": ref_P and input positions are not shaded yet");
        const bool normalMap = m.getSceneClass().getName() == "ProjectTriplanarNormalMap_v2";
        // A normal map reads its textures linear (TEXTURE_GAMMA_OFF), as ImageNormalMap does.
        const int count = get<rdl::Int>(m, "number_of_textures"), gamma = normalMap ? 0 : get<rdl::Int>(m, "gamma");
        need(count == 1 || count == 3 || count == 6, m.getName() + ": invalid number of textures");
        need(gamma >= 0 && gamma <= 2, m.getName() + ": invalid gamma mode");
        shade::MapTriplanarParams q{};
        if (normalMap) {
            const int encoding = get<rdl::Int>(m, "normal_encoding");
            need(encoding == 0 || encoding == 1, m.getName() + ": invalid normal encoding");
            q.normalEncoding = float(encoding);
        }
        static const char* faces[6] = {"positive_x", "positive_y", "positive_z", "negative_x", "negative_y", "negative_z"};
        scene_rdl2::util::Random rand(get<rdl::Int>(m, "random_seed"));
        const bool randomFlip = get<rdl::Bool>(m, "randomize_flip"), randomOffset = get<rdl::Bool>(m, "randomize_offset"),
                   randomRotation = get<rdl::Bool>(m, "randomize_rotation");
        for (int i = 0; i < count; ++i) {
            const std::string f = faces[i];
            const std::string path = get<rdl::String>(m, f + "_texture");
            need(!path.empty(), m.getName() + ": a triplanar texture without a file");
            need(path.find("<UDIM>") == std::string::npos && path.find("<udim>") == std::string::npos,
                 m.getName() + ": UDIM textures are read by ImageMap only");
            const unsigned index = fileIndex(m, path);
            shade::MapTriplanarTexture& t = q.textures[i];
            t.texture = float(index);
            t.active = get<rdl::Bool>(m, f + "_active") ? 1.0f : 0.0f;
            t.wrap = get<rdl::Bool>(m, f + "_wrap_around") ? 1.0f : 0.0f;
            // As ImageMap's validated subset: linearise 8-bit data when gamma is on or auto.
            t.gamma = (gamma != 0 && (*textures.textures)[index]->eightBit) ? 1.0f : 0.0f;
            // composeTransform2D (its swap adds 90 to the angle in radians, as MoonRay's does).
            const auto sc = get<rdl::Vec2f>(m, f + "_scale");
            const auto of = get<rdl::Vec2f>(m, f + "_offset");
            const auto rc = get<rdl::Vec2f>(m, f + "_rotation_center");
            sm::Vec2f scale(sc.x, sc.y), offset(of.x, of.y);
            scale.x = 1.0f / sm::max(scale.x, sm::sEpsilon) * (get<rdl::Bool>(m, f + "_invert_s") ? -1.0f : 1.0f);
            scale.y = 1.0f / sm::max(scale.y, sm::sEpsilon) * (get<rdl::Bool>(m, f + "_invert_t") ? -1.0f : 1.0f);
            if (randomFlip) {
                scale.x *= ((rand.getNextFloat() > 0.5f) ? -1 : 1);
                scale.y *= ((rand.getNextFloat() > 0.5f) ? -1 : 1);
            }
            offset.x *= -1.0f;
            if (randomOffset) {
                offset.x += (rand.getNextFloat() - 0.5f);
                offset.y += (rand.getNextFloat() - 0.5f);
            }
            float rotation = -1.0f * sm::deg2rad(get<rdl::Float>(m, f + "_rotation")) + (get<rdl::Bool>(m, f + "_swap_st") ? 90.0f : 0.0f);
            if (randomRotation) rotation += rand.getNextFloat() * sm::sTwoPi - sm::sPi;
            sm::Xform3f x(sm::one);
            x = x * sm::Xform3f::translate(sm::Vec3f(-rc.x, -rc.y, 0.0f));
            x = x * sm::Xform3f::rotate(sm::Vec3f(0.0f, 0.0f, 1.0f), rotation);
            x = x * sm::Xform3f::translate(sm::Vec3f(rc.x, rc.y, 0.0f));
            x = x * sm::Xform3f::scale(sm::Vec3f(scale.x, scale.y, 0.0f));
            x = x * sm::Xform3f::translate(sm::Vec3f(offset.x, offset.y, 0.0f));
            const float rows[12] = {x.l.vx.x, x.l.vx.y, x.l.vx.z, x.l.vy.x, x.l.vy.y, x.l.vy.z,
                                    x.l.vz.x, x.l.vz.y, x.l.vz.z, x.p.x, x.p.y, x.p.z};
            for (int k = 0; k < 12; ++k) {
                need(std::isfinite(rows[k]), m.getName() + ": invalid texture transform");
                t.xform[k] = rows[k];
            }
        }
        for (int f = 0; f < 6; ++f) q.indices[f] = float(count == 1 ? 0 : count == 3 ? f % 3 : f);
        // calculateTransitionWidthPower
        float twp = 1.0f - sm::clamp(get<rdl::Float>(m, "transition_width"), 0.0f, 1.0f);
        twp = twp * twp * twp * twp * twp;
        twp = 0.025f + (twp * 0.975f);
        q.transitionWidthPower = twp * 100.0f;
        rdl::Mat4d M;
        if (projectorMatrix(m, M)) {
            const auto w = M.inverse();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) q.toObject[4 * i + j] = float(w[j][i]);
                for (int j = 0; j < 3; ++j) q.normalToObject[4 * i + j] = float(M[i][j]);
            }
            for (float v : q.toObject) need(std::isfinite(v), m.getName() + ": its projector cannot be inverted");
            h.object = 1;
        }
        const float* p = reinterpret_cast<const float*>(&q);
        return std::vector<float>(p, p + sizeof(q) / sizeof(float));
    }
    // ProjectCameraMap_v2's prepared data (its update and shading::Xform's render-to-screen): the window from the
    // texture's aspect ratio (its pixels' and their own) or a custom one, the projector's perspective or orthographic
    // projection (computeProjectionMatrix, no stereo), its texture clamped. P and N only (no reference space).
    std::vector<float> projectCameraParams(const rdl::SceneObject& m, shade::MapNodeHeader& h) {
        namespace sm = scene_rdl2::math;
        need(!get<rdl::Bool>(m, "use_reference_space"), m.getName() + ": reference space (ref_P) is not shaded yet");
        // ProjectCameraNormalMap: always black outside, its texture linear, its normal encoding.
        const bool normalMap = m.getSceneClass().getName() == "ProjectCameraNormalMap";
        need(normalMap || (!get<rdl::Bool>(m, "alpha_only") && !get<rdl::Bool>(m, "unpremultiply")),
             m.getName() + ": alpha lookups are not shaded yet");
        const int gamma = normalMap ? 0 : get<rdl::Int>(m, "gamma");
        need(gamma >= 0 && gamma <= 2, m.getName() + ": invalid gamma mode");
        h.texture = textureIndex(m);
        need(h.texture != shade::MapNoTexture, m.getName() + ": a camera projection without a texture");
        const auto& file = *(*textures.textures)[h.texture];
        h.gamma = (gamma != 0 && file.eightBit) ? 1 : 0;
        h.wrap = 0;
        shade::MapProjectCameraParams q{};
        q.projectOnBackFaces = get<rdl::Bool>(m, "project_on_back_faces") ? 1.0f : 0.0f;
        q.blackOutsideProjection = normalMap || get<rdl::Bool>(m, "black_outside_projection") ? 1.0f : 0.0f;
        if (normalMap) {
            const int encoding = get<rdl::Int>(m, "normal_encoding");
            need(encoding == 0 || encoding == 1, m.getName() + ": invalid normal encoding");
            q.normalEncoding = float(encoding);
        }
        const rdl::SceneObject* projector = get<rdl::SceneObject*>(m, "projector");
        if (projector) {
            const std::string cls = projector->getSceneClass().getName();
            need(cls == "PerspectiveCamera" || cls == "OrthographicCamera", m.getName() + ": its projector is not a perspective or orthographic camera");
            float aspect;
            if (get<rdl::Int>(m, "aspect_ratio_source") == 0) {
                OIIO::ImageSpec spec;
                need(file.system->get_imagespec(file.name, spec), m.getName() + ": its texture's size is unknown");
                aspect = float(spec.width) / float(spec.height) * spec.get_float_attribute("PixelAspectRatio", 1.0f);
            } else {
                aspect = get<rdl::Float>(m, "custom_aspect_ratio");
            }
            float window[4];
            if (aspect >= 1.0f) {
                aspect = 1.0f / aspect;
                window[0] = -1.0f; window[1] = aspect; window[2] = 1.0f; window[3] = -aspect;
            } else {
                window[0] = -aspect; window[1] = 1.0f; window[2] = aspect; window[3] = -1.0f;
            }
            // computeProjectionMatrix (interocular offset 0)
            const bool perspective = cls == "PerspectiveCamera";
            const float minimum = perspective ? 0.00001f : 0.01f;
            const float focal = perspective ? sm::max(get<rdl::Float>(*projector, "focal"), 0.00001f) : 1.0f;
            const float halfFilmWidth = sm::max(get<rdl::Float>(*projector, "film_width_aperture"), minimum) * 0.5f;
            const float near = sm::max(get<rdl::Float>(*projector, "near"), minimum), far = sm::max(get<rdl::Float>(*projector, "far"), minimum);
            const float offsetX = get<rdl::Float>(*projector, "horizontal_film_offset"), offsetY = get<rdl::Float>(*projector, "vertical_film_offset");
            float par = get<rdl::Float>(*projector, "pixel_aspect_ratio");
            if (sm::isEqual(par, 0.0f)) par = 1.0f;
            const float factor = perspective ? near / focal : 1.0f;
            const float w0 = (window[0] * halfFilmWidth + offsetX) * factor, w1 = (window[1] * halfFilmWidth + offsetY) * factor / par,
                        w2 = (window[2] * halfFilmWidth + offsetX) * factor, w3 = (window[3] * halfFilmWidth + offsetY) * factor / par;
            sm::Mat4f c2s(sm::zero);
            if (perspective) {
                c2s[0][0] = (2.0f * near) / (w2 - w0);
                c2s[1][1] = (2.0f * near) / (w3 - w1);
                c2s[2][0] = (w2 + w0) / (w2 - w0);
                c2s[2][1] = (w3 + w1) / (w3 - w1);
                c2s[2][2] = -(far + near) / (far - near);
                c2s[2][3] = -1.0f;
                c2s[3][2] = -(2.0f * far * near) / (far - near);
            } else {
                c2s[0][0] = 2.0f / (w2 - w0);
                c2s[1][1] = 2.0f / (w3 - w1);
                c2s[2][2] = -2.0f / (far - near);
                c2s[3][0] = -(w2 + w0) / (w2 - w0);
                c2s[3][1] = -(w3 + w1) / (w3 - w1);
                c2s[3][2] = -(far + near) / (far - near);
                c2s[3][3] = 1.0f;
            }
            const rdl::Mat4d c2w = get<rdl::Mat4d>(*projector, "node_xform");
            const sm::Mat4f w2s = sm::toFloat(c2w.inverse()) * c2s;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) q.worldToScreen[4 * i + j] = w2s[i][j];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) q.normalToCamera[4 * i + j] = float(c2w[i][j]);
            for (float v : q.worldToScreen) need(std::isfinite(v), m.getName() + ": its projector cannot be inverted");
            h.object = 1;
        }
        const float* p = reinterpret_cast<const float*>(&q);
        return std::vector<float>(p, p + sizeof(q) / sizeof(float));
    }
    // The program: {count, node offsets..., required count, required attribute keys...}; NoMapProgram with no maps.
    unsigned finish() {
        if (nodes.empty()) return shade::NoMapProgram;
        const unsigned offset = unsigned(pool.size());
        auto word = [&](unsigned v) {
            float f;
            std::memcpy(&f, &v, sizeof f);
            pool.push_back(f);
        };
        word(unsigned(nodes.size()));
        for (unsigned n : nodes) word(n);
        std::sort(required.begin(), required.end());
        required.erase(std::unique(required.begin(), required.end()), required.end());
        word(unsigned(required.size()));
        for (unsigned k : required) word(k);
        // The bump maps (ModoTextureMap's bump mode, whose heights are its inputs' at shifted states): each one's node
        // and the nodes its inputs need, in program order. One whose inputs need another bump map is refused.
        auto isBump = [&](unsigned i) {
            const auto& h = *reinterpret_cast<const shade::MapNodeHeader*>(pool.data() + nodes[i]);
            return shade::modoNeedsBump(h, pool.data() + nodes[i] + sizeof(shade::MapNodeHeader) / sizeof(float));
        };
        std::vector<unsigned> bumps;
        for (unsigned i = 0; i < nodes.size(); ++i)
            if (isBump(i)) bumps.push_back(i);
        std::vector<std::vector<unsigned>> lists;
        for (unsigned b : bumps) {
            std::set<unsigned> closure;
            std::vector<unsigned> todo = b < nodeDeps.size() ? nodeDeps[b] : std::vector<unsigned>{};
            while (!todo.empty()) {
                const unsigned r = todo.back();
                todo.pop_back();
                if (!closure.insert(r).second) continue;
                if (r < nodeDeps.size()) todo.insert(todo.end(), nodeDeps[r].begin(), nodeDeps[r].end());
            }
            for (unsigned r : closure) need(r < nodes.size() && !isBump(r), "A bump map whose height reads another bump map is not rendered yet");
            lists.emplace_back(closure.begin(), closure.end());
        }
        word(unsigned(bumps.size()));
        for (size_t k = 0; k < bumps.size(); ++k) {
            word(bumps[k]);
            word(unsigned(lists[k].size()));
            for (unsigned r : lists[k]) word(r);
        }
        return offset;
    }
};
}

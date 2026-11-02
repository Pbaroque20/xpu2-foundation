// XPU2's scene indexer, its classification (INDEXER.md, 1): each RDL object's device class. Fallback: XPU2 does not
// render it (MoonRay XPU takes the scene); Cpu: the CPU lane alone (the GPU programs lack it, so its samples go to the
// CPU); Both: either device. Beside it, the GPU feature families the object needs (the GPU program is built from the
// families of the scene's Both objects) and its relative shading cost on each device. The plan builder (cpu_scene.h)
// stays the final check of every attribute: an object classified Both can still be refused there.
#pragma once
#include "routing_index.h"
#include "dwa_admission.h"
#include "map_admission.h"
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <set>
#include <string>

namespace xpu2::interactive::classes {
namespace rdl = scene_rdl2::rdl2;

// The GPU feature families (a bit each).
enum Family : uint64_t {
    FamilyStructure = 1ull << 0,      // scene variables, layers, sets, outputs, user data
    FamilyCamera = 1ull << 1,
    FamilyTriangles = 1ull << 2,
    FamilySubdivision = 1ull << 3,    // tessellated on the host (xpu2_geom.dll)
    FamilyDisplacement = 1ull << 4,   // tessellated and displaced on the host
    FamilyCurves = 1ull << 5,
    FamilyPoints = 1ull << 6,
    FamilyInstances = 1ull << 7,
    FamilyMotion = 1ull << 8,         // geometry or transforms that move over the shutter
    FamilyDwaBase = 1ull << 10,
    FamilyLayers = 1ull << 11,        // DwaLayer, DwaMix, adjust, colour correct, two-sided
    FamilyHair = 1ull << 12,
    FamilyGlitter = 1ull << 13,
    FamilyToon = 1ull << 14,
    FamilySubsurface = 1ull << 15,
    FamilyUsd = 1ull << 16,
    FamilySwitch = 1ull << 17,
    FamilyMapsProcedural = 1ull << 20,
    FamilyMapsImage = 1ull << 21,
    FamilyMapsNormal = 1ull << 22,
    FamilyMapsProjection = 1ull << 23,
    FamilyLights = 1ull << 26,
    FamilyLightTextures = 1ull << 27,
    FamilyMeshLights = 1ull << 28,
    FamilyLightFilters = 1ull << 29,
    FamilyVolumes = 1ull << 32,       // homogeneous BaseVolumes
    FamilyVolumesMapped = 1ull << 33, // heterogeneous (bound attributes)
    FamilyVdb = 1ull << 34,
    FamilyVolumeEmission = 1ull << 35,
    FamilyCutouts = 1ull << 36,
};
// The families the GPU programs implement today (async_gpu.h's cpuOnly refuses the rest).
constexpr uint64_t GpuFamilies = FamilyStructure | FamilyCamera | FamilyTriangles | FamilySubdivision | FamilyDisplacement | FamilyCurves | FamilyPoints | FamilyMotion |
                                 FamilyInstances | FamilyDwaBase | FamilyLayers | FamilyHair | FamilyGlitter | FamilyToon |
                                 FamilySubsurface | FamilyUsd | FamilySwitch | FamilyMapsProcedural | FamilyMapsImage | FamilyMapsNormal |
                                 FamilyMapsProjection | FamilyLights | FamilyLightTextures | FamilyMeshLights | FamilyVolumes |
                                 FamilyVolumesMapped;  // their attenuation baked; mapped colour or emission still deferred

struct Class {
    Support support = Support::Fallback;
    uint64_t families = 0;
    std::string reason;  // the conformance identity (supported) or why not
};

inline bool hasAttribute(const rdl::SceneObject& o, const char* name) {
    for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i)
        if ((*i)->getName() == name) return true;
    return false;
}
template <class T>
inline T attributeOr(const rdl::SceneObject& o, const char* name, T fallback) {
    return hasAttribute(o, name) ? o.get<T>(name) : fallback;
}
// Whether an object's node transform or vertices move over the shutter.
inline bool moves(const rdl::SceneObject& o) {
    if (hasAttribute(o, "node_xform")) {
        const rdl::Mat4d& a = o.get<rdl::Mat4d>("node_xform", rdl::TIMESTEP_BEGIN);
        const rdl::Mat4d& b = o.get<rdl::Mat4d>("node_xform", rdl::TIMESTEP_END);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (a[i][j] != b[i][j]) return true;
    }
    for (const char* n : {"vertex_list_1", "velocity_list_0", "velocity_list_1", "accleration_list"})
        if (hasAttribute(o, n) && !o.get<rdl::Vec3fVector>(n).empty()) return true;
    return false;
}
// Whether any bindable attribute of an object is bound to a map.
inline bool bound(const rdl::SceneObject& o) {
    for (auto i = o.getSceneClass().beginAttributes(); i != o.getSceneClass().endAttributes(); ++i)
        if ((*i)->isBindable() && o.getBinding(**i)) return true;
    return false;
}

// The device class of one RDL object.
inline Class classify(const rdl::SceneObject& o) {
    const std::string cls = o.getSceneClass().getName();
    auto both = [&](uint64_t f) { return Class{(f & ~GpuFamilies) ? Support::Cpu : Support::Both, f, "xpu2:" + cls}; };
    static const std::set<std::string> structure = {"SceneVariables", "Layer", "GeometrySet", "LightSet", "ShadowSet", "ShadowReceiverSet",
                                                    "LightFilterSet", "RenderOutput", "UserData", "Metadata", "TraceSet"};
    if (structure.count(cls)) {
        // Shadow linking runs on the CPU lane (cpuOnly: shadow linking).
        if (cls == "ShadowSet" || cls == "ShadowReceiverSet") return Class{Support::Cpu, FamilyStructure, "xpu2:" + cls};
        return both(FamilyStructure);
    }
    static const std::set<std::string> cameras = {"PerspectiveCamera", "OrthographicCamera", "SphericalCamera", "FisheyeCamera",
                                                  "DomeMaster3DCamera", "BakeCamera"};
    if (cameras.count(cls)) return both(FamilyCamera | (moves(o) ? FamilyMotion : 0));
    if (cls == "RdlMeshGeometry") {
        uint64_t f = attributeOr<rdl::Bool>(o, "is_subd", false) ? FamilySubdivision : FamilyTriangles;
        if (moves(o)) f |= FamilyMotion;
        return both(f);
    }
    if (cls == "RdlCurveGeometry" || cls == "ModoFurGeometry") return both(FamilyCurves | (moves(o) ? FamilyMotion : 0));
    if (cls == "RdlPointGeometry") return both(FamilyPoints | (moves(o) ? FamilyMotion : 0));
    if (cls == "RdlInstancerGeometry") return both(FamilyInstances | (moves(o) ? FamilyMotion : 0));
    if (cls == "VdbGeometry") return both(FamilyVdb | (moves(o) ? FamilyMotion : 0));
    static const std::set<std::string> lights = {"EnvLight", "RectLight", "DiskLight", "SphereLight", "DistantLight", "SpotLight",
                                                 "CylinderLight", "MeshLight", "PortalLight"};
    if (lights.count(cls)) {
        uint64_t f = FamilyLights;
        if (cls == "MeshLight") f |= FamilyMeshLights;
        if (hasAttribute(o, "texture") && !o.get<rdl::String>("texture").empty()) f |= FamilyLightTextures;
        if (hasAttribute(o, "light_filters") && !o.get<rdl::SceneObjectVector>("light_filters").empty()) f |= FamilyLightFilters;
        if (attributeOr<rdl::Bool>(o, "mb", false) && moves(o)) f |= FamilyMotion;
        return both(f);
    }
    static const std::set<std::string> filters = {"IntensityLightFilter", "DecayLightFilter", "ColorRampLightFilter", "RodLightFilter",
                                                  "BarnDoorLightFilter", "CookieLightFilter", "CookieLightFilter_v2", "CombineLightFilter",
                                                  "VdbLightFilter"};
    if (filters.count(cls)) return both(FamilyLightFilters);
    if (cls == "BaseVolume") {
        uint64_t f = FamilyVolumes;
        if (bound(o)) f |= FamilyVolumesMapped;
        const rdl::Rgb e = attributeOr<rdl::Rgb>(o, "emission_color", rdl::Rgb(0.0f));
        if (attributeOr<rdl::Float>(o, "emission_intensity", 0.0f) != 0.0f && !(e.r == 0.0f && e.g == 0.0f && e.b == 0.0f))
            f |= FamilyVolumeEmission;
        return both(f);
    }
    if (cls == "VdbVolume") return both(FamilyVdb);
    if (cls == "CutoutVolume") return both(FamilyCutouts);
    static const std::set<std::string> displacements = {"NormalDisplacement", "VectorDisplacement", "CombineDisplacement", "SwitchDisplacement"};
    if (displacements.count(cls)) return both(FamilyDisplacement);
    if (cpu::dwaClass(cls)) {
        uint64_t f = FamilyDwaBase;
        if (attributeOr<rdl::Bool>(o, "show_glitter", false) && attributeOr<rdl::Float>(o, "glitter", 0.0f) > 0.0f) f |= FamilyGlitter;
        if (attributeOr<rdl::Float>(o, "scattering_radius", 0.0f) > 0.0f || cls == "DwaSkinMaterial") f |= FamilySubsurface;
        return both(f);
    }
    static const std::set<std::string> layers = {"DwaLayerMaterial", "DwaMixMaterial", "DwaAdjustMaterial", "DwaColorCorrectMaterial",
                                                 "DwaTwoSidedMaterial"};
    if (layers.count(cls)) return both(FamilyLayers);
    if (cls == "DwaToonMaterial") return both(FamilyToon);
    if (cls == "UsdPreviewSurface") return both(FamilyUsd);
    if (cls == "HairLayerMaterial" || cls == "HairColorCorrectMaterial" || cls.find("Hair") != std::string::npos) return both(FamilyHair);
    if (cls == "SwitchMaterial" || cls == "RaySwitchMaterial" || cls == "DwaSwitchMaterial") return both(FamilySwitch);
    if (const cpu::MapClass* m = cpu::mapClass(cls)) {
        (void)m;
        uint64_t f = FamilyMapsProcedural;
        if (cls.find("Image") != std::string::npos || cls.find("Udim") != std::string::npos || cls == "ModoTextureMap") f = FamilyMapsImage;
        else if (cls.find("Normal") != std::string::npos) f = FamilyMapsNormal;
        else if (cls.find("Project") != std::string::npos) f = FamilyMapsProjection;
        return both(f);
    }
    return Class{Support::Fallback, 0, "not rendered by XPU2: " + cls};
}

// The scene's summary for the session's log and the GPU: objects by class, the families the GPU must carry, those it
// does not, and the first objects that send the scene to MoonRay.
struct Summary {
    size_t counts[3] = {0, 0, 0};  // Fallback, Cpu, Both
    uint64_t gpuFamilies = 0, cpuFamilies = 0;
    std::vector<std::string> fallbacks;
};

}  // namespace xpu2::interactive::classes

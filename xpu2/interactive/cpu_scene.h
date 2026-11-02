// Private experimental canonical-scene adapter. Preparation is NOT qualification.
#pragma once
#include "canonical_session.h"
#include "surface_core.h"
#include "light_core.h"
#include "shade_material.h"
#include "curve_core.h"
#include "volume_core.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "host_texture.h"
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <array>
#include "light_image_build.h"
#include "light_mesh_build.h"
#include "light_tree_build.h"
#include "aov_plan.h"
#include "camera_core.h"
#include "light_filter_core.h"
#include "xpu2_geom.h"
#include <functional>
#include <set>
#include <mutex>
#include <future>
namespace xpu2::interactive::cpu {
namespace rdl=scene_rdl2::rdl2;
using surface::V3;
struct Unsupported:std::runtime_error{using std::runtime_error::runtime_error;};
inline void need(bool ok,const std::string& why){if(!ok)throw Unsupported(why);}
// Fixed diagnostic text needs no heap allocation on successful validation.
inline void need(bool ok,const char* why){if(!ok)throw Unsupported(why);}
// need, its message made only when the check fails: for checks in loops over a mesh's elements, where making the
// message each time (o.getName()+...) cost seconds (33 million tessellated indices: 3.4 s).
#define X2_NEED(ok,why) do{if(!(ok))need(false,(why));}while(0)
using Tri=surface::Triangle;
enum class Domain {Planar,DiffuseMesh,PlanarTextures,SurfaceTextures};
inline bool isSpatial(Domain d){return d==Domain::DiffuseMesh||d==Domain::SurfaceTextures;}
inline bool gpuTextures(Domain d){return d==Domain::PlanarTextures||d==Domain::SurfaceTextures;}
struct MeshData {std::string name;std::vector<V3> positions;std::vector<Tri> triangles;std::vector<std::shared_ptr<const ix::Buffer>> signature;
 // A plain polygon mesh (its own, still, not displaced): its signature without its node_xform (shape) and the node_xform
 // it was made at (placedBy, row vectors): a later preparation where only its node_xform changed takes it over, placed
 // anew as an instance (prepare), and a placement-only edit moves it on from there (placeMoved).
 bool placeable=false;std::vector<std::shared_ptr<const ix::Buffer>> shape;double placedBy[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
 // A displaced mesh MoonRay tessellated: which displacement is each part's, and whether polyvertices were made (the
 // rest of what shaped it is in the signature, deepSignature): kept across edits while they stay the same.
 std::string displacementKey;
 // A VDB volume's gate (VdbVolume's bounding mesh): its active voxels' box in index space and the world-to-index map
 // (rows); its leaf cells' crossings count only where a ray's half-line meets the box.
 bool gate=false;float gateToIndex[12]={};float gateLo[3]={},gateHi[3]={};
 // The path tracer's shading attributes (shade::MeshAttributes): render-space normals and texture coordinates.
 std::vector<V3> normals;std::vector<float> uvs;unsigned normalRate=shade::RateNone,uvRate=shade::RateNone;
 float worldToObject[12]={1,0,0,0,0,1,0,0,0,0,1,0};
 // Its primitive attributes (UserData) and their values.
 std::vector<shade::PrimAttr> prims;std::vector<float> primData;
 // MoonRay's polyvertices of each triangle (shade::MeshAttributes::polyVertices), where the wireframe or a polygon map
 // reads them: a QuadMesh face's four vertices, a TriMesh triangle's three and ~0u.
 std::vector<unsigned> polyVertices;
 // A subdivision surface's limit derivatives at its vertices (OpenSubdivMesh, shade::MeshAttributes::dpds); empty for any
 // other mesh.
 std::vector<V3> dpds,dpdt;
 // Motion blur: the vertices at the shutter's later time steps (positions holds the open's; Embree's time steps, linear
 // between them) and the shading normals' second time sample (a smooth mesh's from its second step: MoonRay keeps two
 // normal samples whatever the steps); both empty for a still mesh.
 std::vector<std::vector<V3>> steps;std::vector<V3> normals1;
 // Points (RdlPointGeometry, points_core.h) in place of triangles: their count, the centres in positions (and steps)
 // and each one's radius.
 unsigned points=0;std::vector<float> pointRadius;
 // An instancer's instance (MoonRay's instance hits: no toon smooth offset).
 bool instanced=false;
 // A moving instance: shaded between its time steps about the ray's time (as the instance transform there moves it),
 // with its smooth normals at each later step.
 bool stepShading=false;std::vector<std::vector<V3>> stepNormals;
 // A subdivision surface under motion: tessellated at each time step (steps), its limit normals (stepNormals) and
 // derivatives at the later steps.
 bool subdSteps=false;std::vector<std::vector<V3>> stepDpds,stepDpdt;
 // A curve set in place of triangles (curve_core.h's curves::Set): its type (curves::Basis, BasisNone for a mesh),
 // whether round, the flat cubic programs' quads a span; the control points (x, y, z, radius, render space), the
 // spans (first control point, curve, place along the curve) and each curve's span count.
 unsigned curveBasis=0,curveRound=0,curveSegments=4;std::vector<float> curvePoints;std::vector<unsigned> curveSpans,curveSpanCounts;
 // Curves under motion blur: the control points at the later time steps, and pointers to them (curves::Set).
 std::vector<std::vector<float>> curveSteps;std::vector<const float*> curveStepPointers;
 // Made for the coarse first pass (coarse_geometry.h): rendered for pictures only, never taken over by a later plan.
 bool coarse=false;};
// A vertex at a time in the shutter: between the two time steps about it, as Embree interpolates them.
inline V3 positionAt(const MeshData& m,unsigned i,float time){
 if(m.steps.empty())return m.positions[i];
 const unsigned n=unsigned(m.steps.size())+1;const float f=time*float(n-1);
 const unsigned k=std::min(unsigned(std::max(f,0.0f)),n-2);const float u=f-float(k);
 const V3& a=k?m.steps[k-1][i]:m.positions[i];const V3& b=m.steps[k][i];
 return V3{(1.0f-u)*a.x+u*b.x,(1.0f-u)*a.y+u*b.y,(1.0f-u)*a.z+u*b.z};}
// Motion blur's parameters (geom::MotionBlurParams, cpu_scene.h motionParams): on, the shutter's open and close as
// fractions of the first two motion steps' span, and the steps' times in seconds.
struct MotionParams {bool on=false;float openDelta=0,closeDelta=0,t0=0,t1=0,dt=0;float shutterOpen=0,shutterClose=0;
 bool operator==(const MotionParams& o)const{return on==o.on&&openDelta==o.openDelta&&closeDelta==o.closeDelta&&t0==o.t0&&t1==o.t1&&dt==o.dt;}};
// What the mesh loop renders: a geometry of the layer, or an instancer's instance (instancerItems): its reference,
// the instancer, the instance's index and name suffix, its transform at the shutter's ends and whether it moves, and
// the instancer's instance count (its primitive attributes' length).
struct RenderItem {const rdl::SceneObject* geom;const rdl::SceneObject* instancer=nullptr;int instance=-1;
 std::string suffix;scene_rdl2::math::Xform3f sample[2];bool moves=false;size_t count=0;};
// A mesh: its data, its material (~0u: none, a volume alone), and the volume it bounds (volume_core.h; -1: none).
// visibility: rdl2::Geometry::getVisibilityMask's eight surface bits (camera, shadow, the six lobe types), which a ray's
// mask must share; singleSided: side_type single (a mesh bounding no volume), its back faces culled.
struct Mesh {std::shared_ptr<const MeshData> data;unsigned material;int volume=-1;unsigned visibility=0xffu;bool singleSided=false;
 int geomLabel=-1;  // its geometry's label among the material AOVs' (shading::Geometry::getGeomLabelId)
 // A static instance sharing its reference's data (data in the reference's own space, shade::MeshAttributes::instance):
 // 1 placed, 2 placed and reflected; its transform (four rows of three, row vectors), the inverse's transpose, its own
 // worldToObject, and its instancer's primitive attributes for it. 0: the data is its own, in world space.
 unsigned instance=0;float objectToWorld[12]={1,0,0,0,1,0,0,0,1,0,0,0},normalToWorld[9]={1,0,0,0,1,0,0,0,1};
 float worldToObject[12]={1,0,0,0,0,1,0,0,0,0,1,0};std::vector<shade::PrimAttr> instancePrims;std::vector<float> instancePrimData;
 // The scene indexer's families of its geometry and of its material's tree (entity_classes.h's Family bits; 0 where
 // nothing classified them), for the cells by suitability (device_suit.h).
 uint64_t families=0;
 // Left to the CPU lane by the card's budget (routeMeshes): a stand-in on the GPU, its samples deferred to the CPU. Last:
 // the plan builds a Mesh from its first fields in order ({data, material, volume, visibility, singleSided, geomLabel}).
 bool cpuOnly=false;
 // The geometry that places this mesh, for a placement-only edit (placeMoved): its own where it has no instancer, else
 // its instancer (ownerInstancer) where its instances share their reference's data. Empty for an instancer's instances
 // baked each into a mesh of its own, which a move reads again whole.
 std::string owner;bool ownerInstancer=false;int ownerIndex=-1;  // ownerIndex: an instancer's instance, its place
};
// A plan mesh's curve set (curve_core.h) over its own arrays (host memory).
inline curves::Set curveSet(const MeshData& m){curves::Set c;if(!m.curveBasis)return c;
 c.points=m.curvePoints.data();c.spans=m.curveSpans.data();c.curveSpans=m.curveSpanCounts.data();c.spanCount=unsigned(m.curveSpans.size()/3);
 c.pointCount=unsigned(m.curvePoints.size()/4);c.basis=m.curveBasis;c.round=m.curveRound;c.segments=m.curveSegments;
 c.steps=m.curveStepPointers.empty()?nullptr:m.curveStepPointers.data();c.stepCount=1u+unsigned(m.curveStepPointers.size());return c;}
// A mesh light's tables (light_mesh_build.h) and what they were built from, for reuse by the next plan.
struct MeshLightEntry {std::string key;std::vector<std::shared_ptr<const ix::Buffer>> signature;std::shared_ptr<lights::MeshLightTables> tables;};
struct Plan {
 // The kit's MoonRay source switches the integrator reads (mixed::Config::modoSwitches).
 unsigned modoSwitches=0;
    CanonicalView source;surface::Camera camera{};unsigned width=0,height=0,spp=0,seed=0,sampling=3;
    surface::Filter filter=surface::Filter::Box;float filterWidth=1;
    Domain domain=Domain::Planar;unsigned surfaceLimit=1;
    std::vector<surface::Lambert> materials;std::map<unsigned,texture::Binding> diffuseMaps,emissionMaps;std::vector<Mesh> meshes;V3 sky{};
    size_t bytes=0,reusedMeshes=0,preparedMeshes=0;
 // The tessellation handler's routing (routeMeshes): the geometry the GPU holds, and the meshes left to the CPU lane.
 size_t gpuBytes=0;unsigned cpuOnlyMeshes=0;
 // The coarse first pass's meshes made lighter than the full plan's (0: the coarse plan is the full one).
 unsigned coarsened=0;
    // No production route changes merely because preparation succeeds.
    bool qualified=false;
    // MoonRay adaptive sampling (sampling_mode 2), admitted only where the caller
    // renders it (the asynchronous renderer); spp is then the maximum.
    bool adaptive=false;unsigned minSamples=0,maxSamples=0;float targetError=0;
    // A uniform frame by sampling generations (the asynchronous renderer): uniform's samples, its budget, priced as
    // uniform's time and spent where they show; adaptive is then set, minSamples is every pixel's seed, maxSamples
    // its most, the target error is the budget's, and spp stays uniform's.
    unsigned long long sampleBudget=0;
    // The kit's settings for them: the budget spent exactly (not priced as uniform's time), the error no tile need go
    // below (0: none), and whether the CPU lane's workers take part beside the GPU.
    bool budgetBySamples=false;float noiseTarget=0;bool generationsCpu=true;
    // Realtime (the kit's RT, following the scene): a generations frame's first pass is a pixel in each 2x2 of a tile,
    // the second the rest (async_render.h Adaptive), and its pictures go out sooner.
    bool realtime=false;
    // Whether a mesh follows the camera (adaptive tessellation, by the dicing camera): no camera-only fast path then.
    bool cameraDependent=false;
    // MoonRay's path tracer (PATH_TRACER.md), admitted only where the caller
    // renders it (the asynchronous renderer): its lights, each mesh's light set
    // as a mask over them, and the integrator's settings.
    bool paths=false;std::vector<lights::Light> lights;std::vector<unsigned> meshLights;
    // The light sets: each a bitset over the lights, lightSetWords words long; meshLights holds each mesh's set.
    std::vector<unsigned> lightSetBits;unsigned lightSetWords=1;
    // The portals' copies of the lights they reference (lights::Light::ref).
    std::vector<std::shared_ptr<lights::Light>> portalRefs;
    // Adaptive light sampling (light_sampling_mode 1, light_tree.h): each light set's tree root (-1: none) in the
    // shared node array, and the splitting threshold (light_sampling_quality).
    unsigned lightSamplingMode=0;float lightSamplingQuality=0.5f;std::vector<lights::TreeNode> lightTree;std::vector<int> lightTreeRoots;
    // The lights' textures (light_image_build.h): their tables, which the lights point into.
    std::vector<std::shared_ptr<lights::ImageData>> lightImages;
    // The mesh lights' faces and sampling trees (light_mesh_build.h), which those lights point into.
    std::vector<MeshLightEntry> meshLightTables;
    unsigned bsdfSamples=1,lightSamples=1,maxDepth=0,maxDiffuseDepth=0,clampDepth=1;float rrThreshold=0,clampValue=0;
    float presenceThreshold=0.999f,presenceQuality=0.75f;
    unsigned maxGlossyDepth=0,maxMirrorDepth=0,maxHairDepth=0,maxPresenceDepth=0,shadowTerminatorFix=0;float roughnessClamping=0,transparencyThreshold=1;
    unsigned bssrdfSamples=4,maxSubsurfacePerPath=1,enableSss=1;
    // The path tracer's materials (shade_material.h): one record per material, parameters in a pool.
    std::vector<shade::MaterialRecord> materialRecords;std::vector<float> materialPool;
    // World to MoonRay's render space: the camera's space at its first motion step (3x4 rows).
    float worldToRender[12]={1,0,0,0,0,1,0,0,0,0,1,0};
    // The volumes bound to meshes (volume_core.h), and MoonRay's volume scene variables.
    std::vector<volume::Shader> volumes;unsigned maxVolumeDepth=1,volumeIlluminationSamples=4,volumeOverlapMode=0;
    // VDB grids (volume_grid.h): each read once (shared with the next plan while its file is unchanged), and the
    // integrator's view of them; volume_quality and volume_shadow_quality.
    std::vector<std::shared_ptr<struct VdbGridData>> volumeGrids;std::vector<volume::Grid> volumeGridViews;
    // The render space's origin in world space (an empty shading state's P: CookieLightFilter's map evaluation).
    V3 renderOrigin{0,0,0};
    // The mapped volumes' map evaluation (volume_core.h's MapEval, cpu::runVolumeMaps over this plan).
    volume::MapEval volumeMaps{};
    float volumeQuality=0.5f,volumeShadowQuality=1.0f;
    // Emissive volumes as light sources (volume_emission.h): their regions and the distributions' pool.
    std::vector<volume::Region> emissiveRegions;std::vector<float> emissionPool;
    float volumeTransmittanceThreshold=0.005f,volumeAttenuationFactor=0.65f,volumeContributionFactor=0.65f,volumePhaseAttenuationFactor=0.5f;
    // The DwaBase parameter slots the deepest material tree needs (shade_dwalayer.h); whether a material has glitter
    // (the shading threads then need a glitter scratch, shade_glitter.h).
    unsigned dwaSlots=0;bool glitter=false;
    // A random-walk subsurface material, and then MoonRay's integrator sequences its walk draws from (walkSequences).
    bool randomWalk=false;std::shared_ptr<const std::vector<float>> walkSequences;
    // The textures the path tracer's maps read (shade_mapnodes.h): each file once, and its host mip pyramid.
    std::vector<std::shared_ptr<texture::File>> textures;std::vector<const tex::Image*> textureImages;
    // The render outputs and their AOVs (aov_plan.h), and the keys of the primitive attributes they read.
    std::shared_ptr<const aov::Plan> aovs;std::vector<unsigned> aovPrimKeys;
    // The camera's bokeh image (its distribution, light_image_build.h) and a bake camera's maps (camera_core.h's
    // CameraMap over these arrays), which plan.camera points into.
    std::shared_ptr<lights::ImageData> bokehImage;
    struct CameraMapData {std::vector<float> positions,normals,normalMap;surface::CameraMap view{};};
    std::shared_ptr<CameraMapData> cameraMap;
    // The pixels rendered (SceneVariables::getRezedSubViewport: min x, min y, max x, max y, half open, rows from the
    // bottom); the rest of the frame keeps no sample.
    int viewport[4]={0,0,0x7fffffff,0x7fffffff};
    // PathIntegrator::initPrimaryRay's texture differential scale (texture_blur, the resolution divisor).
    float textureDiffScale=1;
    // The camera's pixel sample map (uniform sampling): each pixel's sample count, rows from the bottom; empty: none.
    std::vector<unsigned> pixelSamples;
    // Motion blur (meshMotionSteps, camera_core.h's cameraAtTime) and the camera's shutter bias (the samples' times).
    MotionParams motion;float shutterBias=0;
    // The lights moving under motion blur (light_core.h's LightMotion, by light index), and each light's record at the
    // shutter's close (none: still), which bounds it in the light trees.
    std::vector<lights::LightMotion> lightMotion;std::vector<std::shared_ptr<const lights::Light>> lightClose;
    // Each light's scene object by name, in the lights' order: a light-only edit (updateLights) finds its light by it.
    std::vector<std::string> lightNames;
    // Light filters (light_filter_core.h): the filters, the distinct filter lists, and each mesh's table of a list a
    // light (filterTables from meshFilterTable's offset, lights entries, ~0u: no filters; ~0u offset: none at all),
    // and the cookies' textures.
    // The materials' lobe light sets (shade::Lobe::lightSet), lightSetWords words each over the lights.
    std::vector<unsigned> lobeLightSetBits;
    std::vector<lights::Filter> filters;std::vector<lights::FilterList> filterLists;std::vector<unsigned> filterTables,meshFilterTable;
    std::vector<std::shared_ptr<lights::ImageData>> filterImages;
    // Shadow linking (geom::internal::ShadowLinking, by each caster mesh's layer assignment): a mesh's link (~0u: none),
    // each link the lights it casts no shadow for (lightSetWords words) and the receivers' assignments it casts none on
    // (or the only ones it does, complemented); each mesh's layer assignment.
    struct ShadowLink {unsigned lights=0,receivers=0,receiverCount=0,complement=0;};
    std::vector<ShadowLink> shadowLinks;std::vector<unsigned> meshShadowLink,shadowLinkBits;std::vector<int> shadowReceivers,meshAssignment;
    // World to MoonRay's render space in doubles (row vectors), and the camera frustums MoonRay tessellates subdivision
    // surfaces by (PerspectiveCamera::computeFrustum over the render region, at the shutter's open and close: camera to
    // screen, the six clip planes, the viewport), or a fisheye camera's fishtum (xpu2_geom.h).
    double world2render[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    std::vector<float> frustumC2S,frustumPlanes;std::vector<int> frustumViewports;
    bool fishtum=false;float fishtumData[9]={0,0,0,0,0,0,0,0,0};
};
using PlanView=std::shared_ptr<const Plan>;
// What a scene may hold: room for scenes from Modo (an RTX 3090 holds this much several times over).
// bytes: the prepared geometry the machine's RAM holds (past it the scene falls back); gpuBytes: what the card holds (past
// it the largest meshes are left to the CPU lane, Mesh::cpuOnly).
// coarse: the coarse first pass's plan (coarse_geometry.h): lighter tessellation and dense meshes simplified.
struct Limits {size_t vertices=16000000,triangles=16000000,bytes=1ull<<30,objects=1000000,pixels=8388608,gpuBytes=~size_t(0);bool coarse=false;};
// Whether the plan being prepared is the coarse first pass's (cpu::prepare sets it about the preparation; coarse_geometry.h).
inline std::atomic<bool>& coarseTessellation(){static std::atomic<bool> on{false};return on;}
// MoonRay's tessellation for the coarse pass: a quarter of the resolution, four times the screen error.
inline int coarseResolution(int resolution){return coarseTessellation()?std::max(1,resolution/4):resolution;}
inline float coarseError(float error){return coarseTessellation()&&error>0.0f?error*4.0f:error;}
// The most prepared geometry (positions and triangles) a session may ask for: its own budget is the machine's
// (geometryBudget in xpu2_session.cc), up to this.
constexpr size_t HardGeometryBytes=64ull<<30;
// The budget exceeded, with how much was asked and allowed.
inline std::string geometryBudgetText(size_t bytes,size_t limit){
 return "Prepared geometry budget: "+std::to_string(bytes>>20)+" MB of positions and triangles, of "+std::to_string(limit>>20)+" MB";}
template<class T>inline T get(const rdl::SceneObject& o,const char* name){return o.get<T>(name);}
template<class T>inline T get(const rdl::SceneObject& o,const std::string& name){return o.get<T>(name);}
struct State {bool isDefault,stationary;};
template<class T>inline bool equal(const T& a,const T& b){return a==b;}
inline bool equal(const rdl::SceneObjectIndexable& a,const rdl::SceneObjectIndexable& b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());}
template<class T>inline State state(const rdl::SceneObject& o,const rdl::Attribute& a){
 const auto& begin=o.get<T>(a.getName(),rdl::TIMESTEP_BEGIN);
 const auto& end=o.get<T>(a.getName(),a.isBlurrable()?rdl::TIMESTEP_END:rdl::TIMESTEP_BEGIN);
 // Exact value comparisons, with no temporary serialization or padding reads.
 // Object pointers are compared only within this one canonical replica.
 const auto& defaults=a.getDefaultValue<T>();
 bool stationary=!a.isBlurrable()||equal(begin,end);
 return {stationary&&equal(begin,defaults),stationary};
}
inline State state(const rdl::SceneObject& o,const rdl::Attribute& a){
#define CASE(K,T) case rdl::K:return state<rdl::T>(o,a);
 switch(a.getType()){
 CASE(TYPE_BOOL,Bool) CASE(TYPE_INT,Int) CASE(TYPE_LONG,Long) CASE(TYPE_FLOAT,Float) CASE(TYPE_DOUBLE,Double) CASE(TYPE_STRING,String)
 CASE(TYPE_RGB,Rgb) CASE(TYPE_RGBA,Rgba) CASE(TYPE_VEC2F,Vec2f) CASE(TYPE_VEC2D,Vec2d) CASE(TYPE_VEC3F,Vec3f) CASE(TYPE_VEC3D,Vec3d) CASE(TYPE_VEC4F,Vec4f) CASE(TYPE_VEC4D,Vec4d)
 CASE(TYPE_MAT3F,Mat3f) CASE(TYPE_MAT3D,Mat3d) CASE(TYPE_MAT4F,Mat4f) CASE(TYPE_MAT4D,Mat4d) CASE(TYPE_SCENE_OBJECT,SceneObject*)
 CASE(TYPE_BOOL_VECTOR,BoolVector) CASE(TYPE_INT_VECTOR,IntVector) CASE(TYPE_LONG_VECTOR,LongVector) CASE(TYPE_FLOAT_VECTOR,FloatVector) CASE(TYPE_DOUBLE_VECTOR,DoubleVector) CASE(TYPE_STRING_VECTOR,StringVector)
 CASE(TYPE_RGB_VECTOR,RgbVector) CASE(TYPE_RGBA_VECTOR,RgbaVector) CASE(TYPE_VEC2F_VECTOR,Vec2fVector) CASE(TYPE_VEC2D_VECTOR,Vec2dVector) CASE(TYPE_VEC3F_VECTOR,Vec3fVector) CASE(TYPE_VEC3D_VECTOR,Vec3dVector) CASE(TYPE_VEC4F_VECTOR,Vec4fVector) CASE(TYPE_VEC4D_VECTOR,Vec4dVector)
 CASE(TYPE_MAT3F_VECTOR,Mat3fVector) CASE(TYPE_MAT3D_VECTOR,Mat3dVector) CASE(TYPE_MAT4F_VECTOR,Mat4fVector) CASE(TYPE_MAT4D_VECTOR,Mat4dVector) CASE(TYPE_SCENE_OBJECT_VECTOR,SceneObjectVector) CASE(TYPE_SCENE_OBJECT_INDEXABLE,SceneObjectIndexable)
 default:throw Unsupported(o.getName()+" / unknown attribute type");}
#undef CASE
}
inline void attributes(const rdl::SceneObject& obj,const std::set<std::string>& changed,const std::set<std::string>& bindings={},
                       const std::set<std::string>& moving={}){
 for(auto i=obj.getSceneClass().beginAttributes();i!=obj.getSceneClass().endAttributes();++i){auto& a=**i;
  // A material's or a geometry's label names it in the AOVs and light path expressions alone (aov_plan.h reads it).
  if(a.getName()=="label"&&(obj.isA<rdl::Material>()||obj.isA<rdl::Geometry>()))continue;
  if(a.getName()=="extra_aovs"&&obj.isA<rdl::Material>())continue;
  std::string why=obj.getName()+" / "+a.getName();
  need(!a.isBindable()||!obj.getBinding(a)||bindings.count(a.getName()),why+": map binding unsupported");
  auto s=state(obj,a);need(s.stationary||moving.count(a.getName()),why+": animated value unsupported");
  need(s.isDefault||changed.count(a.getName()),why+": nondefault setting unsupported");
 }
}
inline bool zero(rdl::Rgb c){return c.r==0&&c.g==0&&c.b==0;}
inline V3 color(rdl::Rgb c){V3 v{c.r,c.g,c.b};need(surface::finite(v)&&v.x>=0&&v.y>=0&&v.z>=0,"Invalid color");return v;}
inline rdl::Mat4d translation(const rdl::SceneObject& obj){
 auto m=get<rdl::Mat4d>(obj,"node_xform");
 for(int i=0;i<4;++i)for(int j=0;j<4;++j){need(std::isfinite(m[i][j]),obj.getName()+": nonfinite transform");if(i==3&&j<3)continue;need(m[i][j]==(i==j?1:0),obj.getName()+": only translations supported");}
 return m;
}
// Explicit spatial opt-in, bounded static affine transforms (reflected ones in affine()).
// A camera turned and moved but not scaled or sheared: the path tracer's camera (rotations are MoonRay's
// node_xform as the camera-to-render transform).
inline rdl::Mat4d rigid(const rdl::SceneObject& obj){
 auto m=get<rdl::Mat4d>(obj,"node_xform");
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j]),obj.getName()+": nonfinite transform");
 need(m[0][3]==0&&m[1][3]==0&&m[2][3]==0&&m[3][3]==1,obj.getName()+": projective transform unsupported");
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){double d=m[i][0]*m[j][0]+m[i][1]*m[j][1]+m[i][2]*m[j][2];need(std::abs(d-(i==j?1:0))<=1e-5,obj.getName()+": only rotations and translations supported");}
 double det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
 need(det>0,obj.getName()+": reflected transform unsupported");return m;
}
// A camera at a time in the shutter: its two time steps blended, which MoonRay takes as they come (a rotation blended
// with another is no longer one: Camera::update only warns of its scale or shear). Affine, not reflected.
inline void rigidMatrix(const rdl::Mat4d& m,const std::string& name){
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j]),name+": nonfinite transform");
 // Its blend's float weights sum to one within a float's precision (1 + 3e-8 at a weight of 0.4).
 need(std::abs(m[0][3])<=1e-6&&std::abs(m[1][3])<=1e-6&&std::abs(m[2][3])<=1e-6&&std::abs(m[3][3]-1)<=1e-6,name+": projective transform unsupported");
 double det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
 need(det>1e-12,name+": reflected or degenerate transform unsupported");}
inline rdl::Mat4d affine(const rdl::SceneObject& obj){
 auto m=get<rdl::Mat4d>(obj,"node_xform");
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j])&&std::abs(m[i][j])<=10000,"Spatial transform bound");
 need(m[0][3]==0&&m[1][3]==0&&m[2][3]==0&&m[3][3]==1,"Projective mesh transform unsupported");
 double det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
 // A reflected transform (a mirrored copy) as MoonRay renders it: its vertices are moved into render space with their
 // faces in the order given (PolygonMesh::transformPrimitive before the tessellation), so the geometric normal, and the
 // smooth normals PolyMesh makes from the faces, turn over with the mirror; supplied normals go through the inverse's
 // transpose (Attributes::transformAttributes) and do not. XPU2 bakes its meshes so and keeps the faces' order.
 need(std::isfinite(det)&&std::abs(det)>1.e-12,"Singular transform unsupported");return m;
}
// One of MoonRay's lights as the path tracer's record (light_core.h), or false
// when it is off. Settings beyond these fall back. The arithmetic of each
// light's update(): radiance, normalization by area (and scene scale), and its
// local frame, the node transform turned about X by 180 degrees.
// A light's texture (Light::updateImageMap) with its colour and placement settings, tabulated for its mapping, or
// taken from the previous plan when nothing about it changed. A texture that cannot be read lights with the fatal
// colour (MoonRay logs the error and goes on); one with no usable level turns the light off (false).
// MoonRay's camera (pbr/camera's update functions and the camera DSOs' computeProjectionMatrix) as the path tracer's
// surface::Camera (camera_core.h): its type, the rezed aperture and region windows, the projection (raster to camera as
// the inverse of camera to screen times screen to raster, in floats as MoonRay's), the stereo eye, the lens and its
// bokeh, and the other types' settings. Motion blur is refused, so the camera is at its shutter's open (each setting's
// first time step, as MoonRay reads them without it).
// rndr::readPixelBuffer into a float buffer w x h: the file's first channel, or the luminance of its first three
// (channel_sum .2126 .7152 .0722), resized to the frame by OIIO's default filter, its rows from the bottom.
inline std::vector<float> readPixelSampleMap(const std::string& file,unsigned w,unsigned h){
 auto in=OIIO::ImageInput::open(file);need(in!=nullptr,"Pixel sample map \""+file+"\" cannot be opened");
 const OIIO::ImageSpec spec=in->spec();
 std::vector<unsigned char> raw(size_t(spec.width)*size_t(spec.height)*size_t(spec.nchannels)*spec.format.size());
 need(in->read_image(0,0,0,spec.nchannels,spec.format,raw.data()),"Pixel sample map \""+file+"\" cannot be read");in->close();
 OIIO::ImageBuf source(spec,raw.data());
 if(spec.nchannels!=1){
  need(spec.nchannels==3||spec.nchannels==4,"Pixel sample map \""+file+"\": "+std::to_string(spec.nchannels)+" channels");
  OIIO::ImageBuf converted;const float weights[3]={.2126f,.7152f,.0722f};OIIO::ROI roi=source.roi();roi.chbegin=0;roi.chend=3;
  need(OIIO::ImageBufAlgo::channel_sum(converted,source,weights,roi),"Pixel sample map: channel conversion");
  source.clear();source.copy(converted);}
 if(unsigned(spec.width)!=w||unsigned(spec.height)!=h){
  OIIO::ImageSpec rs(int(w),int(h),1,OIIO::TypeDesc::FLOAT);OIIO::ImageBuf resized(rs);
  need(OIIO::ImageBufAlgo::resize(resized,source),"Pixel sample map: resize");source.clear();source.copy(resized);}
 std::vector<float> data(size_t(w)*h,0.0f);OIIO::ImageSpec fs(int(w),int(h),1,OIIO::TypeDesc::FLOAT);OIIO::ImageBuf flipped(fs,data.data());
 need(OIIO::ImageBufAlgo::flip(flipped,source),"Pixel sample map: flip");
 return data;
}
// SceneObject::get(key, t): a blurrable value at a shutter time (0 the open, 1 the close), its time steps blended at that
// time rescaled into the motion steps (SceneContext::computeTimeRescalingCoeffs over the primary camera's shutter with
// motion blur, the frame's time 0 without).
struct TimeRescale {float scale=0,offset=0;};
inline TimeRescale timeRescale(const rdl::SceneVariables& sv,const rdl::SceneObject* camera){
 const auto& steps=get<rdl::FloatVector>(sv,"motion_steps");TimeRescale r;
 float open=0.0f,close=0.0f;
 if(camera&&get<rdl::Bool>(sv,"enable_motion_blur")){open=get<rdl::Float>(*camera,"mb_shutter_open");close=get<rdl::Float>(*camera,"mb_shutter_close");}
 if(steps.size()>=2&&steps[0]!=steps[1]){const float inv=1.0f/(steps[1]-steps[0]);r.scale=(close-open)*inv;r.offset=(open-steps[0])*inv;}
 return r;}
inline thread_local TimeRescale sceneTimeRescale;  // the scene's, set as a plan is prepared
inline rdl::Mat4d xformAt(const rdl::SceneObject& o,float t){
 const float ts=sceneTimeRescale.scale*t+sceneTimeRescale.offset;
 const auto& a=o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN);const auto& b=o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END);
 return (a*double(1.0f-ts))+(b*double(ts));}
inline float floatAt(const rdl::SceneObject& o,const char* name,float t){
 const float ts=sceneTimeRescale.scale*t+sceneTimeRescale.offset;
 return (o.get<rdl::Float>(name,rdl::TIMESTEP_BEGIN)*(1.0f-ts))+(o.get<rdl::Float>(name,rdl::TIMESTEP_END)*ts);}
inline scene_rdl2::math::Mat4f cameraProjection(const rdl::SceneObject& cam,bool perspective,const std::array<float,4>& window,float interocular,
                                                bool shutterClose=false){
 namespace sm=scene_rdl2::math;
 const float minimum=perspective?0.00001f:0.01f;
 const float focal=perspective?sm::max(floatAt(cam,"focal",shutterClose?1.0f:0.0f),0.00001f):1.0f;
 const float halfFilmWidth=sm::max(get<rdl::Float>(cam,"film_width_aperture"),minimum)*0.5f;
 const float near=sm::max(get<rdl::Float>(cam,"near"),minimum),far=sm::max(get<rdl::Float>(cam,"far"),minimum);
 float offsetX=get<rdl::Float>(cam,"horizontal_film_offset");const float offsetY=get<rdl::Float>(cam,"vertical_film_offset");
 float par=get<rdl::Float>(cam,"pixel_aspect_ratio");if(sm::isEqual(par,0.0f))par=1.0f;
 sm::Mat4f c2s(sm::zero);
 if(perspective){
  // The stereo eye's film offset (see MoonRay's pbr/doc/stereo_projection_top_view.jpg).
  const float convergence=sm::max(get<rdl::Float>(cam,"stereo_convergence_distance"),0.00001f);
  offsetX+=interocular*focal/convergence;
  const float factor=near/focal;
  const float w0=(window[0]*halfFilmWidth+offsetX)*factor,w1=(window[1]*halfFilmWidth+offsetY)*factor/par,
              w2=(window[2]*halfFilmWidth+offsetX)*factor,w3=(window[3]*halfFilmWidth+offsetY)*factor/par;
  c2s[0][0]=(2.0f*near)/(w2-w0);c2s[1][1]=(2.0f*near)/(w3-w1);
  c2s[2][0]=(w2+w0)/(w2-w0);c2s[2][1]=(w3+w1)/(w3-w1);c2s[2][2]=-(far+near)/(far-near);c2s[2][3]=-1.0f;
  c2s[3][2]=-(2.0f*far*near)/(far-near);
  return sm::Mat4f::translate(sm::Vec4f(interocular,0.0f,0.0f,1.0f))*c2s;
 }
 const float w0=(window[0]*halfFilmWidth+offsetX),w1=(window[1]*halfFilmWidth+offsetY)/par,
             w2=(window[2]*halfFilmWidth+offsetX),w3=(window[3]*halfFilmWidth+offsetY)/par;
 c2s[0][0]=2.0f/(w2-w0);c2s[1][1]=2.0f/(w3-w1);c2s[2][2]=-2.0f/(far-near);
 c2s[3][0]=-(w2+w0)/(w2-w0);c2s[3][1]=-(w3+w1)/(w3-w1);c2s[3][2]=-(far+near)/(far-near);c2s[3][3]=1.0f;
 return c2s;
}
inline void admitCamera(const rdl::SceneObject& camera,const rdl::SceneVariables& sv,Plan& plan,const Plan* previous){
 namespace sm=scene_rdl2::math;
 const std::string cls=camera.getSceneClass().getName(),name=camera.getName();
 // The settings each type reads; any other must hold its default (pixel_sample_map, medium_material and
 // medium_geometry among them). A setting's animation is its first time step's value without motion blur.
 static const std::set<std::string> common{"node_xform","near","far","mb_shutter_open","mb_shutter_close","mb_shutter_bias","pixel_sample_map"};
 static const std::map<std::string,std::set<std::string>> reads{
  {"PerspectiveCamera",{"focal","film_width_aperture","horizontal_film_offset","vertical_film_offset","pixel_aspect_ratio","dof","dof_aperture",
   "dof_focus_distance","bokeh","bokeh_sides","bokeh_image","bokeh_angle","bokeh_weight_location","bokeh_weight_strength","stereo_view",
   "stereo_interocular_distance","stereo_convergence_distance"}},
  {"OrthographicCamera",{"film_width_aperture","horizontal_film_offset","vertical_film_offset","pixel_aspect_ratio","dof","dof_aperture",
   "dof_focus_distance","bokeh","bokeh_sides","bokeh_image","bokeh_angle","bokeh_weight_location","bokeh_weight_strength"}},
  {"SphericalCamera",{"min_latitude","max_latitude","latitude_zoom_offset","min_longitude","max_longitude","longitude_zoom_offset","focal",
   "inside_out","offset_radius"}},
  {"FisheyeCamera",{"mapping","format","zoom","fov"}},
  // head_tilt_map is read by no code of MoonRay's (#if 0); the interocular map never scales (camera_core.h's CameraMap).
  {"DomeMaster3DCamera",{"FOV_vertical_angle","FOV_horizontal_angle","flip_ray_x","flip_ray_y","stereo_view","stereo_convergence_distance",
   "stereo_interocular_distance","interocular_distance_map_file_name","head_tilt_map","zenith_mode"}},
  {"BakeCamera",{"geometry","udim","uv_attribute","mode","bias","use_relative_bias","map_factor","normal_map","normal_map_space"}}};
 // scene_rdl2 math degreesToRadians (MathUtil.h).
 auto degreesToRadians=[](float degrees){return degrees*sm::sPi/180.0f;};
 const auto found=reads.find(cls);need(found!=reads.end(),name+": "+cls+" is not a camera MoonRay has");
 for(auto i=camera.getSceneClass().beginAttributes();i!=camera.getSceneClass().endAttributes();++i){auto& a=**i;
  const std::string n=a.getName();if(common.count(n)||found->second.count(n))continue;
  const auto s=state(camera,a);need(s.isDefault&&s.stationary,name+" / "+n+": nondefault setting unsupported");}
 auto& cam=plan.camera;cam=surface::Camera{};
 // Camera::update: the windows.
 const auto ap=sv.getRezedApertureWindow(),rg=sv.getRezedRegionWindow();
 cam.apertureWidth=float(ap.width());cam.apertureHeight=float(ap.height());
 cam.regionX=float(rg.min().x-ap.min().x);cam.regionY=float(rg.min().y-ap.min().y);
 {// Camera::update: the camera at the shutter's open (computeC2W(0)); render space is the camera at the first motion
  // step (Scene::updateActiveCamera), mCamera2Render the one into the other.
  const auto begin=rigid(camera);const sm::Mat4d open=xformAt(camera,0.0f);
  rigidMatrix(open,name);
  const sm::Mat4f c2r=sm::toFloat(open*begin.inverse()),r2c=c2r.inverse(),r2w=sm::toFloat(begin);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j){cam.cameraToRender.m[i][j]=float(open[i][j]);cam.camera2Render.m[i][j]=c2r[i][j];
   cam.render2Camera.m[i][j]=r2c[i][j];cam.render2World.m[i][j]=r2w[i][j];}}
 cam.shutterDuration=get<rdl::Float>(camera,"mb_shutter_close")-get<rdl::Float>(camera,"mb_shutter_open");
 if(get<rdl::Bool>(sv,"enable_motion_blur")){
  // Camera::update under motion blur: the camera at the shutter's close relative to its open, each way, as a rotation
  // (Quaternion3f of its rows, made positive) and a translation; computeCamera2Render slerps and lerps them.
  const sm::Mat4d c2w0=xformAt(camera,0.0f),c2w1=xformAt(camera,1.0f);
  rigidMatrix(c2w1,name);
  const sm::Mat4d rel=c2w1*c2w0.inverse();
  const sm::Mat4f close2open=sm::toFloat(rel),open2close=sm::toFloat(rel.inverse());
  auto quat=[](const sm::Mat4f& m,float* q){sm::Quaternion3f r(sm::asVec3(m.vx),sm::asVec3(m.vy),sm::asVec3(m.vz));if(r.r<0.f)r*=-1.0f;q[0]=r.r;q[1]=r.i;q[2]=r.j;q[3]=r.k;};
  quat(close2open,cam.closeToOpenRot);quat(open2close,cam.openToCloseRot);
  cam.closeToOpenTrans=V3{close2open.vw.x,close2open.vw.y,close2open.vw.z};cam.openToCloseTrans=V3{open2close.vw.x,open2close.vw.y,open2close.vw.z};
  if(!sm::isEqual(c2w0,c2w1))cam.motion=1;}
 cam.nearClip=get<rdl::Float>(camera,"near");cam.farClip=get<rdl::Float>(camera,"far");
 need(std::isfinite(cam.nearClip)&&std::isfinite(cam.farClip),name+": invalid clipping planes");
 auto fin=[&](float v,const char* what){need(std::isfinite(v),name+": invalid "+std::string(what));return v;};
 if(cls=="PerspectiveCamera"||cls=="OrthographicCamera"){
  const bool perspective=cls=="PerspectiveCamera";
  cam.type=perspective?surface::CameraPerspective:surface::CameraOrthographic;
  // ProjectiveCamera::updateImpl: the aperture window, screen to raster, raster to camera.
  const float invAspect=cam.apertureHeight/cam.apertureWidth;
  const std::array<float,4> window{-1.0f,-invAspect,1.0f,invAspect};
  float interocular=0.0f;
  if(perspective){
   const int view=get<rdl::Int>(camera,"stereo_view");need(view>=0&&view<=2,name+": unknown stereo view");
   if(view!=0){interocular=(view==1?0.5f:-0.5f)*get<rdl::Float>(camera,"stereo_interocular_distance");cam.focalPoint={-interocular,0.0f,0.0f};}
  }
  const sm::Mat4f c2s=cameraProjection(camera,perspective,window,interocular);
  if(perspective){
   // ProjectiveCamera::updateImpl's render region window and PerspectiveCamera::computeFrustumImpl (the render region's):
   // the frustums MoonRay's tessellation reads (GeometryManager: the shutter's open and close).
   const float offX=2*cam.regionX/cam.apertureWidth,offY=2*cam.regionY/cam.apertureHeight;
   const float wR=2*float(rg.width())/cam.apertureWidth,hR=2*float(rg.height())/cam.apertureHeight;
   const std::array<float,4> region{-1.0f+offX,invAspect*(-1.0f+offY),-1.0f+offX+wR,invAspect*(-1.0f+offY+hR)};
   for(int close=0;close<2;++close){
    const sm::Mat4f c=cameraProjection(camera,true,region,interocular,close==1);
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)plan.frustumC2S.push_back(c[i][j]);
    const float planes[6][4]={{c[0][3]-c[0][0],c[1][3]-c[1][0],c[2][3]-c[2][0],c[3][3]-c[3][0]},{c[0][3]+c[0][0],c[1][3]+c[1][0],c[2][3]+c[2][0],c[3][3]+c[3][0]},
                              {c[0][3]+c[0][1],c[1][3]+c[1][1],c[2][3]+c[2][1],c[3][3]+c[3][1]},{c[0][3]-c[0][1],c[1][3]-c[1][1],c[2][3]-c[2][1],c[3][3]-c[3][1]},
                              {c[0][3]-c[0][2],c[1][3]-c[1][2],c[2][3]-c[2][2],c[3][3]-c[3][2]},{c[0][3]+c[0][2],c[1][3]+c[1][2],c[2][3]+c[2][2],c[3][3]+c[3][2]}};
    for(const auto& pl:planes)plan.frustumPlanes.insert(plan.frustumPlanes.end(),pl,pl+4);
    const int vx=int(cam.regionX),vy=int(cam.regionY);
    plan.frustumViewports.insert(plan.frustumViewports.end(),{vx,vy,vx+int(rg.width()),vy+int(rg.height())});
   }
  }
  sm::Mat4f s2r(sm::one);
  s2r[0][0]=cam.apertureWidth*0.5f;s2r[1][1]=cam.apertureHeight*0.5f;s2r[3][0]=cam.apertureWidth*0.5f;s2r[3][1]=cam.apertureHeight*0.5f;
  const sm::Mat4f rtoc=(c2s*s2r).inverse();
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)cam.rasterToCamera.m[i][j]=fin(rtoc[i][j],"projection");
  // The projection at the shutter's close (mC1toS, mRtoC1: a perspective camera's focal may move), and camera to
  // screen at both (the motion vectors').
  {const sm::Mat4f c2s1=cameraProjection(camera,perspective,window,interocular,true),rtoc1=(c2s1*s2r).inverse();
   for(int i=0;i<4;++i)for(int j=0;j<4;++j){cam.rasterToCamera1.m[i][j]=fin(rtoc1[i][j],"projection");cam.cameraToScreen0.m[i][j]=c2s[i][j];cam.cameraToScreen1.m[i][j]=c2s1[i][j];}
   if(get<rdl::Bool>(sv,"enable_motion_blur")&&!sm::isEqual(rtoc,rtoc1))cam.motion=1;}
  // The lens: depth of field with the scene's enable_dof and the camera's dof, its radius from the f-stop.
  if(get<rdl::Bool>(sv,"enable_dof")&&get<rdl::Bool>(camera,"dof")){
   const float focalLength=perspective?get<rdl::Float>(camera,"focal"):1.0f,fstop=get<rdl::Float>(camera,"dof_aperture");
   const float worldToMm=get<rdl::Float>(sv,"scene_scale")*1000.0f,pupil=focalLength/fstop;
   cam.lensRadius=(pupil/2.0f)/worldToMm;cam.focusDistance=get<rdl::Float>(camera,"dof_focus_distance");
   cam.dof=cam.lensRadius>0.0f?1u:0u;
   if(cam.dof){
    fin(cam.lensRadius,"lens radius");fin(cam.focusDistance,"focus distance");
    // LensDistribution::update: an image's distribution, else a polygon of three sides or more, else the disk.
    if(get<rdl::Bool>(camera,"bokeh")){
     const int sides=get<rdl::Int>(camera,"bokeh_sides");const std::string image=get<rdl::String>(camera,"bokeh_image");
     cam.bokehAngle=fin(sm::deg2rad(get<rdl::Float>(camera,"bokeh_angle")),"bokeh angle");
     bool imaged=false;
     if(!image.empty()){
      lights::ImageSettings st;st.file=image;st.mapping=lights::MapPlanar;const std::string key=st.key();
      std::shared_ptr<lights::ImageData> data;bool opened=true;
      if(previous&&previous->bokehImage&&previous->bokehImage->key==key)data=previous->bokehImage;
      else try{data=lights::buildImage(st);}catch(const std::exception&){opened=false;}
      // MoonRay goes on in disk mode when the image cannot be opened (its polygon is then never built: where the
      // sides ask for one it reads no vertices; XPU2 builds the polygon).
      if(opened){need(data!=nullptr,name+": the bokeh image has no level MoonRay can sample");plan.bokehImage=data;cam.bokeh=2;cam.bokehImage=&data->view;imaged=true;}
     }
     if(!imaged&&sides>=3){cam.bokeh=1;cam.bokehSides=sides;}
    }
   }
  }
 }else if(cls=="SphericalCamera"){
  cam.type=surface::CameraSpherical;
  const float zoom=30.0f/get<rdl::Float>(camera,"focal");
  const float thetaMin=(sm::sPi/180.0f)*get<rdl::Float>(camera,"min_latitude"),thetaMax=(sm::sPi/180.0f)*get<rdl::Float>(camera,"max_latitude");
  const float thetaOfs=(sm::sPi/180.0f)*get<rdl::Float>(camera,"latitude_zoom_offset");
  const float thetaMid=0.5f*(thetaMin+thetaMax)+thetaOfs;
  cam.k[surface::SphThetaScale]=zoom*(thetaMax-thetaMin)/cam.apertureHeight;cam.k[surface::SphThetaOffset]=sm::lerp(thetaMid,thetaMin,zoom);
  float phiMin=(sm::sPi/180.0f)*get<rdl::Float>(camera,"min_longitude"),phiMax=(sm::sPi/180.0f)*get<rdl::Float>(camera,"max_longitude");
  // The legacy sphere (the default ranges) faces -x: its longitudes turn by 90 degrees.
  if(get<rdl::Float>(camera,"min_latitude")==-90.0f&&get<rdl::Float>(camera,"max_latitude")==90.0f&&
     get<rdl::Float>(camera,"min_longitude")==-180.0f&&get<rdl::Float>(camera,"max_longitude")==180.0f){phiMin=-3.0f*sm::sHalfPi;phiMax=sm::sHalfPi;}
  const float phiOfs=(sm::sPi/180.0f)*get<rdl::Float>(camera,"longitude_zoom_offset"),phiMid=0.5f*(phiMin+phiMax)+phiOfs;
  cam.k[surface::SphPhiScale]=zoom*(phiMax-phiMin)/cam.apertureWidth;cam.k[surface::SphPhiOffset]=sm::lerp(phiMid,phiMin,zoom);
  cam.flags=get<rdl::Bool>(camera,"inside_out")?unsigned(surface::SphInsideOut):0u;cam.k[surface::SphOffsetRadius]=get<rdl::Float>(camera,"offset_radius");
  for(int i=0;i<5;++i)fin(cam.k[i],"spherical setting");
 }else if(cls=="FisheyeCamera"){
  cam.type=surface::CameraFisheye;
  cam.mode=get<rdl::Int>(camera,"mapping");need(cam.mode>=0&&cam.mode<=3,name+": unknown mapping");
  const int format=get<rdl::Int>(camera,"format");need(format>=0&&format<=2,name+": unknown format");
  const float w=cam.apertureWidth,h=cam.apertureHeight;
  const float diameter=format==0?sm::min(w,h):format==1?sm::max(w,h):sm::sqrt(w*w+h*h);
  float radialScale=2.0f/diameter;radialScale/=get<rdl::Float>(camera,"zoom");
  cam.k[surface::FishRadialScale]=fin(radialScale,"zoom");
  cam.k[surface::FishHalfFov]=fin(0.5f*degreesToRadians(get<rdl::Float>(camera,"fov")),"field of view");
  // FisheyeCamera::computeFishtumImpl: what MoonRay's tessellation reads of it.
  plan.fishtum=true;
  const float fish[9]={radialScale,1.0f/(radialScale*h),float(cam.mode),w,h,cam.regionX,cam.regionY,cam.regionX+float(rg.width()),cam.regionY+float(rg.height())};
  std::copy(fish,fish+9,plan.fishtumData);
 }else if(cls=="DomeMaster3DCamera"){
  cam.type=surface::CameraDome;
  const int view=get<rdl::Int>(camera,"stereo_view");need(view>=0&&view<=2,name+": unknown stereo view");cam.mode=view;
  cam.k[surface::DomeInterocular]=view==0?0.0f:(view==1?-0.5f:0.5f)*get<rdl::Float>(camera,"stereo_interocular_distance");
  cam.k[surface::DomeWidthRecip]=1.0f/cam.apertureWidth;cam.k[surface::DomeHeightRecip]=1.0f/cam.apertureHeight;
  cam.k[surface::DomeFovH]=degreesToRadians(get<rdl::Float>(camera,"FOV_horizontal_angle"));
  cam.k[surface::DomeFovV]=degreesToRadians(get<rdl::Float>(camera,"FOV_vertical_angle"));
  cam.k[surface::DomeParallax]=get<rdl::Float>(camera,"stereo_convergence_distance");
  cam.flags=(get<rdl::Bool>(camera,"zenith_mode")?unsigned(surface::DomeZenith):0u)|(get<rdl::Bool>(camera,"flip_ray_x")?unsigned(surface::DomeFlipX):0u)|
            (get<rdl::Bool>(camera,"flip_ray_y")?unsigned(surface::DomeFlipY):0u);
  for(int i=0;i<6;++i)fin(cam.k[i],"dome setting");
 }else{
  // BakeCamera: its mode and bias here; its maps once its mesh is read (buildBakeMap).
  cam.type=surface::CameraBake;
  cam.mode=get<rdl::Int>(camera,"mode");need(cam.mode>=0&&cam.mode<=3,name+": unknown bake mode");
  cam.k[surface::BakeBias]=fin(get<rdl::Float>(camera,"bias"),"bias");
  cam.flags=get<rdl::Bool>(camera,"use_relative_bias")?unsigned(surface::BakeRelativeBias):0u;
  need(std::isfinite(get<rdl::Float>(camera,"map_factor")),name+": invalid map factor");
 }
}
// A bake camera's position map (BakeCamera::bakeUvMapsImpl over the mesh's TriMesh::bakePosMap or QuadMesh::bakePosMap,
// Mesh::rasterizeTrianglePos, interpolatePosInTri and udimxform): each face's triangles laid out by their texture
// coordinates in the camera's UDIM tile, each covered texel holding its interpolated position (w 1) and, where the bake
// reads normals, its normal. faces: MoonRay's faces in order, each three or four corners (a vertex and its corner in the
// face-vertex list; a QuadMesh's triangle repeats its last corner); quads: a QuadMesh's (triangles 0 1 3 and 2 3 1).
struct BakeFace {unsigned vertex[4];size_t corner[4];size_t face;};
inline bool udimTransform(int udim,float st[2]){
 const int minS=(udim-1001)%10,minT=(udim-1001)/10;
 if(!(st[0]>=float(minS)&&st[0]<=float(minS+1)&&st[1]>=float(minT)&&st[1]<=float(minT+1)))return false;
 st[0]-=float(minS);st[1]-=float(minT);
 st[0]=std::min(std::max(st[0],0.0f),1.0f);st[1]=std::min(std::max(st[1],0.0f),1.0f);return true;
}
inline void bakeTriangle(int width,int height,const float* a,const float* b,const float* c,V3 pA,V3 pB,V3 pC,const V3* nA,const V3* nB,const V3* nC,
                         std::vector<float>& pos,std::vector<float>* nrm){
 const float lo[2]={std::min(a[0],std::min(b[0],c[0])),std::min(a[1],std::min(b[1],c[1]))},hi[2]={std::max(a[0],std::max(b[0],c[0])),std::max(a[1],std::max(b[1],c[1]))};
 const int x0=int(std::floor(lo[0]*float(width-1))),y0=int(std::floor(lo[1]*float(height-1))),x1=int(std::ceil(hi[0]*float(width-1))),y1=int(std::ceil(hi[1]*float(height-1)));
 for(int y=y0;y<=y1;++y){const float t=float(y)/float(height-1);
  for(int x=x0;x<=x1;++x){const float p[2]={float(x)/float(width-1),t};
   // Inside all three edges' half planes (interpolatePosInTri), then barycentric by areas.
   auto side=[&](const float* e0,const float* e1,const float* mid0,const float* mid1){
    const float nx=e0[1]-e1[1],ny=e1[0]-e0[0],mx=p[0]-(mid0[0]+mid1[0])*0.5f,my=p[1]-(mid0[1]+mid1[1])*0.5f;return nx*mx+ny*my<0.0f;};
   const bool ab=side(a,b,a,b),bc=side(b,c,b,c),ca=side(c,a,a,c);
   if(!(ab==bc&&bc==ca))continue;
   const float area=std::abs((c[0]-b[0])*(a[1]-b[1])-(a[0]-b[0])*(c[1]-b[1]));
   if(!(area>0))continue;
   const float areaPBC=std::abs((p[0]-b[0])*(c[1]-b[1])-(c[0]-b[0])*(p[1]-b[1]));
   const float areaAPC=std::abs((c[0]-p[0])*(a[1]-p[1])-(a[0]-p[0])*(c[1]-p[1]));
   const float inv=1.0f/area;float u=areaPBC*inv,v=areaAPC*inv,w=1.0f-u-v;
   u=std::min(std::max(u,0.0f),1.0f);v=std::min(std::max(v,0.0f),1.0f);w=std::min(std::max(w,0.0f),1.0f);
   const size_t at=size_t(y)*size_t(width)+size_t(x);
   const V3 P=surface::add(surface::add(surface::mul(pA,u),surface::mul(pB,v)),surface::mul(pC,w));
   pos[at*4]=P.x;pos[at*4+1]=P.y;pos[at*4+2]=P.z;pos[at*4+3]=1.0f;
   if(nrm){const V3 N=surface::add(surface::add(surface::mul(*nA,u),surface::mul(*nB,v)),surface::mul(*nC,w));(*nrm)[at*3]=N.x;(*nrm)[at*3+1]=N.y;(*nrm)[at*3+2]=N.z;}
  }}
}
// The bake camera's maps over a mesh: positions (render space for MoonRay, world space here, as XPU2 renders), each
// face corner's texture coordinates (uv) and normal (normals; none: MoonRay's fallback, cross(p1 - p2, p3 - p2) of each
// triangle, its normal reversed, a QuadMesh's first triangle's corners 1 and 3 taking its second triangle's).
inline void buildBakeMap(Plan::CameraMapData& out,int width,int height,int udim,const std::vector<V3>& positions,const std::vector<BakeFace>& faces,bool quads,
                         const std::function<void(const BakeFace&,int,float*)>& uv,const std::function<bool(const BakeFace&,int,V3&)>& normal,bool needNrm){
 out.positions.assign(size_t(width)*size_t(height)*4,0.0f);
 if(needNrm)out.normals.assign(size_t(width)*size_t(height)*3,0.0f);else out.normals.clear();
 auto faceNormal=[&](unsigned i1,unsigned i2,unsigned i3){V3 n{0,0,0};surface::normalize(surface::cross(surface::add(positions[i1],surface::mul(positions[i2],-1.0f)),surface::add(positions[i3],surface::mul(positions[i2],-1.0f))),n);return n;};
 for(const auto& f:faces){
  const int n=quads?4:3;float st[4][2];bool inTile=true;
  for(int k=0;k<n;++k){uv(f,k,st[k]);inTile=udimTransform(udim,st[k])&&inTile;}
  if(!inTile)continue;
  V3 nr[4]={{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
  if(needNrm){bool given=true;for(int k=0;k<n;++k)given=normal(f,k,nr[k])&&given;
   if(!given){
    if(quads){nr[0]=nr[1]=nr[3]=faceNormal(f.vertex[0],f.vertex[1],f.vertex[3]);nr[2]=nr[3]=nr[1]=faceNormal(f.vertex[2],f.vertex[3],f.vertex[1]);}
    else nr[0]=nr[1]=nr[2]=faceNormal(f.vertex[0],f.vertex[1],f.vertex[2]);}}
  std::vector<float>* nrm=needNrm?&out.normals:nullptr;
  auto tri=[&](int i,int j,int k){bakeTriangle(width,height,st[i],st[j],st[k],positions[f.vertex[i]],positions[f.vertex[j]],positions[f.vertex[k]],&nr[i],&nr[j],&nr[k],out.positions,nrm);};
  if(quads){tri(0,1,3);tri(2,3,1);}else tri(0,1,2);
 }
}
// MoonRay's subdivision surfaces (RdlMeshGeometry with is_subd, its OpenSubdivMesh), tessellated by MoonRay's own code in
// xpu2_geom.dll (xpu2_geom.h): the very quads MoonRay intersects, each tessellated vertex's limit normal, texture
// coordinates and derivatives (shaded as OpenSubdivMesh::postIntersect, shade::hitState). The control mesh goes in in
// MoonRay's render space (the camera's), as the tessellation's camera frustums are; the quads come back to world space.
struct SubdLibrary{X2SubdTessellateFn tessellate=nullptr;X2SubdFreeFn release=nullptr;X2VdbReadFn vdbRead=nullptr;X2VdbFreeFn vdbFree=nullptr;
 X2VdbHistogramFn vdbHistogram=nullptr;X2VdbHistogramFreeFn vdbHistogramFree=nullptr;X2VdbHistogramMappedFn vdbHistogramMapped=nullptr;
 X2VdbVelocityFn vdbVelocity=nullptr;X2PolyTessellateFn polyTessellate=nullptr;X2PolyFreeFn polyRelease=nullptr;};
inline const SubdLibrary& subdLibrary(){
 static const SubdLibrary lib=[]{SubdLibrary l;wchar_t path[MAX_PATH];const DWORD n=GetModuleFileNameW(nullptr,path,MAX_PATH);
  if(n&&n<MAX_PATH)if(HMODULE h=LoadLibraryW((fs::path(path).parent_path()/"xpu2_geom.dll").c_str())){
   l.tessellate=reinterpret_cast<X2SubdTessellateFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2SubdTessellate")));
   l.release=reinterpret_cast<X2SubdFreeFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2SubdFree")));
   l.vdbRead=reinterpret_cast<X2VdbReadFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbRead")));
   l.vdbFree=reinterpret_cast<X2VdbFreeFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbFree")));
   l.vdbHistogram=reinterpret_cast<X2VdbHistogramFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbHistogram")));
   l.vdbHistogramFree=reinterpret_cast<X2VdbHistogramFreeFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbHistogramFree")));
   l.vdbHistogramMapped=reinterpret_cast<X2VdbHistogramMappedFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbHistogramMapped")));
   l.vdbVelocity=reinterpret_cast<X2VdbVelocityFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2VdbVelocity")));
   l.polyTessellate=reinterpret_cast<X2PolyTessellateFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2PolyTessellate")));
   l.polyRelease=reinterpret_cast<X2PolyFreeFn>(reinterpret_cast<void*>(GetProcAddress(h,"x2PolyFree")));}
  return l;}();
 return lib;
}
// A tessellated attribute (xpu2_geom.h's X2SubdAttributeOut): its XPU2 key and components, kind (1 constant, 2 a
// quad's, 3 a vertex's, 4 a quad corner's) and values.
struct SubdAttribute{unsigned key=0,components=1;int kind=1;std::vector<float> values;};
struct SubdSurface{std::vector<V3> positions,normals,dpds,dpdt;std::vector<float> st;std::vector<unsigned> quads;std::vector<int> controlFace;
 std::vector<SubdAttribute> attributes;};
inline void primitiveAttributes(const rdl::SceneObject& o,struct MeshData& m,size_t uniformCount,size_t vertexCount,size_t varyingCount,size_t faceVaryingCount,
                                std::vector<std::pair<std::string,unsigned>>* named);
// scene_rdl2 xfmPoint with an Xform3f made of a Mat4d: (x vx + (y vy + z vz)) + p in floats.
inline V3 xformPointF(const float m[4][3],const rdl::Vec3f& p){
 V3 r;for(int c=0;c<3;++c){const float yz=p.y*m[1][c]+p.z*m[2][c];(&r.x)[c]=(p.x*m[0][c]+yz)+m[3][c];}return r;}
// A displacement for the tessellation (SubdDisplacement): the DLL's callback and its user data.
struct SubdDisplaceCall{X2SubdDisplaceFn fn=nullptr;void* user=nullptr;X2SubdDisplaceAttrFn with=nullptr;};
// A mesh as the DLL takes it (X2SubdInput, render space), handed to call with its primitive attributes (control, in
// the input's order) and world to render.
template<class Call> inline auto withMeshInput(const rdl::SceneObject& o,const rdl::Layer& layer,const Plan& plan,const rdl::Mat4d& objectToWorld,
                                               const std::vector<V3>* worldVertices,const SubdDisplaceCall& displaceCall,Call&& call,bool reference=false){
 const auto& v=o.get<rdl::Vec3fVector>("vertex_list");const auto& counts=o.get<rdl::IntVector>("face_vertex_count");const auto& indices=o.get<rdl::IntVector>("vertices_by_index");
 need(!v.empty()&&!counts.empty()&&!indices.empty(),o.getName()+": an empty mesh");
 // Procedural::addPrimitive's prim2render (node_xform, then world to render), as floats.
 rdl::Mat4d w2r;for(int i=0;i<4;++i)for(int j=0;j<4;++j)w2r[i][j]=plan.world2render[4*i+j];
 // A shared primitive stays in its own space (its prim to render is one).
 const rdl::Mat4d identity(scene_rdl2::math::one);
 const rdl::Mat4d p2r=reference?identity:objectToWorld*w2r;float xf[4][3];for(int i=0;i<4;++i)for(int c=0;c<3;++c)xf[i][c]=float(p2r[i][c]);
 std::vector<float> vertices;vertices.reserve(v.size()*3);
 if(worldVertices){
  // A time step's vertices (meshMotionSteps, world space) to render space.
  need(worldVertices->size()==v.size(),o.getName()+": a time step's vertices do not match the mesh");
  float wr[4][3];for(int i=0;i<4;++i)for(int c=0;c<3;++c)wr[i][c]=float(w2r[i][c]);
  for(const auto& p:*worldVertices){const V3 r=xformPointF(wr,rdl::Vec3f(p.x,p.y,p.z));vertices.insert(vertices.end(),{r.x,r.y,r.z});}}
 else for(const auto& p:v){X2_NEED(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),o.getName()+": invalid vertex");const V3 r=xformPointF(xf,p);vertices.insert(vertices.end(),{r.x,r.y,r.z});}
 // Normals as the tessellation's transformAttributes takes them (normals, the inverse transpose), uvs as they are.
 std::vector<float> normals,uvs;
 {const auto& nl=o.get<rdl::Vec3fVector>("normal_list");
  if(!nl.empty()){double M[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)M[i][j]=p2r[i][j];
   const double det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
   double inv[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,d=(i+2)%3;inv[i][j]=(M[a][cc]*M[b][d]-M[a][d]*M[b][cc])/det;}
   for(const auto& n:nl)normals.insert(normals.end(),{float(n.x*inv[0][0]+n.y*inv[0][1]+n.z*inv[0][2]),float(n.x*inv[1][0]+n.y*inv[1][1]+n.z*inv[1][2]),float(n.x*inv[2][0]+n.y*inv[2][1]+n.z*inv[2][2])});}
  for(const auto& u:o.get<rdl::Vec2fVector>("uv_list"))uvs.insert(uvs.end(),{u.x,u.y});}
 // Each face's layer assignment and part (createPerFaceAssignmentId, the procedural's faceToPart).
 const auto* geom=o.asA<rdl::Geometry>();const int whole=layer.getAssignmentId(geom,"");
 std::vector<int> assignment(counts.size(),whole),faceToPart(counts.size(),0);
 const auto& parts=get<rdl::StringVector>(o,"part_list");const auto& partCounts=get<rdl::IntVector>(o,"part_face_count_list");const auto& partFaces=get<rdl::IntVector>(o,"part_face_indices");
 {size_t at=0;for(size_t p=0;p<partCounts.size()&&p<parts.size();++p){const int id=layer.getAssignmentId(geom,parts[p]);
   for(int k=0;k<partCounts[p]&&at<partFaces.size();++k,++at){const int f=partFaces[at];if(f>=0&&size_t(f)<counts.size()){assignment[size_t(f)]=id;faceToPart[size_t(f)]=int(p);}}}}
 const auto& creaseIndices=get<rdl::IntVector>(o,"subd_crease_indices");const auto& creaseSharp=get<rdl::FloatVector>(o,"subd_crease_sharpnesses");
 const auto& cornerIndices=get<rdl::IntVector>(o,"subd_corner_indices");const auto& cornerSharp=get<rdl::FloatVector>(o,"subd_corner_sharpnesses");
 X2SubdInput in{};
 in.vertices=vertices.data();in.vertexCount=unsigned(v.size());
 in.faceVertexCount=counts.data();in.faceCount=unsigned(counts.size());in.indices=indices.data();in.indexCount=unsigned(indices.size());
 in.uvs=uvs.empty()?nullptr:uvs.data();in.uvCount=unsigned(uvs.size()/2);
 in.normals=normals.empty()?nullptr:normals.data();in.normalCount=unsigned(normals.size()/3);
 in.creaseIndices=creaseIndices.data();in.creaseIndexCount=unsigned(creaseIndices.size());in.creaseSharpnesses=creaseSharp.data();in.creaseSharpnessCount=unsigned(creaseSharp.size());
 in.cornerIndices=cornerIndices.data();in.cornerIndexCount=unsigned(cornerIndices.size());in.cornerSharpnesses=cornerSharp.data();in.cornerSharpnessCount=unsigned(cornerSharp.size());
 in.faceToPart=parts.empty()?nullptr:faceToPart.data();in.partCount=unsigned(parts.size());in.assignmentIds=assignment.data();
 in.scheme=get<rdl::Int>(o,"subd_scheme");in.boundary=get<rdl::Int>(o,"subd_boundary");in.fvarLinear=get<rdl::Int>(o,"subd_fvar_linear");
 in.meshResolution=int(get<rdl::Float>(o,"mesh_resolution"));  // RdlMeshGeometry: int meshResolution = its float
 in.adaptiveError=get<rdl::Float>(o,"adaptive_error");
 // The coarse first pass: a quarter of the resolution, four times the screen error (coarse_geometry.h).
 in.meshResolution=coarseResolution(in.meshResolution);in.adaptiveError=coarseError(in.adaptiveError);
 in.singleSided=get<rdl::Int>(o,"side_type")==1?1:0;in.reverseNormals=get<rdl::Bool>(o,"reverse_normals")?1:0;in.leftHanded=0;
 in.layer=&layer;in.geometry=geom;
 in.frustumCount=unsigned(plan.frustumViewports.size()/4);in.frustumC2S=plan.frustumC2S.data();in.frustumPlanes=plan.frustumPlanes.data();in.frustumViewports=plan.frustumViewports.data();
 in.hasFishtum=plan.fishtum?1:0;for(int i=0;i<9;++i)in.fishtum[i]=plan.fishtumData[i];
 for(int i=0;i<16;++i)in.world2render[i]=plan.world2render[i];
 // A dicing camera (the scene's, else the geometry's; GeometryManager::getDicingCameraFrustums): a PerspectiveCamera's
 // own frustums over the render region, their clip planes moved into the main camera's render space by its node
 // transform, and its world to render; any other camera class leaves the main camera's (MoonRay warns).
 std::vector<float> dicingC2S,dicingPlanes;std::vector<int> dicingViewports;
 {const rdl::SceneVariables& sv=o.getSceneClass().getSceneContext()->getSceneVariables();
  const rdl::SceneObject* dc=get<rdl::SceneObject*>(sv,"dicing_camera");if(!dc)dc=get<rdl::SceneObject*>(o,"dicing_camera");
  if(dc&&dc->getSceneClass().getName()=="PerspectiveCamera"){namespace sm=scene_rdl2::math;
   const auto ap=sv.getRezedApertureWindow(),rg=sv.getRezedRegionWindow();
   const float apW=float(ap.width()),apH=float(ap.height()),regionX=float(rg.min().x-ap.min().x),regionY=float(rg.min().y-ap.min().y),invAspect=apH/apW;
   float interocular=0.0f;const int view=get<rdl::Int>(*dc,"stereo_view");if(view!=0)interocular=(view==1?0.5f:-0.5f)*get<rdl::Float>(*dc,"stereo_interocular_distance");
   const float offX=2*regionX/apW,offY=2*regionY/apH,wR=2*float(rg.width())/apW,hR=2*float(rg.height())/apH;
   const std::array<float,4> region{-1.0f+offX,invAspect*(-1.0f+offY),-1.0f+offX+wR,invAspect*(-1.0f+offY+hR)};
   const rdl::Mat4d dicingRender2World=dc->get<rdl::Mat4d>("node_xform");
   const rdl::Mat4d moveInverse=(dicingRender2World*w2r).inverse();
   for(int close=0;close<2;++close){const sm::Mat4f c=cameraProjection(*dc,true,region,interocular,close==1);
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)dicingC2S.push_back(c[i][j]);
    const float planes[6][4]={{c[0][3]-c[0][0],c[1][3]-c[1][0],c[2][3]-c[2][0],c[3][3]-c[3][0]},{c[0][3]+c[0][0],c[1][3]+c[1][0],c[2][3]+c[2][0],c[3][3]+c[3][0]},
                              {c[0][3]+c[0][1],c[1][3]+c[1][1],c[2][3]+c[2][1],c[3][3]+c[3][1]},{c[0][3]-c[0][1],c[1][3]-c[1][1],c[2][3]-c[2][1],c[3][3]-c[3][1]},
                              {c[0][3]-c[0][2],c[1][3]-c[1][2],c[2][3]-c[2][2],c[3][3]-c[3][2]},{c[0][3]+c[0][2],c[1][3]+c[1][2],c[2][3]+c[2][2],c[3][3]+c[3][2]}};
    // Frustum::transformClipPlanes: each plane times the transform's inverse (as a column).
    for(const auto& pl:planes)for(int i=0;i<4;++i)dicingPlanes.push_back(float(moveInverse[i][0]*pl[0]+moveInverse[i][1]*pl[1]+moveInverse[i][2]*pl[2]+moveInverse[i][3]*pl[3]));
    const int vx=int(regionX),vy=int(regionY);dicingViewports.insert(dicingViewports.end(),{vx,vy,vx+int(rg.width()),vy+int(rg.height())});}
   const rdl::Mat4d dicingWorld2Render=dicingRender2World.inverse();for(int i=0;i<4;++i)for(int j=0;j<4;++j)in.world2render[4*i+j]=dicingWorld2Render[i][j];
   in.frustumCount=2;in.frustumC2S=dicingC2S.data();in.frustumPlanes=dicingPlanes.data();in.frustumViewports=dicingViewports.data();in.hasFishtum=0;}}
 in.enableDisplacement=displaceCall.fn||displaceCall.with?1:0;in.displace=displaceCall.fn;in.displaceUser=displaceCall.user;in.displaceWith=displaceCall.with;
 // Primitive attributes (UserData, processArbitraryData at the control mesh's rate counts: a face uniform, a vertex
 // varying or vertex, a face-vertex face varying), interpolated by MoonRay's tessellation.
 MeshData control;std::vector<std::pair<std::string,unsigned>> named;
 primitiveAttributes(o,control,counts.size(),v.size(),v.size(),indices.size(),&named);
 std::vector<X2SubdAttribute> attrs;
 for(size_t k=0;k<control.prims.size();++k){const auto& a=control.prims[k];const unsigned type=named[k].second;
  need(type!=shade::PrimInt||a.rate==shade::PrimConstant||a.rate==shade::PrimUniform,o.getName()+": an int primitive attribute that varies over a subdivision surface");
  const int rate=a.rate==shade::PrimConstant?1:a.rate==shade::PrimUniform?2:a.rate==shade::PrimVarying?3:a.rate==shade::PrimFaceVarying?4:5;
  const size_t n=a.rate==shade::PrimConstant?1:a.rate==shade::PrimUniform?counts.size():a.rate==shade::PrimFaceVarying?indices.size():v.size();
  attrs.push_back(X2SubdAttribute{named[k].first.c_str(),int(type),rate,unsigned(n),control.primData.data()+a.offset});}
 in.attributes=attrs.empty()?nullptr:attrs.data();in.attributeCount=unsigned(attrs.size());
 in.reference=reference?1:0;
 return call(in,control,reference?identity:w2r);
}
// The subdivision's share of preparation (the session's log): the DLL's tessellations, and its displacement callback's
// calls and their time summed over the threads that make them (more than the tessellations' own: they run in parallel).
struct SubdTimes{std::atomic<long long> meshes{0},tessellateNs{0},displaceCalls{0},displaceNs{0},first{0},last{0},threads{0},round{1};
 void reset(){meshes=0;tessellateNs=0;displaceCalls=0;displaceNs=0;first=0;last=0;threads=0;++round;}
 // The displacements' span (the first call's start to the last one's end) and the threads that made them.
 void seen(long long began,long long ended){long long f=first.load();while((f==0||began<f)&&!first.compare_exchange_weak(f,began)){}
  long long l=last.load();while(ended>l&&!last.compare_exchange_weak(l,ended)){}
  thread_local long long counted=0;if(counted!=round.load()){counted=round.load();threads++;}}};
inline SubdTimes& subdTimes(){static SubdTimes t;return t;}
inline long long nowNs(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
// A loop over [0, n) in chunks on the machine's threads, each element independent (serial below 64K elements).
template<class F> inline void parallelRange(size_t n,const F& f){
 const unsigned threads=std::max(1u,std::min(32u,std::thread::hardware_concurrency()));
 if(n<65536||threads<2){f(size_t(0),n);return;}
 const size_t chunk=(n+threads-1)/threads;std::vector<std::thread> pool;std::mutex failing;std::exception_ptr failed;
 // A failure goes to the caller once every thread has left: thrown past threads still running, it ended the process.
 auto part=[&](size_t a,size_t b){try{f(a,b);}catch(...){std::lock_guard<std::mutex> lock(failing);if(!failed)failed=std::current_exception();}};
 size_t begun=chunk;
 try{for(unsigned t=1;t<threads;++t){const size_t a=size_t(t)*chunk,b=std::min(n,a+chunk);if(a<b){pool.emplace_back([&part,a,b]{part(a,b);});begun=b;}}}catch(const std::system_error&){}
 part(size_t(0),std::min(n,chunk));if(begun<n)part(begun,n);
 for(auto& t:pool)t.join();if(failed)std::rethrow_exception(failed);}
// A part of preparation timed while it is in scope (PrepLaps::parts).
struct PrepPart{const char* what;long long began=nowNs();~PrepPart();};
inline std::shared_ptr<const SubdSurface> tessellateSubdivision(const rdl::SceneObject& o,const rdl::Layer& layer,const Plan& plan,const rdl::Mat4d& objectToWorld,
                                                              const std::vector<V3>* worldVertices=nullptr,const SubdDisplaceCall& displaceCall={},bool reference=false){
 const auto& lib=subdLibrary();need(lib.tessellate&&lib.release,o.getName()+": MoonRay's subdivision (xpu2_geom.dll) did not load");
 return withMeshInput(o,layer,plan,objectToWorld,worldVertices,displaceCall,[&](X2SubdInput& in,const MeshData& control,const rdl::Mat4d& w2r){
 X2SubdOutput out{};char error[512]={0};
 const long long began=nowNs();
 const int made=lib.tessellate(&in,&out,error,sizeof error);
 subdTimes().meshes++;subdTimes().tessellateNs+=nowNs()-began;
 need(made!=0,o.getName()+": MoonRay's subdivision failed: "+std::string(error));
 PrepPart converting{"tessellation's output read"};
 auto s=std::make_shared<SubdSurface>();
 // Back to world space: render to world is the camera's node transform (rigid: its vectors turn alone).
 const rdl::Mat4d r2w=w2r.inverse();
 auto point=[&](const float* p){return V3{float(p[0]*r2w[0][0]+p[1]*r2w[1][0]+p[2]*r2w[2][0]+r2w[3][0]),float(p[0]*r2w[0][1]+p[1]*r2w[1][1]+p[2]*r2w[2][1]+r2w[3][1]),
                                          float(p[0]*r2w[0][2]+p[1]*r2w[1][2]+p[2]*r2w[2][2]+r2w[3][2])};};
 auto vec=[&](const float* p){return V3{float(p[0]*r2w[0][0]+p[1]*r2w[1][0]+p[2]*r2w[2][0]),float(p[0]*r2w[0][1]+p[1]*r2w[1][1]+p[2]*r2w[2][1]),
                                        float(p[0]*r2w[0][2]+p[1]*r2w[1][2]+p[2]*r2w[2][2])};};
 try{
  const size_t n=out.vertexCount;s->positions.resize(n);s->normals.resize(n);s->dpds.resize(n);s->dpdt.resize(n);
  parallelRange(n,[&](size_t a,size_t b){for(size_t i=a;i<b;++i){s->positions[i]=point(out.vertices+3*i);s->normals[i]=vec(out.normals+3*i);
   s->dpds[i]=vec(out.dpds+3*i);s->dpdt[i]=vec(out.dpdt+3*i);}});
  s->st.assign(out.st,out.st+2*size_t(out.vertexCount));s->quads.assign(out.indices,out.indices+4*size_t(out.quadCount));
  s->controlFace.assign(out.controlFace,out.controlFace+out.quadCount);
  for(unsigned k=0;k<out.attributeCount;++k){const auto& a=control.prims[k];const auto& r=out.attributes[k];
   SubdAttribute t;t.key=a.key;t.components=a.components;t.kind=r.kind;t.values.assign(r.values,r.values+size_t(r.count)*a.components);s->attributes.push_back(std::move(t));}
 }catch(...){lib.release(&out);throw;}
 lib.release(&out);
 for(unsigned q:s->quads)X2_NEED(q<s->positions.size(),o.getName()+": a tessellated index out of bounds");
 return s;},reference);
}
// A displaced polygon mesh as MoonRay's PolyMesh tessellates and displaces it (xpu2_geom.dll's x2PolyTessellate, its
// QuadMesh or TriMesh), world space: its faces as hit (four corners, a three-cornered face's fourth its third), each
// one's input face, its vertices at each time step, its shading normals (3: a vertex's, at two time steps when it
// moves; 4: a face corner's), each face corner's st and its primitive attributes (SubdAttribute's kinds, 2 a face's).
struct PolySurface{std::vector<V3> positions,normals,normals1;std::vector<std::vector<V3>> steps;int normalKind=0;std::vector<unsigned> faces;
 std::vector<int> controlFace;std::vector<float> st;bool fourCorners=true;std::vector<SubdAttribute> attributes;};
inline std::shared_ptr<const PolySurface> tessellatePolygons(const rdl::SceneObject& o,const rdl::Layer& layer,const Plan& plan,const rdl::Mat4d& objectToWorld,
                                                           const std::vector<std::vector<V3>>* worldSteps,const SubdDisplaceCall& displaceCall,bool reference=false){
 const auto& lib=subdLibrary();need(lib.polyTessellate&&lib.polyRelease,o.getName()+": MoonRay's polygon tessellation (xpu2_geom.dll) did not load");
 return withMeshInput(o,layer,plan,objectToWorld,worldSteps?&(*worldSteps)[0]:nullptr,displaceCall,[&](X2SubdInput& in,const MeshData& control,const rdl::Mat4d& w2r){
  X2PolyInput pin{};pin.mesh=in;pin.smoothNormal=get<rdl::Bool>(o,"smooth_normal")?1:0;pin.stepCount=1;
  // The later time steps (meshMotionSteps, world space) in render space, as the first.
  std::vector<float> later;
  if(worldSteps&&worldSteps->size()>1){float wr[4][3];for(int i=0;i<4;++i)for(int c=0;c<3;++c)wr[i][c]=float(w2r[i][c]);
   for(size_t k=1;k<worldSteps->size();++k){need((*worldSteps)[k].size()==in.vertexCount,o.getName()+": a time step's vertices do not match the mesh");
    for(const auto& p:(*worldSteps)[k]){const V3 r=xformPointF(wr,rdl::Vec3f(p.x,p.y,p.z));later.insert(later.end(),{r.x,r.y,r.z});}}
   pin.stepCount=unsigned(worldSteps->size());pin.stepVertices=later.data();}
  X2PolyOutput out{};char error[512]={0};
  const long long began=nowNs();
  const int made=lib.polyTessellate(&pin,&out,error,sizeof error);
  subdTimes().meshes++;subdTimes().tessellateNs+=nowNs()-began;
  need(made!=0,o.getName()+": MoonRay's polygon tessellation failed: "+std::string(error));
  PrepPart converting{"tessellation's output read"};
  auto s=std::make_shared<PolySurface>();
  // Back to world space: render to world is the camera's node transform (rigid: its vectors turn alone).
  const rdl::Mat4d r2w=w2r.inverse();
  auto point=[&](const float* p){return V3{float(p[0]*r2w[0][0]+p[1]*r2w[1][0]+p[2]*r2w[2][0]+r2w[3][0]),float(p[0]*r2w[0][1]+p[1]*r2w[1][1]+p[2]*r2w[2][1]+r2w[3][1]),
                                           float(p[0]*r2w[0][2]+p[1]*r2w[1][2]+p[2]*r2w[2][2]+r2w[3][2])};};
  auto vec=[&](const float* p){return V3{float(p[0]*r2w[0][0]+p[1]*r2w[1][0]+p[2]*r2w[2][0]),float(p[0]*r2w[0][1]+p[1]*r2w[1][1]+p[2]*r2w[2][1]),
                                         float(p[0]*r2w[0][2]+p[1]*r2w[1][2]+p[2]*r2w[2][2])};};
  try{
   // Each element on its own (parallelRange): the same values as one after another.
   const size_t n=out.vertexCount;s->positions.resize(n);
   parallelRange(n,[&](size_t a,size_t b){for(size_t i=a;i<b;++i)s->positions[i]=point(out.vertices+3*i);});
   for(unsigned t=1;t<out.stepCount;++t){std::vector<V3> step(n);
    parallelRange(n,[&](size_t a,size_t b){for(size_t i=a;i<b;++i)step[i]=point(out.vertices+3*(size_t(t)*n+i));});s->steps.push_back(std::move(step));}
   s->normalKind=out.normalKind;
   if(out.normalKind==3){s->normals.resize(n);parallelRange(n,[&](size_t a,size_t b){for(size_t i=a;i<b;++i)s->normals[i]=vec(out.normals+3*i);});
    if(out.normalSteps>1){s->normals1.resize(n);parallelRange(n,[&](size_t a,size_t b){for(size_t i=a;i<b;++i)s->normals1[i]=vec(out.normals+3*(n+i));});}}
   else if(out.normalKind==4){const size_t c=size_t(out.faceCount)*4;s->normals.resize(c);
    parallelRange(c,[&](size_t a,size_t b){for(size_t i=a;i<b;++i)s->normals[i]=vec(out.normals+3*i);});}
   s->faces.assign(out.indices,out.indices+4*size_t(out.faceCount));s->controlFace.assign(out.controlFace,out.controlFace+out.faceCount);
   s->st.assign(out.st,out.st+8*size_t(out.faceCount));s->fourCorners=out.tessellated||out.quadBase;
   for(unsigned k=0;k<out.attributeCount;++k){const auto& a=control.prims[k];const auto& r=out.attributes[k];
    SubdAttribute t;t.key=a.key;t.components=a.components;t.kind=r.kind;if(r.values)t.values.assign(r.values,r.values+size_t(r.count)*a.components);s->attributes.push_back(std::move(t));}
  }catch(...){lib.polyRelease(&out);throw;}
  lib.polyRelease(&out);
  for(unsigned q:s->faces)X2_NEED(q<s->positions.size(),o.getName()+": a tessellated index out of bounds");
  return s;},reference);
}
// An instance of a prototype tessellated in its own space (MoonRay's shared primitive): its vertices, shading normals
// (by the inverse transpose) and position derivatives moved by the instance's transform; a moving one at the time
// steps instanceMotionSteps takes (17 when it turns), shaded between them (stepShading) with its normals and
// derivatives at each; its object space the instance's.
inline void instanceSurface(MeshData& m,const struct RenderItem& item,const MotionParams& mp);
// A part of a displaced polygon mesh as the path tracer's mesh: each face's triangles (0 1 3 and, unless its last two
// corners are one vertex, 2 3 1, as MoonRay's quads intersect), st, normals and attributes at their corners.
inline std::shared_ptr<MeshData> polygonMesh(const PolySurface& s,const std::vector<unsigned char>* faceMask,const std::string& name,
                                             std::vector<std::shared_ptr<const ix::Buffer>> signature,const rdl::Mat4d& objectToWorld,bool polyVertices){
 auto m=std::make_shared<MeshData>();m->name=name;m->signature=std::move(signature);
 {const auto w=objectToWorld.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)m->worldToObject[i]=float(rows[i]);}
 m->positions=s.positions;m->steps=s.steps;
 if(s.normalKind==3){m->normals=s.normals;m->normals1=s.normals1;m->normalRate=shade::RateVertex;}
 static const unsigned corners[2][3]={{0,1,3},{2,3,1}};
 std::vector<std::pair<size_t,unsigned>> made;  // each triangle's face and half
 // Which halves of each face make a triangle (its face kept by the mask, a half of some area), each face on its own;
 // then each face's triangles written where the ones before it end: the order one face after another gives.
 const size_t faceCount=s.controlFace.size();std::vector<unsigned char> halves(faceCount,0);
 parallelRange(faceCount,[&](size_t from,size_t to){for(size_t f=from;f<to;++f){
  const int cf=s.controlFace[f];if(faceMask&&(cf<0||size_t(cf)>=faceMask->size()||!(*faceMask)[size_t(cf)]))continue;
  const unsigned* c=s.faces.data()+4*f;
  for(unsigned half=0;half<2;++half){
   if(half==1&&c[2]==c[3])continue;
   const V3 a=m->positions[c[corners[half][0]]],b=m->positions[c[corners[half][1]]],cc=m->positions[c[corners[half][2]]];V3 n;
   if(surface::normalize(surface::cross(surface::add(b,surface::mul(a,-1)),surface::add(cc,surface::mul(a,-1))),n))halves[f]|=1u<<half;}}});
 std::vector<size_t> first(faceCount+1,0);
 for(size_t f=0;f<faceCount;++f)first[f+1]=first[f]+(halves[f]&1u)+(halves[f]>>1&1u);
 const size_t count=first[faceCount];
 m->triangles.resize(count);made.resize(count);m->uvs.resize(6*count);if(s.normalKind==4)m->normals.resize(3*count);if(polyVertices)m->polyVertices.resize(4*count);
 parallelRange(faceCount,[&](size_t from,size_t to){for(size_t f=from;f<to;++f){size_t k=first[f];const unsigned* c=s.faces.data()+4*f;
  for(unsigned half=0;half<2;++half){if(!(halves[f]>>half&1u))continue;
   m->triangles[k]={c[corners[half][0]],c[corners[half][1]],c[corners[half][2]]};made[k]={f,half};
   if(polyVertices){unsigned* p=m->polyVertices.data()+4*k;p[0]=c[0];p[1]=c[1];p[2]=c[2];p[3]=s.fourCorners?c[3]:~0u;}
   for(unsigned j=0;j<3;++j){const unsigned corner=corners[half][j];m->uvs[6*k+2*j]=s.st[8*f+2*corner];m->uvs[6*k+2*j+1]=s.st[8*f+2*corner+1];
    if(s.normalKind==4)m->normals[3*k+j]=s.normals[4*f+corner];}
   ++k;}}});
 need(!m->triangles.empty(),name+": no triangle of any area");
 m->uvRate=shade::RateFaceVarying;if(s.normalKind==4)m->normalRate=shade::RateFaceVarying;
 for(const auto& a:s.attributes){if(a.kind==0)continue;const unsigned n=a.components;std::vector<float> values;unsigned rate=shade::PrimConstant;
  if(a.kind==1)values=a.values;
  else if(a.kind==2){rate=shade::PrimUniform;for(const auto& t:made)values.insert(values.end(),a.values.begin()+std::ptrdiff_t(t.first*n),a.values.begin()+std::ptrdiff_t((t.first+1)*n));}
  else{rate=shade::PrimFaceVarying;
   for(const auto& t:made)for(unsigned k:corners[t.second])values.insert(values.end(),a.values.begin()+std::ptrdiff_t((t.first*4+k)*n),a.values.begin()+std::ptrdiff_t((t.first*4+k+1)*n));}
  m->prims.push_back(shade::PrimAttr{a.key,n,rate,unsigned(m->primData.size())});m->primData.insert(m->primData.end(),values.begin(),values.end());}
 return m;
}
// A part of a subdivision surface as the path tracer's mesh: its quads' triangles (0 1 3, 2 3 1, as MoonRay's quads
// intersect), their vertices' limit normals, texture coordinates and derivatives, the quads as polyvertices.
inline std::shared_ptr<MeshData> subdivisionMesh(const SubdSurface& s,const std::vector<unsigned char>* faceMask,const std::string& name,
                                                 std::vector<std::shared_ptr<const ix::Buffer>> signature,const rdl::Mat4d& objectToWorld,bool polyVertices){
 auto m=std::make_shared<MeshData>();m->name=name;m->signature=std::move(signature);
 {const auto w=objectToWorld.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)m->worldToObject[i]=float(rows[i]);}
 m->positions=s.positions;m->normals=s.normals;m->normalRate=shade::RateVertex;m->uvs=s.st;m->uvRate=shade::RateVertex;m->dpds=s.dpds;m->dpdt=s.dpdt;
 std::vector<unsigned> triangleQuads,triangleHalves;  // each triangle's quad and which half of it
 for(size_t q=0;q<s.controlFace.size();++q){
  const int f=s.controlFace[q];if(faceMask&&(f<0||size_t(f)>=faceMask->size()||!(*faceMask)[size_t(f)]))continue;
  const unsigned* c=s.quads.data()+4*q;const unsigned tris[2][3]={{c[0],c[1],c[3]},{c[2],c[3],c[1]}};
  for(const auto& t:tris){V3 n;const V3 a=m->positions[t[0]],b=m->positions[t[1]],cc=m->positions[t[2]];
   if(!surface::normalize(surface::cross(surface::add(b,surface::mul(a,-1)),surface::add(cc,surface::mul(a,-1))),n))continue;
   m->triangles.push_back({t[0],t[1],t[2]});if(polyVertices)m->polyVertices.insert(m->polyVertices.end(),{c[0],c[1],c[2],c[3]});
   triangleQuads.push_back(unsigned(q));triangleHalves.push_back(&t==&tris[1]?1u:0u);}}
 need(!m->triangles.empty(),name+": no triangle of any area");
 for(const auto& a:s.attributes){const unsigned n=a.components;std::vector<float> values;unsigned rate=shade::PrimConstant;
  if(a.kind==1)values=a.values;
  else if(a.kind==3){rate=shade::PrimVertex;values=a.values;}
  else if(a.kind==2){rate=shade::PrimUniform;for(unsigned q:triangleQuads)values.insert(values.end(),a.values.begin()+std::ptrdiff_t(q*n),a.values.begin()+std::ptrdiff_t((q+1)*n));}
  else{rate=shade::PrimFaceVarying;const unsigned corners[2][3]={{0,1,3},{2,3,1}};
   for(size_t i=0;i<triangleQuads.size();++i){const unsigned q=triangleQuads[i];
    for(unsigned c:corners[triangleHalves[i]])values.insert(values.end(),a.values.begin()+std::ptrdiff_t((q*4+c)*n),a.values.begin()+std::ptrdiff_t((q*4+c+1)*n));}}
  m->prims.push_back(shade::PrimAttr{a.key,n,rate,unsigned(m->primData.size())});m->primData.insert(m->primData.end(),values.begin(),values.end());}
 return m;
}
// Motion blur's parameters (MotionParams, by Plan): motionParams reads them (RenderContext::getMotionBlurParams).
inline MotionParams motionParams(const rdl::SceneVariables& sv,const rdl::SceneObject* camera){
 namespace sm=scene_rdl2::math;MotionParams p;p.on=get<rdl::Bool>(sv,"enable_motion_blur");
 const auto& steps=get<rdl::FloatVector>(sv,"motion_steps");
 const float open=p.on&&camera?get<rdl::Float>(*camera,"mb_shutter_open"):0.0f,close=p.on&&camera?get<rdl::Float>(*camera,"mb_shutter_close"):0.0f;
 const float invFps=1.0f/get<rdl::Float>(sv,"fps");
 p.shutterOpen=open;p.shutterClose=close;
 if(steps.size()>=2){const float m0=steps[0],m1=steps[1];
  if(!sm::isEqual(m0,m1)){p.openDelta=(open-m0)/(m1-m0);p.closeDelta=(close-m0)/(m1-m0);}
  p.t0=m0*invFps;p.t1=m1*invFps;p.dt=p.t1-p.t0;}
 return p;}

// A mesh's vertices over the shutter, world space (geom::Primitive::transformVertexBuffer and RdlMeshProcedural::
// getVertexData): the positions, velocities and accelerations its motion_blur_type picks (static without motion blur,
// which then takes vertex_list_1 where it is given), made into Bezier control points (generateBezierPolygons),
// resampled over the shutter (generateMotionSamples: two time steps for linear motion, curved_motion_blur_sample_count
// for curved) and moved by the node transform, lerped to each step's time where it moves (applyTransformation;
// slerp_xforms falls back). The result's first entry is the shutter's open; one entry: a still mesh.
// Whether a mesh has motion data or a moving transform (meshMotionSteps then places its vertices).
inline bool meshHasMotion(const rdl::SceneObject& o){
 namespace sm=scene_rdl2::math;
 return !sm::isEqual(sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN)),sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END)))||
        !o.get<rdl::Vec3fVector>("vertex_list_1").empty()||!o.get<rdl::Vec3fVector>("velocity_list_0").empty()||
        !o.get<rdl::Vec3fVector>("velocity_list_1").empty()||!o.get<rdl::Vec3fVector>("accleration_list").empty();}
inline std::vector<std::vector<V3>> meshMotionSteps(const rdl::SceneObject& o,const MotionParams& mp,const rdl::Mat4d& xform0,const rdl::Mat4d& xform1,
                                                    bool staticTakesSecond=true,std::vector<float>* radiusScales=nullptr,
                                                    const std::vector<float>* w0=nullptr,const std::vector<float>* w1=nullptr,
                                                    std::vector<std::vector<float>>* widths=nullptr){
 namespace sm=scene_rdl2::math;
 const auto& pos0=o.get<rdl::Vec3fVector>("vertex_list_0");const auto& pos1=o.get<rdl::Vec3fVector>("vertex_list_1");
 const auto& vel0=o.get<rdl::Vec3fVector>("velocity_list_0");const auto& vel1=o.get<rdl::Vec3fVector>("velocity_list_1");
 const auto& acc=o.get<rdl::Vec3fVector>("accleration_list");
 const size_t n=pos0.size();
 const bool pos1Valid=pos1.size()==n,vel0Valid=vel0.size()==n,vel1Valid=vel1.size()==n,acc0Valid=acc.size()==n;
 int type=get<rdl::Int>(o,"motion_blur_type");if(!mp.on)type=0;  // MotionBlurType: static without motion blur
 int numPos=1,numVel=0,numAcc=0;
 switch(type){
 case 0:break;
 case 1:if(vel0Valid)numVel=1;break;
 case 2:if(pos1Valid)numPos=2;break;
 case 3:if(vel0Valid&&acc0Valid){numVel=1;numAcc=1;}break;
 case 4:if(pos1Valid&&vel0Valid&&vel1Valid){numPos=2;numVel=2;}break;
 case -1:
  if(pos1Valid&&vel0Valid&&vel1Valid){numPos=2;numVel=2;}else if(vel0Valid&&acc0Valid){numVel=1;numAcc=1;}
  else if(pos1Valid)numPos=2;else if(vel0Valid)numVel=1;
  break;
 default:break;}
 using P=sm::Vec3f;
 std::vector<std::vector<P>> sets;sets.assign(size_t(numPos),std::vector<P>(n));
 for(size_t i=0;i<n;++i){const auto& p=(pos1Valid&&type==0&&staticTakesSecond)?pos1[i]:pos0[i];sets[0][i]=P(p.x,p.y,p.z);}
 if(numPos==2)for(size_t i=0;i<n;++i)sets[1][i]=P(pos1[i].x,pos1[i].y,pos1[i].z);
 // The widths (a curve point's w): the raw steps' as the positions', velocities adding nothing.
 std::vector<std::vector<float>> ws;
 if(widths&&w0){ws.push_back((pos1Valid&&type==0&&staticTakesSecond&&w1)?*w1:*w0);if(numPos==2)ws.push_back(w1?*w1:*w0);}
 const float scale=get<rdl::Float>(o,"velocity_scale");
 auto velocity=[&](const rdl::Vec3fVector& v,size_t i){return P(v[i].x,v[i].y,v[i].z)*scale;};
 // generateBezierPolygons.
 if(numVel>0){
  std::vector<std::vector<P>> cps;
  if(numAcc==1&&numVel>=1){
   cps.assign(3,std::vector<P>(n));
   for(size_t i=0;i<n;++i){const P p=sets[0][i],v=velocity(vel0,i),a(acc[i].x,acc[i].y,acc[i].z);
    const P p0=p+mp.t0*v+0.5f*mp.t0*mp.t0*a,p1=p+mp.t1*v+0.5f*mp.t1*mp.t1*a,v0=v+mp.t0*a;
    cps[0][i]=p0;cps[1][i]=p0+0.5f*(mp.t1-mp.t0)*v0;cps[2][i]=p1;}
  }else if(numVel==2){
   cps.assign(4,std::vector<P>(n));const float third=(1.0f/3.0f)*mp.dt;
   for(size_t i=0;i<n;++i){cps[0][i]=sets[0][i];cps[1][i]=sets[0][i]+third*velocity(vel0,i);cps[2][i]=sets[1][i]-third*velocity(vel1,i);cps[3][i]=sets[1][i];}
  }else{
   cps.assign(2,std::vector<P>(n));
   for(size_t i=0;i<n;++i){const P p=sets[0][i],v=velocity(vel0,i);cps[0][i]=p+mp.t0*v;cps[1][i]=p+mp.t1*v;}
  }
  if(!ws.empty()){const auto a0=ws[0],a1=ws.size()>1?ws[1]:ws[0];
   ws=cps.size()==3?std::vector<std::vector<float>>{a0,a0,a0}:cps.size()==4?std::vector<std::vector<float>>{a0,a0,a1,a1}:std::vector<std::vector<float>>{a0,a0};}
  sets=std::move(cps);
 }
 // generateMotionSamples.
 int samples=mp.on?std::max(2,std::min(129,get<rdl::Int>(o,"curved_motion_blur_sample_count"))):1;
 if(samples>2&&(type==1||type==2))samples=2;
 const float ds0=mp.openDelta,ds1=mp.closeDelta;
 if(sets.size()==2){
  for(size_t i=0;i<n;++i){const P p0=sets[0][i],p1=sets[1][i];sets[0][i]=sm::lerp(p0,p1,ds0);sets[1][i]=sm::lerp(p0,p1,ds1);}
  if(ws.size()==2)for(size_t i=0;i<n;++i){const float a=ws[0][i],b=ws[1][i];ws[0][i]=sm::lerp(a,b,ds0);ws[1][i]=sm::lerp(a,b,ds1);}
 }else if(sets.size()==3||sets.size()==4){
  if(!ws.empty()){std::vector<std::vector<float>> wout;wout.assign(size_t(samples),std::vector<float>(n));const float dw=1.0f/float(samples-1);
   for(int j=0;j<samples;++j){const float u=float(j)*dw,t=sm::lerp(ds0,ds1,u),s=1.0f-t;
    if(ws.size()==3){const float c0=s*s,c1=2.0f*s*t,c2=t*t;for(size_t i=0;i<n;++i)wout[size_t(j)][i]=c0*ws[0][i]+c1*ws[1][i]+c2*ws[2][i];}
    else{const float t2=t*t,s2=s*s,c0=s*s2,c1=3.0f*t*s2,c2=3.0f*s*t2,c3=t*t2;
     for(size_t i=0;i<n;++i)wout[size_t(j)][i]=c0*ws[0][i]+c1*ws[1][i]+c2*ws[2][i]+c3*ws[3][i];}}
   ws=std::move(wout);}
  std::vector<std::vector<P>> out;out.assign(size_t(samples),std::vector<P>(n));const float du=1.0f/float(samples-1);
  for(int j=0;j<samples;++j){const float u=float(j)*du,t=sm::lerp(ds0,ds1,u),s=1.0f-t;
   if(sets.size()==3){const float c0=s*s,c1=2.0f*s*t,c2=t*t;
    for(size_t i=0;i<n;++i)out[size_t(j)][i]=c0*sets[0][i]+c1*sets[1][i]+c2*sets[2][i];}
   else{const float t2=t*t,s2=s*s,c0=s*s2,c1=3.0f*t*s2,c2=3.0f*s*t2,c3=t*t2;
    for(size_t i=0;i<n;++i)out[size_t(j)][i]=c0*sets[0][i]+c1*sets[1][i]+c2*sets[2][i]+c3*sets[3][i];}}
  sets=std::move(out);
 }
 // applyTransformation: the node transform at each step's time where it moves (GeometryManager's geometry2render:
 // two samples where its ends differ), as floats.
 const sm::Mat4f x0=sm::toFloat(xform0),x1=sm::toFloat(xform1);const bool moves=!sm::isEqual(x0,x1);
 if(moves&&sets.size()==1)sets.push_back(sets[0]);
 if(!ws.empty()&&ws.size()<sets.size())ws.resize(sets.size(),ws[0]);
 const size_t steps=sets.size();const float du=steps>1?1.0f/float(steps-1):0.0f;
 std::vector<std::vector<V3>> result(steps,std::vector<V3>(n));
 for(size_t j=0;j<steps;++j){const float u=float(j)*du,t=sm::lerp(ds0,ds1,u);
  const sm::Xform3f p2r=moves?sm::Xform3f(sm::lerp(sm::xform<sm::Xform3f>(x0),sm::xform<sm::Xform3f>(x1),t)):sm::xform<sm::Xform3f>(x0);
  if(radiusScales)radiusScales->push_back((sm::length(p2r.l.vx)+sm::length(p2r.l.vy)+sm::length(p2r.l.vz))/3.f);
  for(size_t i=0;i<n;++i){const P q=sm::transformPoint(p2r,sets[j][i]);result[j][i]=V3{q.x,q.y,q.z};
   need(surface::finite(result[j][i])&&std::abs(q.x)<=10000&&std::abs(q.y)<=10000&&std::abs(q.z)<=10000,"Transformed vertex bound");}}
 // A mesh with two identical steps is still (motion blur off, or nothing moving).
 if(steps==2){bool same=true;for(size_t i=0;i<n&&same;++i)same=result[0][i].x==result[1][i].x&&result[0][i].y==result[1][i].y&&result[0][i].z==result[1][i].z;
  if(same&&(!radiusScales||(*radiusScales)[0]==(*radiusScales)[1])&&(ws.size()<2||ws[0]==ws[1])){
   result.pop_back();if(radiusScales)radiusScales->pop_back();if(ws.size()==2)ws.pop_back();}}
 if(widths)*widths=std::move(ws);
 return result;}
// MoonRay's light filters (pbr/lightfilter/*.cc update) as the path tracer's records (light_filter_core.h), in world
// space (world2render the identity), their transforms at the shutter's ends (xformAt). A cookie's map shader and a VDB
// filter fall back; a combine within a combine falls back.
// vdbGrid (below): a grid of a VDB file for the plan.
inline unsigned vdbGrid(Plan& plan,const Plan* previous,const std::string& path,const std::string& name,bool vec3,const std::string& owner);
struct FilterAdmission {
 Plan& plan;const rdl::SceneVariables& sv;const Plan* previous;
 std::map<const rdl::SceneObject*,unsigned> ids;
 // A map's program and the register of its result (map_admission.h's MapProgramBuilder, set by the plan).
 std::function<std::pair<unsigned,unsigned>(const rdl::SceneObject&)> mapProgram;
 // updateTransforms: scene_rdl2's decomposition of the local to render transform at both ends (rod, barn door).
 void transforms(lights::Filter& f,const rdl::Mat4d& l2w0,const rdl::Mat4d& l2w1){
  namespace sm=scene_rdl2::math;
  auto m3=[](const sm::Mat3f& m){lights::M3 r;r.r[0]=V3{m.vx.x,m.vx.y,m.vx.z};r.r[1]=V3{m.vy.x,m.vy.y,m.vy.z};r.r[2]=V3{m.vz.x,m.vz.y,m.vz.z};return r;};
  sm::Quaternion3f q[2];
  auto at=[&](const sm::Mat4f& local2Render,int ti){
   const sm::Xform3f x=sm::xform<sm::Xform3f>(local2Render);sm::Mat3f scale;sm::Quaternion3f rot;sm::decompose(x.l,scale,rot);
   f.position[ti]=V3{local2Render.vw.x,local2Render.vw.y,local2Render.vw.z};
   f.local2RenderScale[ti]=m3(scale);
   if(ti==0){const sm::Xform3f inv=x.inverse();f.render2Local0.r[0]=V3{inv.l.vx.x,inv.l.vx.y,inv.l.vx.z};f.render2Local0.r[1]=V3{inv.l.vy.x,inv.l.vy.y,inv.l.vy.z};
    f.render2Local0.r[2]=V3{inv.l.vz.x,inv.l.vz.y,inv.l.vz.z};f.render2Local0.r[3]=V3{inv.p.x,inv.p.y,inv.p.z};
    f.render2LocalRot0=m3(sm::Mat3f(rot).transposed());f.render2LocalScale0=m3(scale.inverse());}
   const sm::Mat3f r(rot);q[ti]=sm::normalize(sm::Quaternion3f(r.vx,r.vy,r.vz));
   if(ti==1&&sm::dot(q[0],q[1])<0)q[1]*=-1.f;
   f.orientation[ti][0]=q[ti].r;f.orientation[ti][1]=q[ti].i;f.orientation[ti][2]=q[ti].j;f.orientation[ti][3]=q[ti].k;};
  const sm::Mat4f a=sm::toFloat(l2w0),b=sm::toFloat(l2w1);
  at(a,0);f.position[1]=f.position[0];f.local2RenderScale[1]=f.local2RenderScale[0];
  for(int k=0;k<4;++k)f.orientation[1][k]=f.orientation[0][k];
  f.mb=0;
  if(get<rdl::Bool>(sv,"enable_motion_blur")&&!sm::isEqual(a,b)){
   at(b,1);
   if(!sm::isEqual(sm::Vec3f(f.position[0].x,f.position[0].y,f.position[0].z),sm::Vec3f(f.position[1].x,f.position[1].y,f.position[1].z)))f.mb|=lights::FilterMbTranslation;
   if(!sm::isEqual(q[0],q[1]))f.mb|=lights::FilterMbRotation;
   bool same=true;for(int i=0;i<3;++i)same=same&&sm::isEqual(sm::Vec3f(f.local2RenderScale[0].r[i].x,f.local2RenderScale[0].r[i].y,f.local2RenderScale[0].r[i].z),
                                                              sm::Vec3f(f.local2RenderScale[1].r[i].x,f.local2RenderScale[1].r[i].y,f.local2RenderScale[1].r[i].z));
   if(!same)f.mb|=lights::FilterMbScale;}
 }
 static void matrix16(float* out,const scene_rdl2::math::Mat4f& m){for(int i=0;i<4;++i)for(int j=0;j<4;++j)out[4*i+j]=m[i][j];}
 // CookieLightFilter_v2::computePerspectiveProjectionMatrix and computeOrthoProjectionMatrix (its own projector).
 static scene_rdl2::math::Mat4f cookieProjection(const rdl::SceneObject& o,bool perspective){
  namespace sm=scene_rdl2::math;
  const float filmWidth=std::max(get<rdl::Float>(o,"projector_film_width_aperture"),0.01f),half=filmWidth*0.5f,near=1.0f,far=10000.0f;
  float par=get<rdl::Float>(o,"projector_pixel_aspect_ratio");if(sm::isEqual(par,0.0f))par=1.0f;
  sm::Mat4f c2s(sm::zero);
  if(perspective){
   const float focal=std::max(floatAt(o,"projector_focal",0.0f),0.1f),factor=near/focal;
   const float w[4]={(-2.f*half)*factor,(-2.f*half)*factor/par,(2.f*half)*factor,(2.f*half)*factor/par};
   c2s[0][0]=(2.0f*near)/(w[2]-w[0]);c2s[1][1]=(2.0f*near)/(w[3]-w[1]);c2s[2][0]=(w[2]+w[0])/(w[2]-w[0]);c2s[2][1]=(w[3]+w[1])/(w[3]-w[1]);
   c2s[2][2]=-(far+near)/(far-near);c2s[2][3]=-1.0f;c2s[3][2]=-(2.0f*far*near)/(far-near);c2s[3][3]=0.0f;
  }else{
   const float w[4]={(-2.f*half),(-2.f*half)/par,(2.f*half),(2.f*half)/par};
   c2s[0][0]=2.0f/(w[2]-w[0]);c2s[1][1]=2.0f/(w[3]-w[1]);c2s[2][2]=-2.0f/(far-near);
   c2s[3][0]=-(w[2]+w[0])/(w[2]-w[0]);c2s[3][1]=-(w[3]+w[1])/(w[3]-w[1]);c2s[3][2]=-(far+near)/(far-near);c2s[3][3]=1.0f;
  }
  return c2s;}
 unsigned admit(const rdl::SceneObject& o,int depth=0){
  namespace sm=scene_rdl2::math;
  auto found=ids.find(&o);if(found!=ids.end())return found->second;
  const std::string cls=o.getSceneClass().getName(),name=o.getName();
  {std::set<std::string> all;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)all.insert((*i)->getName());
   attributes(o,all,{},{"node_xform","projector_focal"});}
  lights::Filter f;
  auto rgb=[&](const char* k){auto c=get<rdl::Rgb>(o,k);return V3{c.r,c.g,c.b};};
  auto floatRamp=[&](shade::FloatRamp& r,std::vector<float> in,std::vector<float> out,std::vector<int> modes,int defaultMode){
   if(in.size()!=out.size()||in.size()!=modes.size()){in={0.f,1.f};out={0.f,1.f};modes={defaultMode,defaultMode};}
   need(in.size()<=shade::RampMaxPoints,name+": a ramp of more than 10 points");
   std::vector<unsigned> m(modes.begin(),modes.end());shade::rampInit(r,unsigned(in.size()),in.data(),out.data(),m.data());};
  if(cls=="IntensityLightFilter"){
   f.kind=lights::FilterIntensity;
   V3 c=surface::mul(rgb("color"),get<rdl::Float>(o,"intensity")*powf(2.0f,get<rdl::Float>(o,"exposure")));
   if(get<rdl::Bool>(o,"invert")){if(c.x!=0)c.x=1/c.x;if(c.y!=0)c.y=1/c.y;if(c.z!=0)c.z=1/c.z;}
   f.color=c;f.pathSelection=unsigned(get<rdl::Int>(o,"light_path_selection"));
  }else if(cls=="DecayLightFilter"){
   f.kind=lights::FilterDecay;f.falloffNear=get<rdl::Bool>(o,"falloff_near");f.falloffFar=get<rdl::Bool>(o,"falloff_far");
   f.nearStart=get<rdl::Float>(o,"near_start");f.nearEnd=std::max(get<rdl::Float>(o,"near_end"),f.nearStart);
   f.farStart=get<rdl::Float>(o,"far_start");if(f.falloffNear)f.farStart=std::max(f.farStart,f.nearEnd);
   f.farEnd=std::max(get<rdl::Float>(o,"far_end"),f.farStart);
  }else if(cls=="ColorRampLightFilter"){
   f.kind=lights::FilterColorRamp;
   for(int k=0;k<2;++k){const sm::Mat4f w=sm::toFloat(xformAt(o,float(k)).inverse());
    for(int i=0;i<4;++i)f.xform[k].r[i]=V3{w[i][0],w[i][1],w[i][2]};}
   f.useXform=get<rdl::Bool>(o,"use_xform");
   float begin=get<rdl::Float>(o,"begin_distance"),end=get<rdl::Float>(o,"end_distance");if(begin>=end){begin=0.f;end=1.f;}
   const auto& cv=o.get<rdl::RgbVector>("colors");const auto& dv=o.get<rdl::FloatVector>("distances");const auto& iv=o.get<rdl::IntVector>("interpolation_types");
   std::vector<shade::Color> colors;std::vector<float> distances(dv.begin(),dv.end());std::vector<unsigned> modes(iv.begin(),iv.end());
   for(auto c:cv)colors.push_back(shade::Color{c.r,c.g,c.b});
   if(distances.size()!=colors.size()||distances.size()!=modes.size()){colors={shade::Color{1,1,1},shade::Color{0,0,0}};distances={0.f,1.f};modes={1,1};}
   need(distances.size()<=shade::RampMaxPoints,name+": a ramp of more than 10 points");
   const float scale=end-begin;for(auto& d:distances)d=begin+scale*d;
   shade::rampInit(f.colorRamp,unsigned(distances.size()),distances.data(),colors.data(),modes.data(),shade::RampRgb);
   f.intensity=get<rdl::Float>(o,"intensity");f.density=std::min(std::max(get<rdl::Float>(o,"density"),0.f),1.f);
   f.mode=unsigned(get<rdl::Int>(o,"mode"));f.wrap=unsigned(get<rdl::Int>(o,"wrap_mode"));
  }else if(cls=="RodLightFilter"){
   f.kind=lights::FilterRod;transforms(f,xformAt(o,0.0f),xformAt(o,1.0f));
   const float w=get<rdl::Float>(o,"width"),d=get<rdl::Float>(o,"depth"),h=get<rdl::Float>(o,"height");
   f.radius=get<rdl::Float>(o,"radius");f.edge=get<rdl::Float>(o,"edge");f.edgeInv=f.edge!=0.0f?1.f/f.edge:1.f;
   f.color=surface::mul(rgb("color"),get<rdl::Float>(o,"intensity"));
   f.density=std::min(std::max(get<rdl::Float>(o,"density"),0.f),1.f);f.invert=get<rdl::Bool>(o,"invert");
   f.boxCorner=V3{std::fabs(w*0.5f),std::fabs(h*0.5f),std::fabs(d*0.5f)};
   f.render2LocalRotAndScale=lights::m3Mul(f.render2LocalRot0,f.render2LocalScale0);
   f.radiusEdgeSum=f.radius+f.edge;
   const auto& in=o.get<rdl::FloatVector>("ramp_in_distances");const auto& out=o.get<rdl::FloatVector>("ramp_out_distances");const auto& modes=o.get<rdl::IntVector>("ramp_interpolation_types");
   floatRamp(f.ramp,std::vector<float>(in.begin(),in.end()),std::vector<float>(out.begin(),out.end()),std::vector<int>(modes.begin(),modes.end()),int(shade::RampMonotoneCubic));
  }else if(cls=="BarnDoorLightFilter"){
   f.kind=lights::FilterBarnDoor;
   const float filmW=std::max(get<rdl::Float>(o,"projector_width"),0.00001f),filmH=std::max(get<rdl::Float>(o,"projector_height"),0.00001f);
   const float sizes[4]={get<rdl::Float>(o,"size_left"),get<rdl::Float>(o,"size_bottom"),get<rdl::Float>(o,"size_right"),get<rdl::Float>(o,"size_top")};
   const float effW=filmW+sizes[0]+sizes[2],effH=filmH+sizes[1]+sizes[3],smallest=std::min(effW,effH);
   f.radius=get<rdl::Float>(o,"radius")*(smallest*0.5f);f.edge=get<rdl::Float>(o,"edge")*smallest;
   const float scales[4]={get<rdl::Float>(o,"edge_scale_left"),get<rdl::Float>(o,"edge_scale_bottom"),get<rdl::Float>(o,"edge_scale_right"),get<rdl::Float>(o,"edge_scale_top")};
   for(int k=0;k<4;++k)f.reciprocalEdgeScales[k]=1.0f/std::max(1e-9f,f.radius+f.edge*scales[k]);
   f.focalDist=std::max(get<rdl::Float>(o,"projector_focal_distance"),0.1f);
   sm::Mat4f proj=get<rdl::Int>(o,"projector_type")==0?sm::Mat4f(1.f,0.f,0.f,0.f,0.f,-1.f,0.f,0.f,0.f,0.f,1.f,1.f,0.f,0.f,2.f,0.f)
                                                       :sm::Mat4f(1.f,0.f,0.f,0.f,0.f,-1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f);
   const float rotAngle=get<rdl::Float>(o,"rotation")*sm::sPi/180.0f;
   proj=sm::Mat4f::rotate(sm::Vec4f(0,0,1,0),rotAngle)*proj;
   matrix16(f.projector2Screen,proj);
   const float fw=effW-2.f*f.radius,fh=effH-2.f*f.radius,cx=(sizes[2]-sizes[0])*0.5f,cy=(sizes[3]-sizes[1])*0.5f;
   f.minCorner[0]=cx-fw*0.5f;f.minCorner[1]=cy-fh*0.5f;f.maxCorner[0]=cx+fw*0.5f;f.maxCorner[1]=cy+fh*0.5f;
   const sm::Mat4d barnRot=sm::Mat4d::rotate(sm::Vec4d(1,0,0,0),sm::sPi);
   transforms(f,barnRot*xformAt(o,0.0f),barnRot*xformAt(o,1.0f));
   f.mode=unsigned(get<rdl::Int>(o,"mode"));f.useLightXform=get<rdl::Bool>(o,"use_light_xform");
   f.preBarnMode=unsigned(get<rdl::Int>(o,"pre_barn_mode"));f.preBarnDist=get<rdl::Float>(o,"pre_barn_distance");
   f.density=std::min(std::max(get<rdl::Float>(o,"density"),0.f),1.f);f.invert=get<rdl::Bool>(o,"invert");f.color=rgb("color");
  }else if(cls=="CookieLightFilter_v2"||cls=="CookieLightFilter"){
   // Cookie v1 (CookieLightFilter.cc): v2's projector and blur, its texture_map a map shader (none: white), its
   // samples needed only for a blur.
   const bool v1=cls=="CookieLightFilter";
   f.kind=v1?lights::FilterCookie:lights::FilterCookie2;f.needsSamples=1;
   const rdl::SceneObject* projector=get<rdl::SceneObject*>(o,"projector");
   sm::Mat4d p0,p1,p2s;
   if(projector){
    const std::string pc=projector->getSceneClass().getName();
    need(pc=="PerspectiveCamera"||pc=="OrthographicCamera",name+": the projector has no projection matrix");
    p0=xformAt(*projector,0.0f);p1=xformAt(*projector,1.0f);
    p2s=sm::Mat4d(cameraProjection(*projector,pc=="PerspectiveCamera",{-2.f,-2.f,2.f,2.f},0.0f));
   }else{
    p0=xformAt(o,0.0f);p1=xformAt(o,1.0f);p2s=sm::Mat4d(cookieProjection(o,get<rdl::Int>(o,"projector_type")==0));
   }
   matrix16(f.projectorR2S[0],sm::Mat4f(p0.inverse()*p2s));matrix16(f.projectorR2S[1],sm::Mat4f(p1.inverse()*p2s));
   f.projectorPos[0]=V3{float(p0[3][0]),float(p0[3][1]),float(p0[3][2])};f.projectorPos[1]=V3{float(p1[3][0]),float(p1[3][1]),float(p1[3][2])};
   f.projectorDir[0]=V3{-float(p0[2][0]),-float(p0[2][1]),-float(p0[2][2])};f.projectorDir[1]=V3{-float(p1[2][0]),-float(p1[2][1]),-float(p1[2][2])};
   f.blurNearDistance=get<rdl::Float>(o,"blur_near_distance");f.blurMidpoint=get<rdl::Float>(o,"blur_midpoint");f.blurFarDistance=get<rdl::Float>(o,"blur_far_distance");
   f.blurNearValue=get<rdl::Float>(o,"blur_near_value");f.blurMidValue=get<rdl::Float>(o,"blur_mid_value");f.blurFarValue=get<rdl::Float>(o,"blur_far_value");
   f.blurType=unsigned(get<rdl::Int>(o,"blur_type"));f.outside=unsigned(get<rdl::Int>(o,"outside_projection"));
   f.density=std::min(std::max(get<rdl::Float>(o,"density"),0.f),1.f);f.invert=get<rdl::Bool>(o,"invert");
   if(v1){f.needsSamples=(f.blurNearValue>0.f||f.blurMidValue>0.f||f.blurFarValue>0.f)?1:0;
    const rdl::SceneObject* m=get<rdl::SceneObject*>(o,"texture_map");
    if(m&&m->isA<rdl::Map>()){need(bool(mapProgram),name+": no map programs for a cookie");const auto pr=mapProgram(*m);f.mapProgram=pr.first;f.mapRegister=pr.second;}}
   const std::string file=v1?std::string():get<rdl::String>(o,"texture");f.fallback=V3{1,1,1};
   if(!file.empty()){
    lights::ImageSettings st;st.file=file;st.mapping=lights::MapPlanar;{auto g=get<rdl::Rgb>(o,"gamma");st.gamma[0]=g.r;st.gamma[1]=g.g;st.gamma[2]=g.b;}
    const std::string key=st.key();std::shared_ptr<lights::ImageData> data;
    for(auto& old:plan.filterImages)if(old->key==key){data=old;break;}
    if(!data&&previous)for(auto& old:previous->filterImages)if(old->key==key){data=old;break;}
    if(!data){try{data=lights::buildImage(st);}catch(const std::exception&){auto c=get<rdl::Rgb>(sv,"fatal_color");f.fallback=V3{c.r,c.g,c.b};}}
    if(data){if(std::find(plan.filterImages.begin(),plan.filterImages.end(),data)==plan.filterImages.end())plan.filterImages.push_back(data);f.image=&data->view;}
   }
  }else if(cls=="CombineLightFilter"){
   need(depth<16,name+": combine light filters nested beyond sixteen (a cycle?)");
   f.kind=lights::FilterCombine;f.mode=unsigned(get<rdl::Int>(o,"mode"));
   std::vector<unsigned> children;
   for(auto* c:o.get<rdl::SceneObjectVector>("light_filters")){if(!c||!c->isA<rdl::LightFilter>()||!get<rdl::Bool>(*c,"on"))continue;children.push_back(admit(*c,depth+1));}
   need(children.size()<=lights::FilterMaxChildren,name+": a combine of more than 8 light filters");
   for(size_t k=0;k<children.size();++k){f.children[k]=children[k];if(plan.filters[children[k]].needsSamples)f.needsSamples=1;}
   f.childCount=unsigned(children.size());
  }else if(cls=="VdbLightFilter"){
   // VdbLightFilter::update: the VDB's space at the shutter's ends, its density grid ("density" by default; one that
   // cannot be read gives density 1), the remap (sizes that differ: linear 0 to 1; rescaled into the given ranges).
   f.kind=lights::FilterVdb;
   auto x34=[](const rdl::Mat4d& m){lights::X34 x;for(int i=0;i<4;++i)x.r[i]=V3{float(m[i][0]),float(m[i][1]),float(m[i][2])};return x;};
   f.vdbR2V[0]=x34(xformAt(o,0.0f).inverse());f.vdbR2V[1]=x34(xformAt(o,1.0f).inverse());
   const std::string path=get<rdl::String>(o,"vdb_map");std::string gridName=get<rdl::String>(o,"density_grid_name");if(gridName.empty())gridName="density";
   f.vdbGrid=volume::NoGrid;
   if(!path.empty()){std::error_code ec;
    if(fs::exists(fs::path(path),ec)){
     try{f.vdbGrid=vdbGrid(plan,previous,path,gridName,false,name);}
     catch(const std::exception& e){if(std::string(e.what()).find("frustum")!=std::string::npos||std::string(e.what()).find("did not load")!=std::string::npos)throw;f.vdbGrid=volume::NoGrid;}}}
   f.vdbInterpolation=unsigned(get<rdl::Int>(o,"vdb_interpolation_type"));
   std::vector<float> in(o.get<rdl::FloatVector>("density_remap_inputs").begin(),o.get<rdl::FloatVector>("density_remap_inputs").end());
   std::vector<float> out(o.get<rdl::FloatVector>("density_remap_outputs").begin(),o.get<rdl::FloatVector>("density_remap_outputs").end());
   std::vector<int> modes(o.get<rdl::IntVector>("density_remap_interpolation_types").begin(),o.get<rdl::IntVector>("density_remap_interpolation_types").end());
   if(in.size()!=out.size()||in.size()!=modes.size()){in={0.f,1.f};out={0.f,1.f};modes={int(shade::RampLinear),int(shade::RampLinear)};}
   if(get<rdl::Bool>(o,"density_rescale_enable")){
    float inMin=get<rdl::Float>(o,"density_remap_input_min"),inMax=get<rdl::Float>(o,"density_remap_input_max");
    float outMin=get<rdl::Float>(o,"density_remap_output_min"),outMax=get<rdl::Float>(o,"density_remap_output_max");
    if(inMin>=inMax){inMin=0.f;inMax=1.f;}if(outMin>=outMax){outMin=0.f;outMax=1.f;}
    for(auto& v:in)v=v*(inMax-inMin)+inMin;for(auto& v:out)v=v*(outMax-outMin)+outMin;}
   floatRamp(f.ramp,in,out,modes,int(shade::RampLinear));
   f.vdbBlur=get<rdl::Float>(o,"blur_value");f.blurType=unsigned(get<rdl::Int>(o,"blur_type"));f.needsSamples=f.vdbBlur>0.0f?1u:0u;
   f.invert=get<rdl::Bool>(o,"invert_density")?1u:0u;f.color=rgb("color_tint");
  }else throw Unsupported(name+": unknown light filter "+cls);
  const unsigned id=unsigned(plan.filters.size());plan.filters.push_back(f);ids[&o]=id;return id;
 }
};
inline bool admitLightTexture(const rdl::SceneObject& o,lights::Light& l,lights::Mapping mapping,const rdl::SceneVariables& sv,
                              std::vector<std::shared_ptr<lights::ImageData>>& images,const std::vector<std::shared_ptr<lights::ImageData>>* previous){
 l.fallback={1,1,1};l.image=nullptr;l.log2TexelAngle=-1e30f;
 const int filter=get<rdl::Int>(o,"texture_filter");need(filter>=0&&filter<=3,o.getName()+": unknown texture filter");l.filter=unsigned(filter);
 const std::string file=get<rdl::String>(o,"texture");if(file.empty())return true;
 lights::ImageSettings st;st.file=file;st.mapping=mapping;
 auto rgb=[&](const char* k,float* v){auto c=get<rdl::Rgb>(o,k);v[0]=c.r;v[1]=c.g;v[2]=c.b;};
 rgb("gamma",st.gamma);rgb("contrast",st.contrast);rgb("saturation",st.saturation);rgb("gain",st.gain);rgb("offset",st.offset);rgb("texture_border_color",st.border);
 {auto t=get<rdl::Vec3f>(o,"temperature");st.temperature[0]=t.x;st.temperature[1]=t.y;st.temperature[2]=t.z;}
 st.rotation=get<rdl::Float>(o,"texture_rotation");
 {auto t=get<rdl::Vec2f>(o,"texture_translation");st.translation[0]=t.x;st.translation[1]=t.y;}
 {auto c=get<rdl::Vec2f>(o,"texture_coverage");st.coverage[0]=c.x;st.coverage[1]=c.y;}
 st.repsU=get<rdl::Float>(o,"texture_reps_u");st.repsV=get<rdl::Float>(o,"texture_reps_v");
 st.mirrorU=get<rdl::Bool>(o,"texture_mirror_u");st.mirrorV=get<rdl::Bool>(o,"texture_mirror_v");
 const std::string key=st.key();std::shared_ptr<lights::ImageData> data;
 const std::vector<std::shared_ptr<lights::ImageData>>* sets[2]={&images,previous};
 for(auto* set:sets)if(set&&!data)for(auto& old:*set)if(old->key==key){data=old;break;}
 if(!data){
  try{data=lights::buildImage(st);}
  catch(const std::exception&){auto f=get<rdl::Rgb>(sv,"fatal_color");l.fallback={f.r,f.g,f.b};return true;}
 }
 if(!data)return false;
 if(std::find(images.begin(),images.end(),data)==images.end())images.push_back(data);
 l.image=&data->view;
 return true;
}
// Whether the camera sees the light (Light::updateVisibilityFlags): force off, force on, or the scene's
// lights_visible_in_camera.
inline unsigned lightVisibleInCamera(const rdl::SceneObject& o,const rdl::SceneVariables& sv){
 const int v=get<rdl::Int>(o,"visible_in_camera");return v==2?(get<rdl::Bool>(sv,"lights_visible_in_camera")?1u:0u):v==1?1u:0u;
}
// The settings every light has that this renderer reads (texture, its filter and controls, camera visibility).
inline const std::set<std::string>& lightCommonAttributes(){
 static const std::set<std::string> s{"on","color","intensity","exposure","node_xform","label","visible_in_camera","texture","texture_filter",
  "gamma","contrast","saturation","gain","offset","temperature","texture_rotation","texture_translation","texture_coverage",
  "texture_reps_u","texture_reps_v","texture_mirror_u","texture_mirror_v","texture_border_color","min_shadow_distance","max_shadow_distance",
  "visible_diffuse_reflection","visible_diffuse_transmission","visible_glossy_reflection","visible_glossy_transmission","visible_mirror_reflection",
  "visible_mirror_transmission","presence_shadows","light_filters","ray_termination"};return s;}
// Light::updatePresenceShadows: on, off, or the scene's enable_presence_shadows.
inline unsigned lightPresenceShadows(const rdl::SceneObject& o,const rdl::SceneVariables& sv){
 const int v=get<rdl::Int>(o,"presence_shadows");return v==2?(get<rdl::Bool>(sv,"enable_presence_shadows")?1u:0u):v==1?1u:0u;}
// rdl2::Light::getVisibilityMask: the lobe types that see the light.
inline unsigned lightVisibilityMask(const rdl::SceneObject& o){
 unsigned m=0;const char* names[6]={"visible_diffuse_reflection","visible_diffuse_transmission","visible_glossy_reflection","visible_glossy_transmission",
  "visible_mirror_reflection","visible_mirror_transmission"};
 for(int i=0;i<6;++i)if(get<rdl::Bool>(o,names[i]))m|=1u<<(2+i);
 return m;
}
// Light::updateMinShadowDistance, updateMaxShadowDistance, and UPDATE_ATTRS_CLEAR_RADIUS where the light has one.
inline void lightShadowDistances(const rdl::SceneObject& o,lights::Light& l,bool clearRadius){
 l.minShadowDistance=get<rdl::Float>(o,"min_shadow_distance");l.maxShadowDistance=get<rdl::Float>(o,"max_shadow_distance");
 need(std::isfinite(l.minShadowDistance)&&std::isfinite(l.maxShadowDistance),o.getName()+": invalid shadow distance");
 if(!clearRadius)return;
 l.clearRadius=std::max(get<rdl::Float>(o,"clear_radius"),0.f);l.clearRadiusFalloff=std::max(get<rdl::Float>(o,"clear_radius_falloff_distance"),0.f);
 const int type=get<rdl::Int>(o,"clear_radius_interpolation_type");need(std::isfinite(l.clearRadius)&&std::isfinite(l.clearRadiusFalloff)&&type>=0,o.getName()+": invalid clear radius");
 l.clearRadiusInterpolation=unsigned(type);
}
inline const std::set<std::string>& clearRadiusAttributes(){static const std::set<std::string> s{"clear_radius","clear_radius_falloff_distance","clear_radius_interpolation_type"};return s;}
inline bool admitLight(const rdl::SceneObject& o,float sceneScale,lights::Light& out,const rdl::SceneVariables& sv,
                       std::vector<std::shared_ptr<lights::ImageData>>& images,const std::vector<std::shared_ptr<lights::ImageData>>* previousImages,
                       float shutterTime=0.0f){
 auto cls=o.getSceneClass().getName();
 // A PortalLight is a RectLight (its rect) that its reference lights the scene through (resolved by the caller).
 const bool portal=cls=="PortalLight";if(portal)cls="RectLight";
 // A label names the light for light path expressions only (light AOVs fall back).
 std::set<std::string> allowed=lightCommonAttributes();allowed.insert({"normalized","apply_scene_scale","sidedness","mb"});
 if(portal)allowed.insert("light");
 if(cls=="EnvLight")allowed.insert("sample_upper_hemisphere_only");
 if(cls=="RectLight")allowed.insert({"width","height","spread"});else if(cls=="DiskLight")allowed.insert({"radius","spread"});else if(cls=="SphereLight")allowed.insert("radius");
 else if(cls=="CylinderLight")allowed.insert({"radius","height"});else if(cls=="DistantLight")allowed.insert("angular_extent");
 else if(cls=="SpotLight")allowed.insert({"lens_radius","aspect_ratio","focal_plane_distance","outer_cone_angle","inner_cone_angle","angle_falloff_type","black_level"});
 const bool clearRadius=cls=="RectLight"||cls=="DiskLight"||cls=="SphereLight"||cls=="SpotLight"||cls=="CylinderLight";
 if(clearRadius)allowed.insert(clearRadiusAttributes().begin(),clearRadiusAttributes().end());
 // A light without its mb switch is still under motion blur: its transform at the first time step (Light::update's
 // local2Render0 alone).
 attributes(o,allowed,{},{"node_xform"});
 if(!get<rdl::Bool>(o,"on"))return false;
 lights::Light l{};
 auto c=color(get<rdl::Rgb>(o,"color"));float intensity=get<rdl::Float>(o,"intensity"),exposure=get<rdl::Float>(o,"exposure");
 need(surface::finite(intensity)&&intensity>=0&&surface::finite(exposure)&&std::abs(exposure)<=32,o.getName()+": invalid light intensity");
 V3 radiance=surface::mul(c,intensity*std::pow(2.f,exposure));
 l.visibleInCamera=lightVisibleInCamera(o,sv);l.fallback={1,1,1};l.visibilityMask=lightVisibilityMask(o);l.presenceShadows=lightPresenceShadows(o,sv);
 l.rayTerminator=get<rdl::Bool>(o,"ray_termination")?1u:0u;
 lightShadowDistances(o,l,clearRadius);
 if(cls=="EnvLight"){
  // EnvLight::update: its frame is sLocalOrientation * node_xform orthonormalized (Mat4f::orthonormalize), rows
  // X = -row2, Y = -row0, Z = row1 of the transform; Z is its direction, up for an untransformed light.
  auto m=xformAt(o,shutterTime);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j]),o.getName()+": nonfinite light transform");
  const V3 r1{float(-m[0][0]),float(-m[0][1]),float(-m[0][2])},r2{float(m[1][0]),float(m[1][1]),float(m[1][2])};
  V3 x,y;need(surface::normalize(surface::cross(r1,r2),x)&&surface::normalize(surface::cross(r2,x),y),o.getName()+": degenerate light transform");
  const V3 z=surface::cross(x,y);
  l.type=lights::Env;l.local2Render[0]=x;l.local2Render[1]=y;l.local2Render[2]=z;l.direction=z;l.position={0,0,0};
  l.hemispherical=get<rdl::Bool>(o,"sample_upper_hemisphere_only")?1u:0u;
  l.invArea=1.0f/(l.hemispherical?2*lights::Pi:4*lights::Pi);l.radiance=radiance;
  if(radiance.x==0&&radiance.y==0&&radiance.z==0)return false;  // a black light is off
  if(!admitLightTexture(o,l,l.hemispherical?lights::MapHemispherical:lights::MapSpherical,sv,images,previousImages))return false;
  if(l.image){const float a=std::max(2*lights::Pi/float(l.image->width),lights::Pi/float(l.image->height));l.log2TexelAngle=std::log2(a);}
  out=l;return surface::finite(radiance);
 }
 auto m=xformAt(o,shutterTime);
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j])&&std::abs(m[i][j])<=10000,o.getName()+": light transform bound");
 need(std::abs(m[0][3])<=1e-6&&std::abs(m[1][3])<=1e-6&&std::abs(m[2][3])<=1e-6&&std::abs(m[3][3]-1)<=1e-6,o.getName()+": projective light transform unsupported");
 // Rows of sRotateX180 * node_xform; Disk and Sphere scale the rotation by their radius first.
 double rows[4][3];
 // The cylinder alone has no sRotateX180 (CylinderLight::update).
 for(int i=0;i<4;++i)for(int j=0;j<3;++j)rows[i][j]=(cls!="CylinderLight"&&(i==1||i==2)?-m[i][j]:m[i][j]);
 auto length=[&](int i){return std::sqrt(rows[i][0]*rows[i][0]+rows[i][1]*rows[i][1]+rows[i][2]*rows[i][2]);};
 const bool normalized=get<rdl::Bool>(o,"normalized");
 constexpr double pi=3.14159265358979323846;
 if(cls=="DistantLight"){
  // Mat4f::orthonormalize: x from y and z, y from z and x, z from both.
  auto cross=[](const double a[3],const double b[3],double r[3]){r[0]=a[1]*b[2]-a[2]*b[1];r[1]=a[2]*b[0]-a[0]*b[2];r[2]=a[0]*b[1]-a[1]*b[0];};
  auto unit=[](double r[3]){double s=std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);need(s>1e-12,"Degenerate light transform");for(int k=0;k<3;++k)r[k]/=s;};
  double x[3],y[3],z[3];cross(rows[1],rows[2],x);unit(x);cross(rows[2],x,y);unit(y);cross(x,y,z);
  for(int k=0;k<3;++k){rows[0][k]=x[k];rows[1][k]=y[k];rows[2][k]=z[k];}
  float extent=std::max(get<rdl::Float>(o,"angular_extent"),lights::Epsilon);need(std::isfinite(extent)&&extent<=360,o.getName()+": invalid angular extent");
  float thetaMax=0.5f*extent*float(pi/180.0),s=std::sin(thetaMax*0.5f);
  l.type=lights::Distant;l.cosThetaMax=std::cos(thetaMax);l.versineThetaMax=2*s*s;l.invArea=1/(l.versineThetaMax*lights::TwoPi);
  if(normalized){float h=std::sin(std::min(thetaMax,lights::Pi/2));radiance=surface::mul(radiance,1/(h*h));}
  for(int i=0;i<3;++i)l.local2Render[i]={float(rows[i][0]),float(rows[i][1]),float(rows[i][2])};
  l.direction=l.local2Render[2];l.radiance=radiance;
  // Its texture as an equisolid fisheye over its cone (DistantLight::update).
  const float sinHalf=std::sin(thetaMax*0.5f);
  l.localToUvConst=0.5f*std::sqrt(0.5f)/sinHalf;l.uvToLocalConst=4.0f*l.versineThetaMax;l.jacobian=0.125f/l.versineThetaMax;
  if(radiance.x==0&&radiance.y==0&&radiance.z==0)return false;
  if(!admitLightTexture(o,l,lights::MapCircular,sv,images,previousImages))return false;
  if(l.image){const float rad=2*thetaMax,a=std::max(rad/float(l.image->width),rad/float(l.image->height));l.log2TexelAngle=std::log2(a);}
  out=l;return surface::finite(radiance);
 }
 if(cls!="SpotLight"){int sidedness=get<rdl::Int>(o,"sidedness");need(sidedness>=0&&sidedness<=2,o.getName()+": invalid sidedness");l.sidedness=unsigned(sidedness);}
 float width=1,height=1;
 if(cls=="RectLight"){width=get<rdl::Float>(o,"width");height=get<rdl::Float>(o,"height");need(std::isfinite(width)&&std::isfinite(height)&&width>0&&height>0,o.getName()+": invalid size");}
 else if(cls=="SpotLight"||cls=="CylinderLight"){}
 else{float r=get<rdl::Float>(o,"radius");need(std::isfinite(r)&&r>0,o.getName()+": invalid radius");for(int i=0;i<3;++i)for(int j=0;j<3;++j)rows[i][j]*=r;}
 double x=length(0),y=length(1),z=length(2),scale;
 if(std::abs(x-y)<=0.0001&&std::abs(x-z)<=0.0001)scale=(x+y+z)/3;
 else{
  need(cls=="RectLight",o.getName()+": non-uniform light scale unsupported");
  // RectLight takes non-uniform scale into its size and keeps a unit frame.
  width*=float(x);height*=float(y);for(int i=0;i<3;++i){double s=length(i);for(int j=0;j<3;++j)rows[i][j]/=s;}scale=1;
 }
 need(scale>1e-12,o.getName()+": degenerate light transform");
 double det=rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])-rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])+rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]);
 need(std::abs(std::abs(det/(scale*scale*scale))-1)<=0.01,o.getName()+": invalid light transform");
 // The inverse of the linear part, rows again.
 double inv[3][3];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,d=(i+2)%3;inv[i][j]=(rows[a][cc]*rows[b][d]-rows[a][d]*rows[b][cc])/det;}
 for(int i=0;i<3;++i){l.local2Render[i]={float(rows[i][0]),float(rows[i][1]),float(rows[i][2])};
  l.render2Local[i]={float(inv[i][0]),float(inv[i][1]),float(inv[i][2])};
  l.render2LocalRot[i]={float(inv[i][0]*scale),float(inv[i][1]*scale),float(inv[i][2]*scale)};}
 l.position={float(rows[3][0]),float(rows[3][1]),float(rows[3][2])};
 l.direction={float(rows[2][0]/scale),float(rows[2][1]/scale),float(rows[2][2]/scale)};
 l.planeOffset=-surface::dot(l.direction,l.position);l.scale=float(scale);
 double area;
 if(cls=="RectLight"){l.type=lights::Rect;l.halfWidth=width*0.5f;l.halfHeight=height*0.5f;area=double(width)*scale*double(height)*scale;}
 else if(cls=="DiskLight"){l.type=lights::Disk;area=pi*scale*scale;}
 else if(cls=="CylinderLight"){
  // CylinderLight::update: its radius and height in local units, scaled to render units by the transform.
  const float r=get<rdl::Float>(o,"radius"),hgt=get<rdl::Float>(o,"height");need(std::isfinite(r)&&std::isfinite(hgt)&&r>0&&hgt>0,o.getName()+": invalid cylinder");
  l.type=lights::Cylinder;l.localRadius=r;l.localHalfHeight=hgt*0.5f;l.radius=float(r*scale);l.rcpRadius=1/l.radius;l.radiusSq=l.radius*l.radius;
  l.actualHalfHeight=float(l.localHalfHeight*scale);area=4*pi*double(l.radius)*l.actualHalfHeight;
  l.axis={float(rows[1][0]/scale),float(rows[1][1]/scale),float(rows[1][2]/scale)};
 }
 else if(cls=="SpotLight"){
  // SpotLight::update: its lens, cones and focal plane.
  float aspect=get<rdl::Float>(o,"aspect_ratio"),lens=get<rdl::Float>(o,"lens_radius"),outer=get<rdl::Float>(o,"outer_cone_angle"),inner=get<rdl::Float>(o,"inner_cone_angle"),focal=get<rdl::Float>(o,"focal_plane_distance");
  need(std::isfinite(aspect)&&aspect>0&&std::isfinite(lens)&&lens>0&&std::isfinite(outer)&&std::isfinite(inner)&&std::isfinite(focal),o.getName()+": invalid spot light");
  int type=get<rdl::Int>(o,"angle_falloff_type");need(type>=0&&type<=4,o.getName()+": invalid falloff");
  float rcpAspect=1/aspect;l.lensRadius=lens;l.rcpLensRadius=1/lens;l.lensRadiusY=lens*rcpAspect;l.rcpLensRadiusY=1/l.lensRadiusY;
  outer=std::min(std::max(outer,-179.999f),179.999f);inner=std::min(std::max(inner,-179.999f),outer);
  float tanOuter=std::tan(float(outer*0.5*pi/180.0)),tanInner=std::tan(float(inner*0.5*pi/180.0)),tanDiff=tanOuter-tanInner;
  if(focal<0.0001f)focal=0.0001f;if(tanOuter<0.0f)focal=std::min(focal,-0.999f*lens/tanOuter);
  l.focalDistance=focal;l.focalRadius=tanOuter*focal+lens;l.rcpFocalRadius=1/l.focalRadius;l.focalRadiusY=l.focalRadius*rcpAspect;l.rcpFocalRadiusY=1/l.focalRadiusY;
  l.falloffGradient=l.focalRadius/std::max(tanDiff*focal,1.0e-10f);l.crossOverDistance=focal*lens/(lens+l.focalRadius);l.falloffType=unsigned(type);
  l.type=lights::Spot;area=pi*(double(lens)*scale)*(double(l.lensRadiusY)*scale);
  double ratio=double(lens)/l.focalRadius;l.focalPlanePdfConst=float(ratio*ratio/area);
 }
 else{l.type=lights::Sphere;l.radius=float(scale);l.radiusSq=l.radius*l.radius;l.rcpRadius=1/l.radius;area=4*pi*scale*scale;}
 l.invArea=float(1/area);
 // computeLightRadiance: normalized lights spread their power over their area, in scene units.
 if(normalized){double inverse=1/area;if(get<rdl::Bool>(o,"apply_scene_scale"))inverse*=1/(double(sceneScale)*sceneScale)/pi;radiance=surface::mul(radiance,float(inverse));}
 l.radiance=radiance;
 if(radiance.x==0&&radiance.y==0&&radiance.z==0)return false;  // a black light is off
 // Spread (rect, disk: an egg-crate's cone), texture mappings (getScaleOffset over the light's half extents, local
 // units), the disk's texture pdf scale and the spot's black level.
 l.spread=1.0f;l.tanSpreadTheta=std::tan(1.0f*lights::Pi*0.5f);
 if(cls=="RectLight"||cls=="DiskLight"){l.spread=std::min(std::max(get<rdl::Float>(o,"spread"),1e-6f),1.0f);l.tanSpreadTheta=std::tan(l.spread*(lights::Pi*0.5f));}
 const float hw=cls=="RectLight"?l.halfWidth:cls=="CylinderLight"?l.localHalfHeight:1.0f,hh=cls=="RectLight"?l.halfHeight:1.0f;
 l.uvScale[0]=(0.0f-1.0f)/(hw-(-hw));l.uvOffset[0]=1.0f-(-hw*l.uvScale[0]);
 l.uvScale[1]=(0.0f-1.0f)/(hh-(-hh));l.uvOffset[1]=1.0f-(-hh*l.uvScale[1]);
 if(cls=="DiskLight")l.distributionPdfScale=1.0f/(4.0f*l.scale*l.scale);
 if(cls=="SpotLight")l.blackLevel=get<rdl::Float>(o,"black_level");
 const lights::Mapping mapping=cls=="RectLight"||cls=="CylinderLight"?lights::MapPlanar:cls=="SphereLight"?lights::MapSpherical:lights::MapCircular;
 if(!admitLightTexture(o,l,mapping,sv,images,previousImages))return false;
 if(portal){
  // PortalLight::update: off without a reference, or with one neither an EnvLight nor a DistantLight.
  const rdl::SceneObject* ref=get<rdl::SceneObject*>(o,"light");
  if(!ref)return false;const std::string rc=ref->getSceneClass().getName();if(rc!="EnvLight"&&rc!="DistantLight")return false;
  l.type=lights::Portal;
 }
 out=l;return surface::finite(radiance);
}
inline bool isCurves(const std::string& cls){return cls=="RdlCurveGeometry"||cls=="ModoFurGeometry";}
inline bool isPoints(const std::string& cls){return cls=="RdlPointGeometry";}
// A geometry's visibility attributes (rdl2::Geometry): the rays that see it. Phase rays (visible_volume) see every
// geometry here, so one hidden from them falls back.
inline const std::set<std::string>& geometryVisibilityAttributes(){
 static const std::set<std::string> s{"visible_in_camera","visible_shadow","visible_diffuse_reflection","visible_diffuse_transmission",
  "visible_glossy_reflection","visible_glossy_transmission","visible_mirror_reflection","visible_mirror_transmission","visible_volume"};return s;}
inline unsigned geometryVisibility(const rdl::SceneObject& o){
 need(get<rdl::Bool>(o,"visible_volume"),o.getName()+": a geometry hidden from volume phase rays falls back");
 const char* names[8]={"visible_in_camera","visible_shadow","visible_diffuse_reflection","visible_diffuse_transmission",
  "visible_glossy_reflection","visible_glossy_transmission","visible_mirror_reflection","visible_mirror_transmission"};
 unsigned m=0;for(int i=0;i<8;++i)if(get<rdl::Bool>(o,names[i]))m|=1u<<i;return m;}
inline bool isLight(const std::string& cls){return cls=="EnvLight"||cls=="RectLight"||cls=="DiskLight"||cls=="SphereLight"||cls=="DistantLight"||cls=="SpotLight"||cls=="CylinderLight"||cls=="MeshLight"||cls=="PortalLight";}
// A BaseVolume (moonray dso/volume/Base) as volume_core.h shades it: homogeneous (no attribute bound), not emissive;
// its ramps over the depth a ray travels through it as RampControl builds them.
// A VDB grid (xpu2_geom.dll's x2VdbRead, as geom::internal::VdbVolume reads it), kept as NanoVDB and freed by the DLL.
struct VdbGridData{X2VdbGrid raw{};std::string key,path,name;std::vector<float> dense;
 // A baked grid's making (bakeVolume), for the next preparation to keep it: what its values came from, and how it is read.
 const MeshData* bakeMesh=nullptr;std::vector<std::shared_ptr<const ix::Buffer>> bakeSignature;double bakeSpace[32]={};volume::Grid bakeView{};
 VdbGridData()=default;VdbGridData(const VdbGridData&)=delete;
 ~VdbGridData(){if(raw.data&&subdLibrary().vdbFree)subdLibrary().vdbFree(&raw);}};
// A grid of a file for the plan (read once a plan; the previous plan's while the file is unchanged): its index.
inline unsigned vdbGrid(Plan& plan,const Plan* previous,const std::string& path,const std::string& name,bool vec3,const std::string& owner){
 std::error_code ec;const auto stamp=fs::last_write_time(fs::path(path),ec);
 need(!ec,owner+": the VDB file \""+path+"\" cannot be read");
 const std::string key=path+"|"+name+(vec3?"|vec3|":"|float|")+std::to_string(stamp.time_since_epoch().count());
 for(size_t i=0;i<plan.volumeGrids.size();++i)if(plan.volumeGrids[i]->key==key)return unsigned(i);
 std::shared_ptr<VdbGridData> g;
 if(previous)for(const auto& old:previous->volumeGrids)if(old->key==key){g=old;break;}
 if(!g){const auto& lib=subdLibrary();need(lib.vdbRead&&lib.vdbFree,owner+": MoonRay's VDB reading (xpu2_geom.dll) did not load");
  g=std::make_shared<VdbGridData>();g->key=key;g->path=path;g->name=name;char error[512]={0};
  need(lib.vdbRead(path.c_str(),name.c_str(),vec3?1:0,&g->raw,error,sizeof error)!=0,owner+": "+std::string(error));}
 need(g->raw.linear!=0,owner+": a VDB grid with a frustum (non-linear) transform is not rendered yet");
 volume::Grid view;view.data=g->raw.data;view.vec3=vec3?1u:0u;
 rdl::Mat4d m;for(int i=0;i<4;++i)for(int j=0;j<4;++j)m[i][j]=g->raw.indexToWorld[4*i+j];
 const rdl::Mat4d inv=m.inverse();for(int i=0;i<4;++i)for(int j=0;j<4;++j)view.worldToIndex[4*i+j]=inv[i][j];
 plan.volumeGrids.push_back(std::move(g));plan.volumeGridViews.push_back(view);
 return unsigned(plan.volumeGrids.size()-1);}
// Distribution3D over a histogram (x fastest): each row's conditional x, each slice's marginal y, the marginal z, as
// Distribution1D records in the plan's emission pool (volume_emission.h's dist1D: integral, count, values, cdf).
inline unsigned emissionDist1D(std::vector<float>& pool,const float* func,int n){
 const unsigned at=unsigned(pool.size());std::vector<float> cdf(size_t(n)+1);cdf[0]=0;
 for(int i=1;i<n+1;++i)cdf[size_t(i)]=cdf[size_t(i-1)]+func[i-1]/float(n);
 const float funcInt=cdf[size_t(n)];
 if(funcInt==0)for(int i=1;i<n+1;++i)cdf[size_t(i)]=float(i)/float(n);else for(int i=1;i<n+1;++i)cdf[size_t(i)]/=funcInt;
 pool.push_back(funcInt);pool.push_back(float(n));pool.insert(pool.end(),func,func+n);pool.insert(pool.end(),cdf.begin(),cdf.end());return at;}
inline volume::Region emissionRegion(std::vector<float>& pool,const std::vector<float>& histogram,const int res[3],const rdl::Mat4d& distToWorld,
                                     float invUnitVolume,unsigned volumeIndex,unsigned visibility){
 volume::Region r{};r.volume=volumeIndex;r.visibility=visibility;for(int a=0;a<3;++a)r.res[a]=res[a];r.invUnitVolume=invUnitVolume;
 const rdl::Mat4d worldToDist=distToWorld.inverse();
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){r.distToWorld[3*i+j]=float(distToWorld[i][j]);r.worldToDist[3*i+j]=float(worldToDist[i][j]);}
 for(int j=0;j<3;++j){r.distToWorld[9+j]=float(distToWorld[3][j]);r.worldToDist[9+j]=float(worldToDist[3][j]);}
 const int nx=res[0],ny=res[1],nz=res[2];
 std::vector<float> xIntegrals,xyIntegrals;xIntegrals.resize(size_t(ny));xyIntegrals.resize(size_t(nz));
 r.conditionalX=unsigned(pool.size());
 std::vector<unsigned> rows;
 for(int z=0;z<nz;++z)for(int y=0;y<ny;++y){const unsigned at=emissionDist1D(pool,&histogram[size_t(z)*size_t(ny)*size_t(nx)+size_t(y)*size_t(nx)],nx);rows.push_back(at);}
 r.marginalY=unsigned(pool.size());
 for(int z=0;z<nz;++z){for(int y=0;y<ny;++y)xIntegrals[size_t(y)]=pool[rows[size_t(z*ny+y)]];
  const unsigned at=emissionDist1D(pool,xIntegrals.data(),ny);xyIntegrals[size_t(z)]=pool[at];}
 r.marginalZ=emissionDist1D(pool,xyIntegrals.data(),nz);
 return r;}
// Primitive::computeEmissionDistribution for a mesh's volume: about a thousand voxels over the mesh's render-space box,
// each the shader's emission luminance at its corner (homogeneous here: the same everywhere).
inline void runVolumeMaps(const void* context,const volume::Shader& s,V3 p,volume::Mapped& m);
inline void meshEmissionRegion(Plan& plan,const MeshData& mesh,unsigned volumeIndex,unsigned visibility,bool mapped=false){
 const volume::Shader& s=plan.volumes[volumeIndex];
 rdl::Mat4d w2r;for(int i=0;i<4;++i)for(int j=0;j<4;++j)w2r[i][j]=plan.world2render[4*i+j];
 float lo[3]={3.4e38f,3.4e38f,3.4e38f},hi[3]={-3.4e38f,-3.4e38f,-3.4e38f};
 for(const auto& p:mesh.positions){const double q[3]={p.x*w2r[0][0]+p.y*w2r[1][0]+p.z*w2r[2][0]+w2r[3][0],p.x*w2r[0][1]+p.y*w2r[1][1]+p.z*w2r[2][1]+w2r[3][1],
   p.x*w2r[0][2]+p.y*w2r[1][2]+p.z*w2r[2][2]+w2r[3][2]};for(int a=0;a<3;++a){lo[a]=std::min(lo[a],float(q[a]));hi[a]=std::max(hi[a],float(q[a]));}}
 const float len[3]={hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]};
 const float sCubed=len[0]*len[1]*len[2]/1000.0f,sz=std::cbrt(sCubed);
 need(std::isfinite(sz)&&sz>0.0f,"An emissive volume of no extent");
 const int res[3]={int(std::ceil(len[0]/sz)),int(std::ceil(len[1]/sz)),int(std::ceil(len[2]/sz))};
 need(res[0]>0&&res[1]>0&&res[2]>0&&size_t(res[0])*size_t(res[1])*size_t(res[2])<(1u<<24),"An emissive volume's distribution budget");
 const float e=0.299f*s.emission.x+0.587f*s.emission.y+0.114f*s.emission.z;
 std::vector<float> histogram(size_t(res[0])*size_t(res[1])*size_t(res[2]),e);
 if(mapped){const rdl::Mat4d r2w=w2r.inverse();float* h=histogram.data();
  for(int k=0;k<res[2];++k){const float z=lo[2]+float(k)*sz;for(int j=0;j<res[1];++j){const float y=lo[1]+float(j)*sz;for(int i=0;i<res[0];++i){const float x=lo[0]+float(i)*sz;
   const V3 pw{float(x*r2w[0][0]+y*r2w[1][0]+z*r2w[2][0]+r2w[3][0]),float(x*r2w[0][1]+y*r2w[1][1]+z*r2w[2][1]+r2w[3][1]),float(x*r2w[0][2]+y*r2w[1][2]+z*r2w[2][2]+r2w[3][2])};
   volume::Mapped m{s.attenuationColor,s.albedoColor,s.emissionColor,s.intensity,s.factor,s.emissionIntensity,s.g};runVolumeMaps(&plan,s,pw,m);
   const V3 em=volume::cmax0(volume::scaled(m.emissionColor,m.emissionIntensity));*h++=0.299f*em.x+0.587f*em.y+0.114f*em.z;}}}}
 rdl::Mat4d distToRender(sz,0,0,0,0,sz,0,0,0,0,sz,0,lo[0],lo[1],lo[2],1);
 plan.emissiveRegions.push_back(emissionRegion(plan.emissionPool,histogram,res,distToRender*w2r.inverse(),1.0f/sCubed,volumeIndex,visibility));}
// VdbGeometry with a VdbVolume (geom::internal::VdbVolume, the VdbVolume shader): its density grid and emission grid
// (each an rdl name of the file), the geometry's interpolation, the shader's opacity gain, colour, incandescence gain
// and anisotropy; its feature size (cube root of a voxel's render-space volume, VdbVolume::initializePhase2) and its
// world to grid world transform. Its boundary: the active voxels' index box grown by the quadratic sampler's reach
// (the density is zero outside), through the grid's and the node's transforms, as a closed mesh facing out.
inline volume::Shader admitVdbVolume(const rdl::SceneObject& g,const rdl::SceneObject& o,unsigned lightSet,Plan& plan,const Plan* previous,
                                     std::shared_ptr<MeshData>& box){
 {std::set<std::string> allowed{"model","interpolation","density_grid","emission_grid","velocity_grid","velocity_scale","velocity_sample_rate",
   "emission_sample_rate","node_xform","primitive_attributes"};
  // Its node transform may move over the shutter (s.moving: the world to grid lerped by the ray's time, as MoonRay's
  // VdbVolume lerps its render to index transforms).
  allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(g,allowed,{},{"node_xform"});}
 const std::string cls=o.getSceneClass().getName();
 need(cls=="VdbVolume",g.getName()+": a "+cls+" on a VDB geometry is not rendered yet");
 {std::set<std::string> all;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)all.insert((*i)->getName());attributes(o,all,all);}
 // Bound attributes: their map program after the maps' admission (admitVdbMaps); a bound incandescence's emissive
 // region with it (bakeVdbVolume).
 bool incandescenceBound=false;
 for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)if((*i)->getName()=="incandescence_gain_mult"&&o.getBinding(**i))incandescenceBound=true;
 const std::string path=get<rdl::String>(g,"model");need(!path.empty(),g.getName()+": no VDB file");
 volume::Shader s{};s.kind=volume::KindVdb;s.lightSet=lightSet;s.flags=0;s.bakedGrid=volume::NoGrid;s.program=shade::NoMapProgram;s.mapped=0;
 for(unsigned k=0;k<volume::MappedCount;++k)s.regs[k]=shade::MapNoRegister;
 s.densityGrid=vdbGrid(plan,previous,path,get<rdl::String>(g,"density_grid"),false,g.getName());
 const std::string emissionGrid=get<rdl::String>(g,"emission_grid");
 s.emissionGrid=emissionGrid.empty()?volume::NoGrid:vdbGrid(plan,previous,path,emissionGrid,true,g.getName());
 s.interpolation=unsigned(std::min(std::max(get<rdl::Int>(g,"interpolation"),0),2));
 auto rgb=[&](const char* n){auto c=get<rdl::Rgb>(o,n);need(std::isfinite(c.r)&&std::isfinite(c.g)&&std::isfinite(c.b),o.getName()+": invalid "+n);return V3{c.r,c.g,c.b};};
 s.gain=rgb("opacity_gain_mult");s.colorMult=rgb("color_mult");s.incandescence=rgb("incandescence_gain_mult");
 // The same as a mapped volume's base values (mappedAt): extinction gain, albedo, emission.
 s.attenuationColor=s.gain;s.albedoColor=s.colorMult;s.emissionColor=s.incandescence;s.emissionIntensity=1.0f;s.intensity=1.0f;s.factor=1.0f;
 {const float a=get<rdl::Float>(o,"anisotropy");need(std::isfinite(a),o.getName()+": invalid anisotropy");s.g=std::min(std::max(a,-1.0f),1.0f);}
 // A moving geometry (its node transform at the shutter's ends): its grid point at the ray's time.
 const rdl::Mat4d node=plan.motion.on?xformAt(g,0.0f):affine(g),node1=plan.motion.on?xformAt(g,1.0f):node;
 s.moving=plan.motion.on&&!(node==node1)?1u:0u;
 const auto& raw=plan.volumeGrids[s.densityGrid]->raw;
 // Velocity motion blur: the shutter's times (a frame's fraction of a second), the velocity grid and the density's
 // padded active box and cells (xpu2_geom.dll's x2VdbVelocity).
 int activeMin[3]={raw.bboxMin[0],raw.bboxMin[1],raw.bboxMin[2]},activeMax[3]={raw.bboxMax[0],raw.bboxMax[1],raw.bboxMax[2]};
 const int* cells=raw.blocks;unsigned cellCount=raw.blockCount,cellSize=raw.blockSize;
 {const std::string velocityName=get<rdl::String>(g,"velocity_grid");const float velocityScale=get<rdl::Float>(g,"velocity_scale");
  if(plan.motion.on&&!velocityName.empty()&&!scene_rdl2::math::isZero(velocityScale)){
   const auto& lib=subdLibrary();need(lib.vdbVelocity&&lib.vdbFree,g.getName()+": MoonRay's VDB velocity (xpu2_geom.dll) did not load");
   const rdl::SceneVariables& sv=g.getSceneClass().getSceneContext()->getSceneVariables();
   const float dt=1.0f/get<rdl::Float>(sv,"fps"),tOpen=dt*plan.motion.shutterOpen,tRange=dt*plan.motion.shutterClose-tOpen;
   auto data=std::make_shared<VdbGridData>();data->key="velocity|"+path+"|"+velocityName+"|"+g.getName();X2VdbVelocity v{};char error[512]={0};
   const int made=lib.vdbVelocity(path.c_str(),velocityName.c_str(),get<rdl::String>(g,"density_grid").c_str(),velocityScale,get<rdl::Float>(g,"velocity_sample_rate"),
                                  tOpen,tOpen+tRange,&v,error,sizeof error);
   need(made!=0,g.getName()+": "+std::string(error));
   if(made==1){data->raw=v.velocity;volume::Grid view;view.data=data->raw.data;view.vec3=1;
    rdl::Mat4d m;for(int i=0;i<4;++i)for(int j=0;j<4;++j)m[i][j]=data->raw.indexToWorld[4*i+j];
    const rdl::Mat4d inv=m.inverse();for(int i=0;i<4;++i)for(int j=0;j<4;++j)view.worldToIndex[4*i+j]=inv[i][j];
    s.velocityGrid=unsigned(plan.volumeGrids.size());plan.volumeGrids.push_back(std::move(data));plan.volumeGridViews.push_back(view);
    s.hasVelocity=1;s.tShutterOpen=tOpen;s.tShutterRange=tRange;
    for(int a=0;a<3;++a){activeMin[a]=v.densityMin[a];activeMax[a]=v.densityMax[a];}
    const auto& kept=plan.volumeGrids.back()->raw;cells=kept.blocks;cellCount=kept.blockCount;cellSize=kept.blockSize;}}}
 rdl::Mat4d i2g;for(int i=0;i<4;++i)for(int j=0;j<4;++j)i2g[i][j]=raw.indexToWorld[4*i+j];
 const rdl::Mat4d i2w=i2g*node;
 {const double det=i2w[0][0]*(i2w[1][1]*i2w[2][2]-i2w[1][2]*i2w[2][1])-i2w[0][1]*(i2w[1][0]*i2w[2][2]-i2w[1][2]*i2w[2][0])+i2w[0][2]*(i2w[1][0]*i2w[2][1]-i2w[1][1]*i2w[2][0]);
  s.featureSize=float(std::cbrt(det));need(std::isfinite(s.featureSize)&&s.featureSize>0.0f,g.getName()+": a degenerate VDB transform");}
 {auto rows=[](const rdl::Mat4d& m,float* out){for(int r=0;r<3;++r){for(int c=0;c<3;++c)out[4*r+c]=float(m[c][r]);out[4*r+3]=float(m[3][r]);}};
  rows(node.inverse(),s.worldToGrid);rows(node1.inverse(),s.worldToGrid1);rows(node,s.gridToWorld);rows(node1,s.gridToWorld1);}
 s.visibility=geometryVisibility(g)|0x300u;
 // Its emissive region (GeometryManager::getEmissiveRegions: secondary visibility and an emission grid): the grid's
 // active voxels' histogram by xpu2_geom.dll, each voxel's luminance times the shader's emission luminance.
 const float shaderEmission=0.299f*s.incandescence.x+0.587f*s.incandescence.y+0.114f*s.incandescence.z;
 if(!incandescenceBound&&s.emissionGrid!=volume::NoGrid&&(s.visibility&0x3fcu)&&!(s.incandescence.x==0&&s.incandescence.y==0&&s.incandescence.z==0)){
  const auto& lib=subdLibrary();need(lib.vdbHistogram&&lib.vdbHistogramFree,g.getName()+": MoonRay's VDB reading (xpu2_geom.dll) did not load");
  X2VdbHistogram h{};char error[512]={0};
  need(lib.vdbHistogram(path.c_str(),emissionGrid.c_str(),shaderEmission,&h,error,sizeof error)!=0,g.getName()+": "+std::string(error));
  try{
   const size_t n=size_t(h.res[0])*size_t(h.res[1])*size_t(h.res[2]);
   if(n>0){need(n<(1u<<26),g.getName()+": an emission grid too large for its distribution");
    std::vector<float> histogram(h.values,h.values+n);
    rdl::Mat4d i2p;for(int i=0;i<4;++i)for(int j=0;j<4;++j)i2p[i][j]=h.indexToWorld[4*i+j];
    const rdl::Mat4d i2w=i2p*node;
    const double det=i2w[0][0]*(i2w[1][1]*i2w[2][2]-i2w[1][2]*i2w[2][1])-i2w[0][1]*(i2w[1][0]*i2w[2][2]-i2w[1][2]*i2w[2][0])+i2w[0][2]*(i2w[1][0]*i2w[2][1]-i2w[1][1]*i2w[2][0]);
    const rdl::Mat4d distToIndex(1,0,0,0,0,1,0,0,0,0,1,0,h.pMin[0]-0.5,h.pMin[1]-0.5,h.pMin[2]-0.5,1);
    plan.emissiveRegions.push_back(emissionRegion(plan.emissionPool,histogram,h.res,distToIndex*i2w,float(1.0/det),unsigned(plan.volumes.size()),s.visibility));}
  }catch(...){lib.vdbHistogramFree(&h);throw;}
  lib.vdbHistogramFree(&h);}
 // The boundary box.
 // A moving geometry's box: the world box about both ends' (the march's empty stretches add nothing).
 auto m=std::make_shared<MeshData>();
 const double lo[3]={activeMin[0]-1.5,activeMin[1]-1.5,activeMin[2]-1.5},hi[3]={activeMax[0]+1.5,activeMax[1]+1.5,activeMax[2]+1.5};
 auto corner=[&](int i,const rdl::Mat4d& t){const double p[3]={i&1?hi[0]:lo[0],i&2?hi[1]:lo[1],i&4?hi[2]:lo[2]};
  return V3{float(p[0]*t[0][0]+p[1]*t[1][0]+p[2]*t[2][0]+t[3][0]),float(p[0]*t[0][1]+p[1]*t[1][1]+p[2]*t[2][1]+t[3][1]),float(p[0]*t[0][2]+p[1]*t[1][2]+p[2]*t[2][2]+t[3][2])};};
 std::vector<Tri> cellTriangles;
 if(s.moving){const rdl::Mat4d i2w1=i2g*node1;V3 bl{3.4e38f,3.4e38f,3.4e38f},bh{-3.4e38f,-3.4e38f,-3.4e38f};
  for(int i=0;i<8;++i)for(const V3& w:{corner(i,i2w),corner(i,i2w1)}){bl=V3{std::min(bl.x,w.x),std::min(bl.y,w.y),std::min(bl.z,w.z)};bh=V3{std::max(bh.x,w.x),std::max(bh.y,w.y),std::max(bh.z,w.z)};}
  for(int i=0;i<8;++i)m->positions.push_back(V3{i&1?bh.x:bl.x,i&2?bh.y:bl.y,i&4?bh.z:bl.z});}
 else if(cells&&cellCount){
  // VolumeRayIntersector's march: the runs of leaf cells along the ray (its corner-based index space: a cell from
  // its origin to origin + size), the union's surface the volume's boundary (a face where a cell meets no cell).
  std::set<std::array<int,3>> set;for(unsigned c=0;c<cellCount;++c)set.insert({cells[3*c],cells[3*c+1],cells[3*c+2]});
  std::map<std::array<int,3>,unsigned> vertexIds;
  auto vertexAt=[&](int x,int y,int z){const std::array<int,3> k{x,y,z};auto it=vertexIds.find(k);if(it!=vertexIds.end())return it->second;
   const double p[3]={double(x),double(y),double(z)};
   const V3 w{float(p[0]*i2w[0][0]+p[1]*i2w[1][0]+p[2]*i2w[2][0]+i2w[3][0]),float(p[0]*i2w[0][1]+p[1]*i2w[1][1]+p[2]*i2w[2][1]+i2w[3][1]),
              float(p[0]*i2w[0][2]+p[1]*i2w[1][2]+p[2]*i2w[2][2]+i2w[3][2])};
   const unsigned id=unsigned(m->positions.size());m->positions.push_back(w);vertexIds.emplace(k,id);return id;};
  const int n=int(cellSize);
  static const int dirs[6][3]={{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  for(const auto& c:set)for(const auto& dv:dirs){
   if(set.count({c[0]+dv[0]*n,c[1]+dv[1]*n,c[2]+dv[2]*n}))continue;
   const int axis=dv[0]?0:dv[1]?1:2,u=(axis+1)%3,w=(axis+2)%3;
   std::array<int,3> q[4];for(int k=0;k<4;++k){q[k]=c;if(dv[axis]>0)q[k][axis]+=n;}
   q[1][u]+=n;q[2][u]+=n;q[2][w]+=n;q[3][w]+=n;
   unsigned id[4];for(int k=0;k<4;++k)id[k]=vertexAt(q[k][0],q[k][1],q[k][2]);
   // Wound to face out of the cell (the box's sense: its normal along dv).
   const V3 a=m->positions[id[0]],b=m->positions[id[1]],e=m->positions[id[2]];
   const V3 nrm=surface::cross(surface::add(b,surface::mul(a,-1)),surface::add(e,surface::mul(a,-1)));
   const V3 out{float(dv[0]*i2w[0][0]+dv[1]*i2w[1][0]+dv[2]*i2w[2][0]),float(dv[0]*i2w[0][1]+dv[1]*i2w[1][1]+dv[2]*i2w[2][1]),
                float(dv[0]*i2w[0][2]+dv[1]*i2w[1][2]+dv[2]*i2w[2][2])};
   if(surface::dot(nrm,out)>=0.0f){cellTriangles.push_back({id[0],id[1],id[2]});cellTriangles.push_back({id[0],id[2],id[3]});}
   else{cellTriangles.push_back({id[0],id[2],id[1]});cellTriangles.push_back({id[0],id[3],id[2]});}}}
 else for(int i=0;i<8;++i)m->positions.push_back(corner(i,i2w));
 for(const V3& w:m->positions)X2_NEED(surface::finite(w)&&std::abs(w.x)<=10000&&std::abs(w.y)<=10000&&std::abs(w.z)<=10000,g.getName()+": a VDB volume out of bounds");
 static const unsigned faces[12][3]={{0,4,6},{0,6,2},{1,3,7},{1,7,5},{0,1,5},{0,5,4},{2,6,7},{2,7,3},{0,2,3},{0,3,1},{4,5,7},{4,7,6}};
 if(!cellTriangles.empty()){
  m->triangles=std::move(cellTriangles);
  const rdl::Mat4d w2i=i2w.inverse();m->gate=true;
  for(int r=0;r<3;++r){for(int c=0;c<3;++c)m->gateToIndex[4*r+c]=float(w2i[c][r]);m->gateToIndex[4*r+3]=float(w2i[3][r]);
   m->gateLo[r]=float(activeMin[r]);m->gateHi[r]=float(activeMax[r]);}}
 else for(const auto& f:faces)m->triangles.push_back(Tri{f[0],f[1],f[2]});
 {const auto w=node.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)m->worldToObject[i]=float(rows[i]);}
 box=std::move(m);
 return s;}
// Whether a BaseVolume has bound attributes it reads through maps (BaseVolume's evalColor and evalFloat; its ramps'
// depths, bindable, are read as values).
inline bool volumeMapped(const rdl::SceneObject& o){
 for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i){const rdl::Attribute& a=**i;
  if(a.isBindable()&&o.getBinding(a)&&a.getName().find("_depth")==std::string::npos)return true;}
 return false;}
inline volume::Shader admitVolume(const rdl::SceneObject& o,unsigned lightSet,bool mapped=false){
 need(o.getSceneClass().getName()=="BaseVolume",o.getName()+": only BaseVolume volume shaders are rendered");
 std::set<std::string> all;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)all.insert((*i)->getName());
 attributes(o,all,mapped?all:std::set<std::string>{});
 if(!mapped)need(!volumeMapped(o),o.getName()+": a volume with bound attributes is admitted by admitMappedVolume");
 auto rgb=[&](const char* n){auto c=get<rdl::Rgb>(o,n);need(std::isfinite(c.r)&&std::isfinite(c.g)&&std::isfinite(c.b),o.getName()+": invalid "+n);return V3{c.r,c.g,c.b};};
 auto f1=[&](const char* n){float f=get<rdl::Float>(o,n);need(std::isfinite(f),o.getName()+": invalid "+n);return f;};
 const V3 emission=rgb("emission_color");const float emissionIntensity=f1("emission_intensity");
 volume::Shader s{};
 // emission(): density times emission_intensity times emission_color (the camera's rays see it; it lights the scene
 // as an emissive region).
 s.emission=V3{emission.x*emissionIntensity,emission.y*emissionIntensity,emission.z*emissionIntensity};
 s.kind=volume::KindBase;s.densityGrid=s.emissionGrid=volume::NoGrid;s.bakedGrid=volume::NoGrid;
 s.emissionColor=emission;s.emissionIntensity=emissionIntensity;s.program=shade::NoMapProgram;s.mapped=0;
 for(unsigned k=0;k<volume::MappedCount;++k)s.regs[k]=shade::MapNoRegister;
 s.attenuationColor=rgb("attenuation_color");s.albedoColor=rgb("diffuse_color");
 s.intensity=f1("attenuation_intensity");s.factor=f1("attenuation_factor");
 s.g=std::min(std::max(f1("anisotropy"),-1.0f),1.0f);
 s.flags=(get<rdl::Bool>(o,"use_attenuation_ramp")?volume::UseAttenuationRamp:0u)|(get<rdl::Bool>(o,"use_diffuse_ramp")?volume::UseAlbedoRamp:0u)|
         (get<rdl::Bool>(o,"match_diffuse")?volume::MatchAlbedo:0u)|(get<rdl::Bool>(o,"invert_attenuation_color")?volume::InvertAttenuation:0u);
 s.lightSet=lightSet;
 s.attenuationMin=f1("attenuation_min_depth");s.attenuationMax=f1("attenuation_max_depth");
 s.albedoMin=f1("diffuse_min_depth");s.albedoMax=f1("diffuse_max_depth");
 s.densityMin=f1("density_min_depth");s.densityMax=f1("density_max_depth");
 auto colorRamp=[&](const char* pos,const char* values,const char* interps,shade::ColorRamp& r){
  const auto& p=get<rdl::FloatVector>(o,pos);const auto& v=get<rdl::RgbVector>(o,values);const auto& in=get<rdl::IntVector>(o,interps);
  need(p.size()<=shade::RampMaxPoints&&v.size()>=p.size()&&in.size()>=p.size(),o.getName()+": a volume ramp of more than ten points");
  float ip[shade::RampMaxPoints];shade::Color out[shade::RampMaxPoints];unsigned it[shade::RampMaxPoints];
  for(size_t i=0;i<p.size();++i){ip[i]=p[i];out[i]=shade::Color{v[i].r,v[i].g,v[i].b};need(in[i]>=0&&in[i]<=6,o.getName()+": invalid ramp interpolation");it[i]=unsigned(in[i]);}
  shade::rampInit(r,unsigned(p.size()),ip,out,it,shade::RampRgb);};
 colorRamp("attenuation_distances","attenuation_colors","attenuation_interpolations",s.attenuationRamp);
 colorRamp("diffuse_distances","diffuse_colors","diffuse_interpolations",s.albedoRamp);
 {const auto& p=get<rdl::FloatVector>(o,"density_distances");const auto& v=get<rdl::FloatVector>(o,"densities");const auto& in=get<rdl::IntVector>(o,"density_interpolations");
  need(p.size()<=shade::RampMaxPoints&&v.size()>=p.size()&&in.size()>=p.size(),o.getName()+": a volume ramp of more than ten points");
  float ip[shade::RampMaxPoints],out[shade::RampMaxPoints];unsigned it[shade::RampMaxPoints];
  for(size_t i=0;i<p.size();++i){ip[i]=p[i];out[i]=v[i];need(in[i]>=0&&in[i]<=6,o.getName()+": invalid ramp interpolation");it[i]=unsigned(in[i]);}
  shade::rampInit(s.densityRamp,unsigned(p.size()),ip,out,it);}
 return s;
}
// The scene indexer's families of an object and of everything it shades with (its bound maps, the materials and
// objects its attributes name), from the router's decisions; each object once.
inline uint64_t indexedFamilies(const rdl::SceneObject& o,const CanonicalView& frame,std::map<const rdl::SceneObject*,uint64_t>& seen){
 const auto found=seen.find(&o);if(found!=seen.end())return found->second;
 seen[&o]=0;
 const auto& decisions=frame->routing->decisions;const std::string id=ri::objectId(o.getName());
 uint64_t f=decisions.count(id)?decisions.at(id).families:0;
 for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i){
  const rdl::Attribute& a=**i;
  if(a.isBindable())if(const rdl::SceneObject* b=o.getBinding(a))f|=indexedFamilies(*b,frame,seen);
  if(a.getType()==rdl::TYPE_SCENE_OBJECT){if(const rdl::SceneObject* c=o.get<rdl::SceneObject*>(a.getName()))f|=indexedFamilies(*c,frame,seen);}
  else if(a.getType()==rdl::TYPE_SCENE_OBJECT_VECTOR){for(const rdl::SceneObject* c:o.get<rdl::SceneObjectVector>(a.getName()))if(c)f|=indexedFamilies(*c,frame,seen);}
 }
 seen[&o]=f;return f;
}
inline std::vector<std::shared_ptr<const ix::Buffer>> signature(const rdl::SceneObject& o,const CanonicalView& frame){
 auto entity=frame->routing->scene->entities.at(ri::objectId(o.getName()));std::vector<std::shared_ptr<const ix::Buffer>> value;
 for(auto& id:entity->dependencies){auto attr=frame->routing->scene->entities.at(id);for(auto& pair:attr->buffers)value.push_back(pair.second);}return value;
}
// An object's signature without its node_xform (MeshData::shape): the same buffers while nothing but where it is changed.
inline std::vector<std::shared_ptr<const ix::Buffer>> shapeSignature(const rdl::SceneObject& o,const CanonicalView& frame){
 auto entity=frame->routing->scene->entities.at(ri::objectId(o.getName()));const std::string placement=ri::attributeId(o.getName(),"node_xform");
 std::vector<std::shared_ptr<const ix::Buffer>> value;
 for(auto& id:entity->dependencies){if(id==placement)continue;auto attr=frame->routing->scene->entities.at(id);for(auto& pair:attr->buffers)value.push_back(pair.second);}return value;
}
// An object's signature with every object its attributes bind or name, through and through (a displacement and its
// maps, their maps): the same buffers while none of them has changed.
inline std::vector<std::shared_ptr<const ix::Buffer>> deepSignature(const rdl::SceneObject& o,const CanonicalView& frame){
 const auto& entities=frame->routing->scene->entities;std::vector<std::shared_ptr<const ix::Buffer>> value;
 std::set<std::string> seen;std::vector<std::string> todo{ri::objectId(o.getName())};
 while(!todo.empty()){const std::string id=todo.back();todo.pop_back();if(!seen.insert(id).second)continue;
  const auto object=entities.find(id);if(object==entities.end())continue;
  for(const auto& attribute:object->second->dependencies){const auto a=entities.find(attribute);if(a==entities.end())continue;
   for(const auto& pair:a->second->buffers)value.push_back(pair.second);
   for(const auto& bound:a->second->dependencies)todo.push_back(bound);}}
 return value;
}
}
#include "map_admission.h"
#include "dwa_admission.h"
#include "mesh_displacement.h"
#include "coarse_geometry.h"
namespace xpu2::interactive::cpu {
// A BaseVolume with bound attributes (KindBaseMapped): its attenuation colour, intensity and factor, diffuse colour,
// emission colour and intensity and anisotropy, each bound one a register of one map program, the rest as admitVolume
// reads them. Its extinction is baked or evaluated at the render origin once the textures are read (bakeVolume).
inline volume::Shader admitMappedVolume(const rdl::SceneObject& o,unsigned lightSet,Plan& plan,TextureContext& textureContext){
 volume::Shader s=admitVolume(o,lightSet,true);
 s.kind=volume::KindBaseMapped;s.featureSize=1.0f;
 MapProgramBuilder program{plan.materialPool,textureContext};
 static const char* const names[volume::MappedCount]={"attenuation_color","attenuation_intensity","attenuation_factor","diffuse_color","emission_color",
                                                     "emission_intensity","anisotropy"};
 for(unsigned k=0;k<volume::MappedCount;++k){
  const rdl::Attribute* a=nullptr;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)if((*i)->getName()==names[k])a=*i;
  const rdl::SceneObject* bound=a?o.getBinding(*a):nullptr;
  if(bound){s.regs[k]=program.reg(*bound);s.mapped|=1u<<k;}}
 need(!program.polyVertices,o.getName()+": a polygon map in a volume is not rendered");
 s.program=program.finish();
 s.originAttenuation=(s.flags&volume::MatchAlbedo)?s.albedoColor:s.attenuationColor;s.originIntensity=s.intensity;s.originFactor=s.factor;
 return s;}
// A mapped volume's bound values at p (world): its map program on a State with P alone (MoonRay's isect.init and
// setP), each bound value times its map's colour (evalColor) or the colour's average (evalFloat), unless zero.
template<class Tex> inline void volumeMapped(const volume::Shader& s,const float* pool,const Tex& textures,const float* worldToRender,V3 p,volume::Mapped& m){
 if(!s.mapped||s.program==shade::NoMapProgram)return;
 shade::State st;st.P=p;st.Ng=st.N=st.dPds=st.dPdt=st.wo=V3{0,0,0};st.worldToRender=worldToRender;st.worldToObject=s.worldToGrid;
 shade::Color regs[shade::MapMaxRegisters];
 if(shade::runMapProgram(pool,s.program,st,textures,regs)!=tex::Ready)return;
 auto color=[&](unsigned k,V3& v){if(!(s.mapped>>k&1)||(v.x==0.0f&&v.y==0.0f&&v.z==0.0f))return;const shade::Color c=regs[s.regs[k]];v=V3{v.x*c.x,v.y*c.y,v.z*c.z};};
 auto scalar=[&](unsigned k,float& f){if(!(s.mapped>>k&1)||shade::isZero(f))return;const shade::Color c=regs[s.regs[k]];f*=(c.x+c.y+c.z)/3.0f;};
 color(volume::MapAttenuationColor,m.attenuationColor);scalar(volume::MapAttenuationIntensity,m.attenuationIntensity);
 scalar(volume::MapAttenuationFactor,m.attenuationFactor);color(volume::MapAlbedo,m.albedoColor);color(volume::MapEmissionColor,m.emissionColor);
 scalar(volume::MapEmissionIntensity,m.emissionIntensity);scalar(volume::MapAnisotropy,m.anisotropy);}
// A VdbVolume's bound attributes (VdbVolume.cc): opacity_gain_mult, color_mult, incandescence_gain_mult and
// anisotropy, a register each of one map program.
inline void admitVdbMaps(volume::Shader& s,const rdl::SceneObject& o,Plan& plan,TextureContext& textureContext){
 static const std::pair<const char*,unsigned> names[]={{"opacity_gain_mult",volume::MapAttenuationColor},{"color_mult",volume::MapAlbedo},
                                                       {"incandescence_gain_mult",volume::MapEmissionColor},{"anisotropy",volume::MapAnisotropy}};
 MapProgramBuilder program{plan.materialPool,textureContext};
 for(const auto& [name,k]:names){const rdl::Attribute* a=nullptr;
  for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)if((*i)->getName()==name)a=*i;
  const rdl::SceneObject* bound=a?o.getBinding(*a):nullptr;if(bound){s.regs[k]=program.reg(*bound);s.mapped|=1u<<k;}}
 need(!program.polyVertices,o.getName()+": a polygon map in a volume is not rendered");
 s.program=program.finish();}
inline void runVolumeMaps(const void* context,const volume::Shader& s,V3 p,volume::Mapped& m){
 const Plan& plan=*static_cast<const Plan*>(context);
 const displace::Textures textures{plan.textureImages.data(),unsigned(plan.textureImages.size())};
 volumeMapped(s,plan.materialPool.data(),textures,plan.worldToRender,p,m);}
inline volume::Mapped mappedAt(const Plan& plan,const volume::Shader& s,V3 p){
 volume::Mapped m{s.attenuationColor,s.albedoColor,s.emissionColor,s.intensity,s.factor,s.emissionIntensity,s.g};runVolumeMaps(&plan,s,p,m);return m;}
// A VdbVolume's bound opacity gain baked (VdbVolume::bakeVolumeShaderDensityMap): a dense grid over the density's
// active box in its index space at rez (bake_resolution_mode: one, bake_divisions over the box's longest side, or the
// voxel size over bake_voxel_size), each voxel extinct() of density one with its maps at the voxel's render point (no
// geometry), read at the nearest voxel; no resolution: a gain of one. Its emissive region with a bound incandescence:
// the histogram with the emission's luminance at each voxel (x2VdbHistogramMapped).
inline void bakeVdbVolume(Plan& plan,unsigned volumeIndex,const rdl::SceneObject& geometry,const rdl::SceneObject& shader){
 volume::Shader& s=plan.volumes[volumeIndex];
 const rdl::Mat4d node=plan.motion.on?xformAt(geometry,0.0f):affine(geometry);
 auto toWorld=[&](double x,double y,double z){return V3{float(x*node[0][0]+y*node[1][0]+z*node[2][0]+node[3][0]),float(x*node[0][1]+y*node[1][1]+z*node[2][1]+node[3][1]),
                                                         float(x*node[0][2]+y*node[1][2]+z*node[2][2]+node[3][2])};};
 if(s.mapped&(1u<<volume::MapAttenuationColor)){
  const auto& raw=plan.volumeGrids[s.densityGrid]->raw;
  rdl::Mat4d i2g;for(int i=0;i<4;++i)for(int j=0;j<4;++j)i2g[i][j]=raw.indexToWorld[4*i+j];
  double rez[3]={1,1,1};
  const int mode=get<rdl::Int>(shader,"bake_resolution_mode");
  if(mode==1){const float maxDivisions=float(std::max(raw.bboxMax[0]-raw.bboxMin[0],std::max(raw.bboxMax[1]-raw.bboxMin[1],raw.bboxMax[2]-raw.bboxMin[2])));
   const double r=double(float(get<rdl::Int>(shader,"bake_divisions"))/maxDivisions);rez[0]=rez[1]=rez[2]=r;}
  else if(mode==2){const float voxel=get<rdl::Float>(shader,"bake_voxel_size");for(int a=0;a<3;++a)rez[a]=raw.voxelSize[a]/double(voxel);}
  if(rez[0]<=0.0||rez[1]<=0.0||rez[2]<=0.0){s.gain=V3{1,1,1};}
  else{
   int lo[3],hi[3];for(int a=0;a<3;++a){lo[a]=int(raw.bboxMin[a]*rez[a]);hi[a]=int(raw.bboxMax[a]*rez[a]);}
   const int dims[3]={hi[0]-lo[0]+1,hi[1]-lo[1]+1,hi[2]-lo[2]+1};
   need(dims[0]>0&&dims[1]>0&&dims[2]>0&&size_t(dims[0])*size_t(dims[1])*size_t(dims[2])<=(size_t(1)<<26),shader.getName()+": a baked VDB gain beyond the budget");
   // The baked grid's index to grid: the density's after preScale(1 / rez).
   const rdl::Mat4d pre(1.0/rez[0],0,0,0,0,1.0/rez[1],0,0,0,0,1.0/rez[2],0,0,0,0,1);const rdl::Mat4d bakedToGrid=pre*i2g;
   auto data=std::make_shared<VdbGridData>();data->key="baked-vdb|"+shader.getName()+"|"+std::to_string(volumeIndex);
   data->dense.assign(size_t(dims[0])*size_t(dims[1])*size_t(dims[2])*3,0.0f);
   const unsigned threads=std::max(1u,std::min(16u,std::thread::hardware_concurrency()));
   std::vector<std::thread> pool;std::atomic<int> nextSlice{0};std::mutex failLock;std::string failure;
   for(unsigned t=0;t<threads;++t)pool.emplace_back([&]{
    try{for(int kk=nextSlice++;kk<dims[2];kk=nextSlice++)for(int jj=0;jj<dims[1];++jj)for(int ii=0;ii<dims[0];++ii){
     const double ijk[3]={double(ii+lo[0]),double(jj+lo[1]),double(kk+lo[2])};
     const double pg[3]={ijk[0]*bakedToGrid[0][0]+ijk[1]*bakedToGrid[1][0]+ijk[2]*bakedToGrid[2][0]+bakedToGrid[3][0],
                         ijk[0]*bakedToGrid[0][1]+ijk[1]*bakedToGrid[1][1]+ijk[2]*bakedToGrid[2][1]+bakedToGrid[3][1],
                         ijk[0]*bakedToGrid[0][2]+ijk[1]*bakedToGrid[1][2]+ijk[2]*bakedToGrid[2][2]+bakedToGrid[3][2]};
     const V3 pw=toWorld(double(float(pg[0])),double(float(pg[1])),double(float(pg[2])));
     const volume::Mapped m=mappedAt(plan,s,pw);
     const V3 e=volume::cmax0(m.attenuationColor);
     float* out=data->dense.data()+3*((size_t(kk)*size_t(dims[1])+size_t(jj))*size_t(dims[0])+size_t(ii));out[0]=e.x;out[1]=e.y;out[2]=e.z;}}
    catch(const std::exception& e){std::lock_guard<std::mutex> l(failLock);if(failure.empty())failure=e.what();}});
   for(auto& t:pool)t.join();
   need(failure.empty(),shader.getName()+": "+failure);
   volume::Grid view;view.dense=data->dense.data();view.vec3=1;for(int a=0;a<3;++a){view.dims[a]=dims[a];view.denseMin[a]=lo[a];}
   const rdl::Mat4d gridToBaked=bakedToGrid.inverse();for(int i=0;i<4;++i)for(int j=0;j<4;++j)view.worldToIndex[4*i+j]=gridToBaked[i][j];
   s.bakedGrid=unsigned(plan.volumeGrids.size());plan.volumeGrids.push_back(std::move(data));plan.volumeGridViews.push_back(view);}}
 if((s.mapped&(1u<<volume::MapEmissionColor))&&s.emissionGrid!=volume::NoGrid&&(s.visibility&0x3fcu)){
  const auto& lib=subdLibrary();need(lib.vdbHistogramMapped&&lib.vdbHistogramFree,geometry.getName()+": MoonRay's VDB reading (xpu2_geom.dll) did not load");
  struct Context{Plan* plan;const volume::Shader* s;decltype(toWorld)* toWorld;};Context ctx{&plan,&s,&toWorld};
  const X2VdbShaderFn fn=[](void* user,const double* p)->float{auto& c=*static_cast<Context*>(user);
   const volume::Mapped m=mappedAt(*c.plan,*c.s,(*c.toWorld)(double(float(p[0])),double(float(p[1])),double(float(p[2]))));
   const V3 e=volume::cmax0(m.emissionColor);return 0.299f*e.x+0.587f*e.y+0.114f*e.z;};
  X2VdbHistogram h{};char error[512]={0};
  need(lib.vdbHistogramMapped(get<rdl::String>(geometry,"model").c_str(),get<rdl::String>(geometry,"emission_grid").c_str(),fn,&ctx,&h,error,sizeof error)!=0,
       geometry.getName()+": "+std::string(error));
  try{
   const size_t n=size_t(h.res[0])*size_t(h.res[1])*size_t(h.res[2]);
   if(n>0){need(n<(1u<<26),geometry.getName()+": an emission grid too large for its distribution");
    std::vector<float> histogram(h.values,h.values+n);
    rdl::Mat4d i2p;for(int i=0;i<4;++i)for(int j=0;j<4;++j)i2p[i][j]=h.indexToWorld[4*i+j];
    const rdl::Mat4d i2w=i2p*node;
    const double det=i2w[0][0]*(i2w[1][1]*i2w[2][2]-i2w[1][2]*i2w[2][1])-i2w[0][1]*(i2w[1][0]*i2w[2][2]-i2w[1][2]*i2w[2][0])+i2w[0][2]*(i2w[1][0]*i2w[2][1]-i2w[1][1]*i2w[2][0]);
    const rdl::Mat4d distToIndex(1,0,0,0,0,1,0,0,0,0,1,0,h.pMin[0]-0.5,h.pMin[1]-0.5,h.pMin[2]-0.5,1);
    plan.emissiveRegions.push_back(emissionRegion(plan.emissionPool,histogram,h.res,distToIndex*i2w,float(1.0/det),volumeIndex,s.visibility));}
  }catch(...){lib.vdbHistogramFree(&h);throw;}
  lib.vdbHistogramFree(&h);}}
// A mapped volume's extinction once the textures are read. On a polygon mesh with a bound attenuation colour, intensity
// or factor (Primitive::bakeVolumeShaderDensityMap): a dense Vec3 grid over the mesh's local bounds (its render-space
// bounds through prim to render's inverse), rez voxels along its longest world side (bake_resolution_mode: 100,
// bake_divisions, or bake_voxel_size), each voxel's extinct() of density one at its index point, read at the nearest
// voxel; its feature size the grid's largest voxel side. Otherwise (no bound extinction, another geometry, or no
// resolution) extinct() of MoonRay's empty state: its maps at the render origin.
inline void bakeVolume(Plan& plan,unsigned volumeIndex,const MeshData& mesh,const rdl::SceneObject& shader,const rdl::Mat4d& nodeXform,bool polygons,
                       const Plan* previous=nullptr,const std::vector<std::shared_ptr<const ix::Buffer>>* signature=nullptr){
 namespace sm=scene_rdl2::math;
 volume::Shader& s=plan.volumes[volumeIndex];
 rdl::Mat4d w2r;for(int i=0;i<4;++i)for(int j=0;j<4;++j)w2r[i][j]=plan.world2render[4*i+j];
 const rdl::Mat4d r2w=w2r.inverse();
 const bool extinctionMapped=(s.mapped&((1u<<volume::MapAttenuationColor)|(1u<<volume::MapAttenuationIntensity)|(1u<<volume::MapAttenuationFactor)))!=0;
 int rez=100;
 {const int mode=get<rdl::Int>(shader,"bake_resolution_mode");if(mode==1)rez=get<rdl::Int>(shader,"bake_divisions");}
 // Render-space bounds of the vertices (computeAABB), then local and world bounds (transformBBox, in floats).
 sm::BBox3f renderBox(sm::Vec3f(3.402823466e+38f),sm::Vec3f(-3.402823466e+38f));
 auto toRender=[&](const V3& p){return sm::Vec3f(float(p.x*w2r[0][0]+p.y*w2r[1][0]+p.z*w2r[2][0]+w2r[3][0]),float(p.x*w2r[0][1]+p.y*w2r[1][1]+p.z*w2r[2][1]+w2r[3][1]),
                                                  float(p.x*w2r[0][2]+p.y*w2r[1][2]+p.z*w2r[2][2]+w2r[3][2]));};
 for(const auto& p:mesh.positions)renderBox.extend(toRender(p));
 for(const auto& step:mesh.steps)for(const auto& p:step)renderBox.extend(toRender(p));
 const sm::Mat4f primToRender=sm::toFloat(nodeXform*w2r);
 const sm::BBox3f localBox=sm::transformBBox(primToRender.inverse(),renderBox);
 const sm::BBox3f worldBox=sm::transformBBox(sm::toFloat(nodeXform),localBox);
 const sm::Vec3f worldSize=worldBox.upper-worldBox.lower;const float maxSize=std::max(worldSize.x,std::max(worldSize.y,worldSize.z));
 if(get<rdl::Int>(shader,"bake_resolution_mode")==2)rez=int(std::ceil(maxSize/get<rdl::Float>(shader,"bake_voxel_size")));
 if(!polygons||!extinctionMapped||rez<=0){
  // The empty state's P: the render origin.
  const V3 origin{float(r2w[3][0]),float(r2w[3][1]),float(r2w[3][2])};
  const volume::Mapped m=mappedAt(plan,s,origin);
  s.originAttenuation=(s.flags&volume::MatchAlbedo)?m.albedoColor:m.attenuationColor;s.originIntensity=m.attenuationIntensity;s.originFactor=m.attenuationFactor;
  return;}
 const sm::Vec3f upper=float(rez)*worldSize/maxSize;
 const int dims[3]={int(upper.x)+1,int(upper.y)+1,int(upper.z)+1};
 need(dims[0]>0&&dims[1]>0&&dims[2]>0&&size_t(dims[0])*size_t(dims[1])*size_t(dims[2])<=(size_t(1)<<26),shader.getName()+": a baked volume grid beyond the budget");
 const sm::Vec3f scale=(localBox.upper-localBox.lower)/upper;
 // index to render: ((ijk * scale) + localMin) * primToRender, in doubles (createLinearTransform, preTranslate, preScale).
 rdl::Mat4d l2r;for(int i=0;i<4;++i)for(int j=0;j<4;++j)l2r[i][j]=double(primToRender[i][j]);
 const rdl::Mat4d st(double(scale.x),0,0,0,0,double(scale.y),0,0,0,0,double(scale.z),0,0,0,0,1);
 const rdl::Mat4d tr(1,0,0,0,0,1,0,0,0,0,1,0,double(localBox.lower.x),double(localBox.lower.y),double(localBox.lower.z),1);
 const rdl::Mat4d indexToRender=st*tr*l2r;
 // The same volume, from the same mesh, maps, place and render space as the last preparation's: its grid kept. Any edit of
 // a scene with a baked volume baked it again (0.56 s on the car).
 const std::string bakedKey="baked|"+shader.getName()+"|"+std::to_string(volumeIndex);
 double bakedSpace[32];for(int i=0;i<4;++i)for(int j=0;j<4;++j){bakedSpace[4*i+j]=nodeXform[i][j];bakedSpace[16+4*i+j]=w2r[i][j];}
 if(previous&&signature)for(const auto& old:previous->volumeGrids)if(old->key==bakedKey&&old->bakeMesh==&mesh&&old->bakeSignature==*signature&&std::memcmp(old->bakeSpace,bakedSpace,sizeof bakedSpace)==0){
  s.densityGrid=unsigned(plan.volumeGrids.size());plan.volumeGrids.push_back(old);plan.volumeGridViews.push_back(old->bakeView);
  double kept=0;for(int a=0;a<3;++a)kept=std::max(kept,std::sqrt(indexToRender[a][0]*indexToRender[a][0]+indexToRender[a][1]*indexToRender[a][1]+indexToRender[a][2]*indexToRender[a][2]));
  s.featureSize=float(kept);return;}
 auto g=std::make_shared<VdbGridData>();g->key=bakedKey;g->dense.assign(size_t(dims[0])*size_t(dims[1])*size_t(dims[2])*3,0.0f);
 // Each voxel's extinct() of density one with its maps at the voxel's index point, sliced over the threads.
 const unsigned threads=std::max(1u,std::min(16u,std::thread::hardware_concurrency()));
 std::vector<std::thread> pool;std::atomic<int> nextSlice{0};std::mutex failLock;std::string failure;
 for(unsigned t=0;t<threads;++t)pool.emplace_back([&]{
  try{for(int k=nextSlice++;k<dims[2];k=nextSlice++)for(int j=0;j<dims[1];++j)for(int i=0;i<dims[0];++i){
   const double pr[3]={i*indexToRender[0][0]+j*indexToRender[1][0]+k*indexToRender[2][0]+indexToRender[3][0],
                       i*indexToRender[0][1]+j*indexToRender[1][1]+k*indexToRender[2][1]+indexToRender[3][1],
                       i*indexToRender[0][2]+j*indexToRender[1][2]+k*indexToRender[2][2]+indexToRender[3][2]};
   const sm::Vec3f p=sm::Vec3f(float(pr[0]),float(pr[1]),float(pr[2]));
   const V3 pw{float(p.x*r2w[0][0]+p.y*r2w[1][0]+p.z*r2w[2][0]+r2w[3][0]),float(p.x*r2w[0][1]+p.y*r2w[1][1]+p.z*r2w[2][1]+r2w[3][1]),
               float(p.x*r2w[0][2]+p.y*r2w[1][2]+p.z*r2w[2][2]+r2w[3][2])};
   const volume::Mapped m=mappedAt(plan,s,pw);
   const V3 color=volume::attenuationColorFrom(s,(s.flags&volume::MatchAlbedo)?m.albedoColor:m.attenuationColor,-1.0f);
   const V3 e=volume::cmax0(volume::scaled(volume::times(volume::scaled(V3{1,1,1},m.attenuationIntensity),color),m.attenuationFactor));
   float* out=g->dense.data()+3*((size_t(k)*size_t(dims[1])+size_t(j))*size_t(dims[0])+size_t(i));out[0]=e.x;out[1]=e.y;out[2]=e.z;}}
  catch(const std::exception& e){std::lock_guard<std::mutex> l(failLock);if(failure.empty())failure=e.what();}});
 for(auto& t:pool)t.join();
 need(failure.empty(),shader.getName()+": "+failure);
 // The grid read through the mesh's world to object (the volume's worldToGrid), then object to index.
 volume::Grid view;view.dense=g->dense.data();view.vec3=1;for(int a=0;a<3;++a)view.dims[a]=dims[a];
 const rdl::Mat4d objectToIndex=(st*tr).inverse();for(int i=0;i<4;++i)for(int j=0;j<4;++j)view.worldToIndex[4*i+j]=objectToIndex[i][j];
 g->bakeMesh=&mesh;if(signature)g->bakeSignature=*signature;std::memcpy(g->bakeSpace,bakedSpace,sizeof bakedSpace);g->bakeView=view;
 s.densityGrid=unsigned(plan.volumeGrids.size());plan.volumeGrids.push_back(std::move(g));plan.volumeGridViews.push_back(view);
 // transform->voxelSize(): each index axis's length in render space; the feature size their largest.
 double fs=0;for(int a=0;a<3;++a)fs=std::max(fs,std::sqrt(indexToRender[a][0]*indexToRender[a][0]+indexToRender[a][1]*indexToRender[a][1]+indexToRender[a][2]*indexToRender[a][2]));
 s.featureSize=float(fs);}
// A NormalDisplacement or a tangent-space VectorDisplacement (mesh_displacement.h), its maps in a program of its own.
inline displace::Shader admitDisplacement(const rdl::SceneObject& o,std::vector<float>& pool,TextureContext& textures,int depth=0){
 const std::string cls=o.getSceneClass().getName();
 need(depth<16,o.getName()+": displacement tree too deep");
 // CombineDisplacement: its two inputs (each a displacement or none), their scales and the operation.
 if(cls=="CombineDisplacement"){
  attributes(o,{"operation","input_1","scale_1","input_2","scale_2"});displace::Shader d;d.kind=displace::Shader::Combine;
  d.operation=get<rdl::Int>(o,"operation");d.scale1=get<rdl::Float>(o,"scale_1");d.scale2=get<rdl::Float>(o,"scale_2");
  need(std::isfinite(d.scale1)&&std::isfinite(d.scale2),o.getName()+": invalid scale");
  for(int k=1;k<=2;++k){const rdl::SceneObject* in=get<rdl::SceneObject*>(o,("input_"+std::to_string(k)).c_str());
   if(!in)continue;need(in->isA<rdl::Displacement>(),o.getName()+": an input that is not a displacement");
   auto s=std::make_shared<displace::Shader>(admitDisplacement(*in,pool,textures,depth+1));(k==1?d.input1:d.input2)=s;}
  return d;
 }
 // SwitchDisplacement: its choice (cycled through 64) at preparation, none for an empty one.
 if(cls=="SwitchDisplacement"){
  // choice is a float read as an int (getAttrChoice): a binding on it is not evaluated.
  {std::set<std::string> all;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)all.insert((*i)->getName());attributes(o,all,{"choice"});}
  const int choice=int(get<rdl::Float>(o,"choice"))%64;displace::Shader none;none.kind=displace::Shader::None;
  if(choice<0)return none;
  const rdl::SceneObject* in=get<rdl::SceneObject*>(o,("displacement"+std::to_string(choice)).c_str());
  if(!in)return none;need(in->isA<rdl::Displacement>(),o.getName()+": its choice is not a displacement");
  return admitDisplacement(*in,pool,textures,depth+1);
 }
 MapProgramBuilder maps{pool,textures};displace::Shader d;
 auto bound=[&](const char* n)->unsigned{const rdl::SceneObject* m=o.getBinding(*o.getSceneClass().getAttribute(n));return m?maps.reg(*m):shade::MapNoRegister;};
 auto f1=[&](const char* n){float f=get<rdl::Float>(o,n);need(std::isfinite(f),o.getName()+": invalid "+n);return f;};
 if(cls=="NormalDisplacement"){
  attributes(o,{"height","zero_value","height_multiplier","bound_padding"},{"height","height_multiplier"});
  d.kind=displace::Shader::Normal;d.height=f1("height");d.zero=f1("zero_value");d.multiplier=f1("height_multiplier");
  d.heightReg=bound("height");d.multiplierReg=bound("height_multiplier");
 }else if(cls=="VectorDisplacement"){
  attributes(o,{"vector","factor","source_space","tangent_space_style","bound_padding"},{"vector"});
  const int sourceSpace=get<rdl::Int>(o,"source_space");need(sourceSpace==0||sourceSpace==1,o.getName()+": invalid source space");
  const int style=get<rdl::Int>(o,"tangent_space_style");need(style==0||style==1,o.getName()+": invalid tangent space style");
  const auto v=get<rdl::Vec3f>(o,"vector");d.kind=displace::Shader::Vector;d.vector=V3{v.x,v.y,v.z};need(surface::finite(d.vector),o.getName()+": invalid vector");
  d.factor=f1("factor");d.style=style;d.vectorReg=bound("vector");d.objectSpace=sourceSpace;
 }else throw Unsupported(o.getName()+": displacement class "+cls+" is not rendered yet");
 need(!maps.polyVertices&&maps.required.empty(),o.getName()+": polygon and attribute-requiring maps in a displacement are not rendered yet");
 d.program=maps.finish();
 return d;
}
// A displaced subdivision surface's displacements (xpu2_geom.dll's callback, subdDisplace): each Displacement of its
// assignments as XPU2 evaluates it, the material pool and textures its maps read, and render to world space and back
// (MoonRay displaces in render space, XPU2's displacement takes world space). A failure inside the tessellation is
// kept and thrown after it (check).
struct SubdDisplacement{
 std::map<const void*,displace::Shader> shaders;std::vector<const tex::Image*> images;const float* pool=nullptr;const float* worldToRender=nullptr;
 rdl::Mat4d w2r,r2w;std::mutex lock;std::string failure;float worldToObject[12]={1,0,0,0,0,1,0,0,0,0,1,0};
 void check(const rdl::SceneObject& o){need(failure.empty(),o.getName()+": "+failure);}
 // A worker's own copy of the material pool (TessellationJob): the plan's grows, and moves, as the loop goes on.
 std::vector<float> ownPool;
};
// A displacement for a worker (TessellationJob): the same shaders, images and transforms, a lock and failure of its own.
inline std::shared_ptr<SubdDisplacement> copyDisplacement(const SubdDisplacement& d){auto c=std::make_shared<SubdDisplacement>();
 c->shaders=d.shaders;c->images=d.images;c->pool=d.pool;c->worldToRender=d.worldToRender;c->w2r=d.w2r;c->r2w=d.r2w;
 std::memcpy(c->worldToObject,d.worldToObject,sizeof c->worldToObject);return c;}
inline void subdDisplaceAt(void* user,const void* object,const float* P,const float* N,const float* dPds,const float* dPdt,const float* st,const float* fp,
                           const shade::PrimAttr* prims,const float* primData,unsigned primCount,float* out);
inline void subdDisplace(void* user,const void* object,const float* P,const float* N,const float* dPds,const float* dPdt,const float* st,const float* fp,float* out){
 subdDisplaceAt(user,object,P,N,dPds,dPdt,st,fp,nullptr,nullptr,0,out);}
// The same with the control mesh's primitive attributes at the vertex (xpu2_geom.dll's displaceWith: MoonRay's
// displacement state's, three floats each, the first NaN where the vertex has none): the hit's attributes for the
// displacement's maps (a named UV set's AttributeMap or ModoTextureMap), constant at the vertex and without st
// derivatives, as MoonRay's displacement state gives them (State::subdivision: ModoTextureMap's uv_affine stands for them).
inline void subdDisplaceWith(void* user,const void* object,const float* P,const float* N,const float* dPds,const float* dPdt,const float* st,const float* fp,
                             const X2SubdAttribute* attributes,unsigned attributeCount,const float* values,float* out){
 shade::PrimAttr prims[32];unsigned n=0;
 for(unsigned i=0;i<attributeCount&&n<32;++i){if(std::isnan(values[3*i]))continue;const int type=attributes[i].type;
  const unsigned prim=type==1?shade::PrimVec2f:type==2?shade::PrimVec3f:type==3?shade::PrimRgb:type==4?shade::PrimInt:shade::PrimFloat;
  const unsigned components=type==1?2u:type==2||type==3?3u:1u;
  prims[n++]=shade::PrimAttr{shade::primKey(attributes[i].name,prim),components,shade::PrimConstant,3*i};}
 subdDisplaceAt(user,object,P,N,dPds,dPdt,st,fp,prims,values,n,out);}
inline void subdDisplaceAt(void* user,const void* object,const float* P,const float* N,const float* dPds,const float* dPdt,const float* st,const float* fp,
                           const shade::PrimAttr* prims,const float* primData,unsigned primCount,float* out){
 const long long began=nowNs();struct Done{long long began;~Done(){const long long ended=nowNs();subdTimes().displaceCalls++;subdTimes().displaceNs+=ended-began;subdTimes().seen(began,ended);}}done{began};
 auto& d=*static_cast<SubdDisplacement*>(user);out[0]=out[1]=out[2]=0.0f;
 static std::once_flag traced;std::call_once(traced,[&]{std::fprintf(stderr,"xpu2: subdivision displacement at %g %g %g, %zu shaders\n",P[0],P[1],P[2],d.shaders.size());std::fflush(stderr);});
 const auto it=d.shaders.find(object);if(it==d.shaders.end())return;
 auto point=[&](const float* q,const rdl::Mat4d& m){return V3{float(q[0]*m[0][0]+q[1]*m[1][0]+q[2]*m[2][0]+m[3][0]),float(q[0]*m[0][1]+q[1]*m[1][1]+q[2]*m[2][1]+m[3][1]),
                                                      float(q[0]*m[0][2]+q[1]*m[1][2]+q[2]*m[2][2]+m[3][2])};};
 auto vec=[&](const float* q,const rdl::Mat4d& m){return V3{float(q[0]*m[0][0]+q[1]*m[1][0]+q[2]*m[2][0]),float(q[0]*m[0][1]+q[1]*m[1][1]+q[2]*m[2][1]),
                                                    float(q[0]*m[0][2]+q[1]*m[1][2]+q[2]*m[2][2])};};
 try{
  shade::State s;s.P=point(P,d.r2w);s.N=s.Ng=vec(N,d.r2w);s.dPds=vec(dPds,d.r2w);s.dPdt=vec(dPdt,d.r2w);s.wo=V3{0,0,0};s.displacement=true;
  s.s=st[0];s.t=st[1];s.dsdx=fp[0];s.dsdy=fp[1];s.dtdx=fp[2];s.dtdy=fp[3];s.worldToRender=d.worldToRender;s.worldToObject=d.worldToObject;
  s.prims=prims;s.primData=primData;s.primCount=primCount;s.subdivision=true;
  const displace::Textures textures{d.images.data(),unsigned(d.images.size())};
  const V3 w=displace::displacement(it->second,d.pool,s,textures);const float wf[3]={w.x,w.y,w.z};const V3 r=vec(wf,d.w2r);
  out[0]=r.x;out[1]=r.y;out[2]=r.z;
 }catch(const std::exception& e){std::lock_guard<std::mutex> g(d.lock);if(d.failure.empty())d.failure=e.what();}
}
// UserData primitive attributes (geom/PrimitiveUserData processArbitraryData) at a geometry's rate counts: constant,
// uniform (a face, or a curve), vertex, varying, face varying; pickRate's guess from the count when the rate is auto.
inline void primitiveAttributes(const rdl::SceneObject& o,MeshData& m,size_t uniformCount,size_t vertexCount,size_t varyingCount,size_t faceVaryingCount,
                                std::vector<std::pair<std::string,unsigned>>* named=nullptr){
 for(auto* object:o.get<rdl::SceneObjectVector>("primitive_attributes")){
  need(object&&object->getSceneClass().getName()=="UserData",o.getName()+": a primitive attribute that is not UserData");
  attributes(*object,{"bool_key","bool_values","int_key","int_values","float_key","float_values_0","float_values_1","string_key","string_values","color_key","color_values_0","color_values_1","vec2f_key","vec2f_values_0","vec2f_values_1","vec3f_key","vec3f_values_0","vec3f_values_1","mat3f_key","mat3f_values_0","mat3f_values_1","mat4f_key","mat4f_values_0","mat4f_values_1","rate"});
  const int explicitRate=get<rdl::Int>(*object,"rate");
  auto pick=[&](size_t n)->unsigned{
   switch(explicitRate){case 1:return n==1?1u:0u;case 3:return n==uniformCount?3u:0u;case 4:return n==vertexCount?4u:0u;case 5:return n==varyingCount?5u:0u;case 6:return faceVaryingCount&&n==faceVaryingCount?6u:0u;case 0:break;default:return 0u;}
   if(n==0)return 0u;if(n==1)return 1u;if(faceVaryingCount&&n==faceVaryingCount)return 6u;if(n==uniformCount)return 3u;if(n==vertexCount)return 4u;if(n==varyingCount)return 5u;return 0u;};
  auto add=[&](const std::string& key,unsigned type,unsigned components,const std::vector<float>& values,size_t count){
   if(key.empty())return;const unsigned r=pick(count);need(r!=0,o.getName()+": primitive attribute "+key+" has an unsupported rate or size");
   for(float f:values)X2_NEED(std::isfinite(f),o.getName()+": invalid primitive attribute "+key);
   m.prims.push_back(shade::PrimAttr{shade::primKey(key.c_str(),type),components,r,unsigned(m.primData.size())});m.primData.insert(m.primData.end(),values.begin(),values.end());
   if(named)named->emplace_back(key,type);};
  {const auto& d=get<rdl::FloatVector>(*object,"float_values_0");add(get<rdl::String>(*object,"float_key"),shade::PrimFloat,1,std::vector<float>(d.begin(),d.end()),d.size());}
  {const auto& d=get<rdl::IntVector>(*object,"int_values");std::vector<float> f;for(int i:d)f.push_back(float(i));add(get<rdl::String>(*object,"int_key"),shade::PrimInt,1,f,d.size());}
  {const auto& d=get<rdl::RgbVector>(*object,"color_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.r);f.push_back(c.g);f.push_back(c.b);}add(get<rdl::String>(*object,"color_key"),shade::PrimRgb,3,f,d.size());}
  {const auto& d=get<rdl::Vec2fVector>(*object,"vec2f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);}add(get<rdl::String>(*object,"vec2f_key"),shade::PrimVec2f,2,f,d.size());}
  {const auto& d=get<rdl::Vec3fVector>(*object,"vec3f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);f.push_back(c.z);}add(get<rdl::String>(*object,"vec3f_key"),shade::PrimVec3f,3,f,d.size());}
 }
}
inline bool hasPrim(const MeshData& m,const char* name,unsigned type){const unsigned k=shade::primKey(name,type);for(auto& a:m.prims)if(a.key==k)return true;return false;}
// MoonRay's curves (geom/prim/Curves.h) from RdlCurveGeometry or ModoFurGeometry (the plugin's fur: its fibres in a
// file written by modo_fur_grow, linear curves with a texture coordinate a point). The control points go to render
// space with their radii scaled as Primitive::applyTransformation scales them; the spans are numbered as
// LineSegments, BezierSpanChains and BSpline number them; uv_list becomes the "uv" primitive attribute at its rate.
// instanceMotionSteps (below): a moving instance's steps of its reference.
inline std::vector<std::vector<V3>> instanceMotionSteps(const rdl::SceneObject& o,const MotionParams& mp,const RenderItem& item,
                                                       bool staticTakesSecond,std::vector<float>* scales,
                                                       const std::vector<float>* w0,const std::vector<float>* w1,
                                                       std::vector<std::vector<float>>* widths);
inline void admitCurves(const rdl::SceneObject& o,const rdl::Mat4d& transform,MeshData& m,const MotionParams* motion=nullptr,
                        const RenderItem* item=nullptr){
 // The raw time steps' radii under motion (screen-space radii make each step's its own): widths for meshMotionSteps.
 std::vector<float> rawWidth0,rawWidth1;
 const bool fur=o.getSceneClass().getName()=="ModoFurGeometry";
 std::vector<unsigned> counts;std::vector<float> points;  // object space: x, y, z, radius
 int type=0;
 if(fur){
  {std::set<std::string> allowed{"file","curves_subtype","tessellation_rate","node_xform"};allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(o,allowed);}
  const std::string path=get<rdl::String>(o,"file");std::ifstream source(path,std::ios::binary);
  char magic[4];uint32_t fibres=0;uint64_t total=0;float ends[2];
  need(bool(source.read(magic,4))&&std::memcmp(magic,"MFC1",4)==0&&bool(source.read(reinterpret_cast<char*>(&fibres),4))&&bool(source.read(reinterpret_cast<char*>(&total),8))&&bool(source.read(reinterpret_cast<char*>(ends),8)),o.getName()+": cannot read the fur file");
  need(fibres>0&&total>0&&total<=(uint64_t(1)<<28),o.getName()+": empty or oversized fur");
  counts.resize(fibres);points.resize(size_t(total)*4);std::vector<float> across(size_t(total)*2);
  need(bool(source.read(reinterpret_cast<char*>(counts.data()),counts.size()*4))&&bool(source.read(reinterpret_cast<char*>(points.data()),points.size()*4))&&bool(source.read(reinterpret_cast<char*>(across.data()),across.size()*4)),o.getName()+": the fur file is cut short");
  uint64_t counted=0;for(auto c:counts)counted+=c;need(counted==total,o.getName()+": the fur file does not add up");
  m.prims.push_back(shade::PrimAttr{shade::primKey("uv",shade::PrimVec2f),2,shade::PrimVertex,unsigned(m.primData.size())});m.primData.insert(m.primData.end(),across.begin(),across.end());
  m.prims.push_back(shade::PrimAttr{shade::primKey("modo_fine",shade::PrimFloat),1,shade::PrimConstant,unsigned(m.primData.size())});m.primData.push_back(1.0f);
 }else{
  {std::set<std::string> allowed{"curves_vertex_count","vertex_list_0","radius_list","uv_list","curve_type","curves_subtype","tessellation_rate","node_xform","primitive_attributes",
    "vertex_list_1","velocity_list_0","velocity_list_1","accleration_list","velocity_scale","motion_blur_type","curved_motion_blur_sample_count",
    "use_screen_space_radius"};
   allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(o,allowed,{},{"node_xform"});}
  type=get<rdl::Int>(o,"curve_type");need(type>=0&&type<=2,o.getName()+": unknown curve type");
  const auto& c=o.get<rdl::IntVector>("curves_vertex_count");const auto& v=o.get<rdl::Vec3fVector>("vertex_list_0");
  for(int n:c){X2_NEED(n>0,o.getName()+": invalid curve vertex count");counts.push_back(unsigned(n));}
  // The radii (RdlCurveGeometry::generate): constant, a curve, a vertex, or a span end (varying), else 0.5.
  const size_t vertCount=v.size(),curves=counts.size();
  const size_t varyingCount=type==0?vertCount:type==1?(vertCount-curves)/3+curves:vertCount-2*curves;
  const auto& rv=o.get<rdl::FloatVector>("radius_list");std::vector<float> radius;
  const size_t n=rv.size();
  if(n==0)radius.assign(vertCount,0.5f);
  else if(n==1)radius.assign(vertCount,rv[0]);
  else if(n==curves){for(size_t i=0;i<curves;++i)for(size_t k=counts[i];k--;)radius.push_back(rv[i]);}
  else if(n==vertCount)radius.assign(rv.begin(),rv.end());
  else if(n==varyingCount){
   if(type==0)radius.assign(rv.begin(),rv.end());
   // MoonRay's Bezier expansion writes more radii than points (its loop runs to the curve's last point): refused.
   else if(type==1)throw Unsupported(o.getName()+": Bezier radii a span end");
   else{size_t i=0;for(size_t count:counts){radius.push_back(2*rv[i]-rv[i+1]);for(size_t j=0;j<count-2;j++)radius.push_back(rv[i++]);radius.push_back(2*rv[i-1]-rv[i-2]);}}
  }else throw Unsupported(o.getName()+": radius_list has an unsupported rate");
  need(radius.size()==vertCount,o.getName()+": radii do not match the curves");
  points.reserve(vertCount*4);for(size_t i=0;i<vertCount;++i){points.push_back(v[i].x);points.push_back(v[i].y);points.push_back(v[i].z);points.push_back(radius[i]);}
  primitiveAttributes(o,m,curves,vertCount,varyingCount,0);
  // use_screen_space_radius (applyScreenSpaceRadius over the curves): each point's radius times its distance from the
  // primary camera over the camera's focal, times its min/max distance radius lerped between min_distance and
  // max_distance (one a curve, a point, or one for all; one each by default).
  if(get<rdl::Bool>(o,"use_screen_space_radius")){namespace sm=scene_rdl2::math;
   // Under motion each raw time step (vertex_list_0, vertex_list_1) has radii from its own points.
   const auto& v1=o.get<rdl::Vec3fVector>("vertex_list_1");
   const bool second=motion&&motion->on&&v1.size()==vertCount;
   const rdl::SceneVariables& sv=o.getSceneClass().getSceneContext()->getSceneVariables();
   const rdl::SceneObject* camera=get<rdl::SceneObject*>(sv,"camera");
   bool hasFocal=false;if(camera)for(auto i=camera->getSceneClass().beginAttributes();i!=camera->getSceneClass().endAttributes();++i)if((*i)->getName()=="focal")hasFocal=true;
   if(camera&&hasFocal){
    const float focal=get<rdl::Float>(*camera,"focal");const rdl::Mat4d cm=camera->get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN);
    const sm::Vec3f cameraPosition=sm::Vec3f(float(cm[3][0]),float(cm[3][1]),float(cm[3][2]));
    const sm::Mat4f node=sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN));
    auto values=[&](const char* key){std::vector<float> out(vertCount,1.0f);
     for(const auto& a:m.prims)if(a.key==shade::primKey(key,shade::PrimFloat)){
      if(a.rate==shade::PrimConstant)std::fill(out.begin(),out.end(),m.primData[a.offset]);
      else if(a.rate==shade::PrimUniform){size_t i=0;for(size_t cc=0;cc<curves;++cc)for(unsigned k=0;k<counts[cc];++k)out[i++]=m.primData[a.offset+cc];}
      else if(a.rate==shade::PrimVertex||a.rate==shade::PrimVarying){for(size_t i=0;i<vertCount&&a.offset+i<m.primData.size();++i)out[i]=m.primData[a.offset+i];}}
     return out;};
    const auto minDist=values("min_distance"),maxDist=values("max_distance"),minRadius=values("min_distance_radius"),maxRadius=values("max_distance_radius");
    auto screen=[&](const sm::Vec3f& local,size_t i,float r){
     const sm::Vec3f position=sm::transformPoint(node,local);
     const float distance=sm::distance(position,cameraPosition);
     const float distT=sm::clamp((distance-minDist[i])/(maxDist[i]-minDist[i]),0.0f,1.0f);
     const float distFactor=(minRadius[i]*(1.0f-distT))+(maxRadius[i]*distT);
     return std::max(0.f,r*(distance/focal)*distFactor);};
    if(second){rawWidth1.resize(vertCount);for(size_t i=0;i<vertCount;++i)rawWidth1[i]=screen(sm::Vec3f(v1[i].x,v1[i].y,v1[i].z),i,points[4*i+3]);}
    for(size_t i=0;i<vertCount;++i)points[4*i+3]=screen(sm::Vec3f(points[4*i],points[4*i+1],points[4*i+2]),i,points[4*i+3]);
    if(second){rawWidth0.resize(vertCount);for(size_t i=0;i<vertCount;++i)rawWidth0[i]=points[4*i+3];}}}
  // uv_list, unless the primitive attributes already give "st" or "uv".
  const auto& ul=o.get<rdl::Vec2fVector>("uv_list");
  if(!ul.empty()&&!hasPrim(m,"st",shade::PrimVec2f)&&!hasPrim(m,"uv",shade::PrimVec2f)){
   const size_t un=ul.size();const unsigned r=un==1?1u:un==curves?3u:un==vertCount?4u:un==varyingCount?5u:0u;need(r!=0,o.getName()+": uv_list has an unsupported rate");
   m.prims.push_back(shade::PrimAttr{shade::primKey("uv",shade::PrimVec2f),2,r,unsigned(m.primData.size())});
   for(auto u:ul){X2_NEED(std::isfinite(u.x)&&std::isfinite(u.y),o.getName()+": invalid uv");m.primData.push_back(u.x);m.primData.push_back(u.y);}
  }
 }
 const int subtype=get<rdl::Int>(o,"curves_subtype");need(subtype==0||subtype==1,o.getName()+": normal-oriented curves (MoonRay XPU draws them facing the ray)");
 const int rate=get<rdl::Int>(o,"tessellation_rate");need(rate>=1&&rate<=64,o.getName()+": tessellation rate out of range");
 // Curves::checkPrimitiveData: whole spans, as many points as the curves say.
 size_t vertices=0;for(auto c:counts){vertices+=c;X2_NEED(type==0?c>=2:type==1?(c>=4&&(c-1)%3==0):c>=4,o.getName()+": a curve with an invalid number of points");}
 need(vertices*4==points.size()&&vertices>0,o.getName()+": curve points do not match the counts");
 m.curveBasis=type==0?curves::BasisLinear:type==1?curves::BasisBezier:curves::BasisBspline;m.curveRound=unsigned(subtype);m.curveSegments=unsigned(rate);
 // Render space: the points as the mesh's; the radii scaled by the transform's mean axis length.
 const float radiusScale=(float(std::sqrt(transform[0][0]*transform[0][0]+transform[0][1]*transform[0][1]+transform[0][2]*transform[0][2]))+
  float(std::sqrt(transform[1][0]*transform[1][0]+transform[1][1]*transform[1][1]+transform[1][2]*transform[1][2]))+
  float(std::sqrt(transform[2][0]*transform[2][0]+transform[2][1]*transform[2][1]+transform[2][2]*transform[2][2])))/3.f;
 m.curvePoints.reserve(points.size());
 for(size_t i=0;i<vertices;++i){const double x=points[4*i],y=points[4*i+1],z=points[4*i+2];
  const float p[3]={float(x*transform[0][0]+y*transform[1][0]+z*transform[2][0]+transform[3][0]),float(x*transform[0][1]+y*transform[1][1]+z*transform[2][1]+transform[3][1]),float(x*transform[0][2]+y*transform[1][2]+z*transform[2][2]+transform[3][2])};
  const float r=points[4*i+3]*radiusScale;
  need(std::isfinite(p[0])&&std::isfinite(p[1])&&std::isfinite(p[2])&&std::isfinite(r)&&r>=0&&std::abs(p[0])<=10000&&std::abs(p[1])<=10000&&std::abs(p[2])<=10000,o.getName()+": invalid curve point");
  m.curvePoints.insert(m.curvePoints.end(),{p[0],p[1],p[2],r});}
 // Under motion blur (Curves::transformPrimitive's transformVertexBuffer): the points at each time step as a mesh's
 // vertices (the static case keeps vertex_list_0), the radii scaled by each step's transform.
 // A moving instance: the reference's steps through the instance's transform at each step's time (instanceMotionSteps).
 if(!fur&&motion&&motion->on&&(meshHasMotion(o)||(item&&item->moves))){
  std::vector<float> scales;std::vector<std::vector<float>> widths;
  const std::vector<float>* w0=rawWidth0.empty()?nullptr:&rawWidth0;const std::vector<float>* w1=rawWidth1.empty()?nullptr:&rawWidth1;
  const auto steps=item?instanceMotionSteps(o,*motion,*item,false,&scales,w0,w1,&widths):
                   meshMotionSteps(o,*motion,o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN),o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END),false,&scales,w0,w1,&widths);
  need(steps.size()==scales.size()&&steps[0].size()==vertices&&(widths.empty()||widths.size()==steps.size()),o.getName()+": curve motion steps do not match the points");
  for(size_t k=0;k<steps.size();++k){std::vector<float> step(points.size());
   for(size_t i=0;i<vertices;++i){step[4*i]=steps[k][i].x;step[4*i+1]=steps[k][i].y;step[4*i+2]=steps[k][i].z;
    step[4*i+3]=(widths.empty()?points[4*i+3]:widths[k][i])*scales[k];
    need(std::isfinite(step[4*i+3])&&step[4*i+3]>=0,o.getName()+": invalid curve point");}
   if(k==0)m.curvePoints=std::move(step);else m.curveSteps.push_back(std::move(step));}
  for(const auto& s:m.curveSteps)m.curveStepPointers.push_back(s.data());
 }
 // The index buffer: a span's first point, its curve, its place along the curve.
 unsigned offset=0;
 for(unsigned chain=0;chain<counts.size();++chain){const unsigned c=counts[chain];
  const unsigned spans=type==0?c-1:type==1?(c-1)/3:c-3;m.curveSpanCounts.push_back(spans);
  for(unsigned j=0;j<spans;++j){m.curveSpans.insert(m.curveSpans.end(),{offset,chain,j});offset+=type==1?3u:1u;}
  offset+=type==0?1u:type==1?1u:3u;}
 need(m.curveSpans.size()/3<(1u<<30),o.getName()+": too many curve spans");
}

// Where MoonRay's sampler tables are (beside the GPU program in a build), and the tables a random walk draws from:
// pbr/sampler's bc_1D integrator sequences (4096 of 1024 floats) then its ppd_2D ones (4096 of 1024 pairs), loaded
// once and shared.
inline fs::path& sequenceDirectory(){static fs::path p;return p;}
inline std::shared_ptr<const std::vector<float>> walkSequences(){
 static std::mutex m;static std::weak_ptr<const std::vector<float>> cache;std::lock_guard<std::mutex> lock(m);
 if(auto p=cache.lock())return p;
 auto v=std::make_shared<std::vector<float>>(size_t(4096)*1024*3);
 auto read=[&](const char* name,float* dst,size_t n){std::ifstream f(sequenceDirectory()/name,std::ios::binary);
  need(bool(f),std::string("Missing MoonRay sampler table ")+name);f.read(reinterpret_cast<char*>(dst),std::streamsize(n*sizeof(float)));
  need(size_t(f.gcount())==n*sizeof(float),std::string("Short MoonRay sampler table ")+name);};
 read("bc_1D_integrator_points_4096_sequences_of_1024.bin",v->data(),size_t(4096)*1024);
 read("ppd_2D_integrator_points_4096_sequences_of_1024.bin",v->data()+size_t(4096)*1024,size_t(4096)*1024*2);
 cache=v;return v;
}
// MoonRay's polygon meshes (geom/PolygonMesh.cc). With no face above four vertices, a QuadMesh when its quads are
// fewer primitives than triangles would be: each face a quad, a triangle's as v0 v1 v2 v2 (QuadMesh::splitNGons),
// which Embree and XPU's GPU split as (0, 1, 3) and (2, 3, 1). Otherwise a TriMesh: each face a fan (0, k, k + 1),
// but a planar n-gon above four vertices of some area, which it ear-clips in object space (TriMesh::splitNGons).
// Each triangle keeps its face, its corners' face-vertices (for face-varying values) and its corners' vertices,
// which TriMesh::splitNGons can make differ (see splitPolygons).
struct PolygonSplit {
 bool quads=false;
 std::vector<int> quadVertices;  // a QuadMesh's quads, four vertex indices each (its smooth normals)
 std::vector<unsigned> corners;  // three face-vertex indices a triangle
 std::vector<int> vertices;      // three vertex indices a triangle, as MoonRay renders them
 std::vector<unsigned> faceOf;   // each triangle's face
 std::vector<unsigned char> second;  // a QuadMesh's: the quad's second triangle, (v2 v3 v1) (getQuadST's !isFirst)
};
namespace polygon {
struct F3{float x,y,z;};
inline F3 sub(F3 a,F3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline F3 crs(F3 a,F3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline float dt(F3 a,F3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline F3 nrm(F3 a){const float r=1.0f/std::sqrt(dt(a,a));return {a.x*r,a.y*r,a.z*r};}
inline unsigned loop(int i,int n){return unsigned(i>=0&&i<n?i:i>=n?i%n:i%n+n);}
inline F3 at(const rdl::Vec3fVector& v,const rdl::IntVector& ind,size_t k){const auto& q=v[size_t(ind[k])];return {q.x,q.y,q.z};}
// at for a buffer TriMesh::splitNGons may have rewritten: an index out of range reads the origin (MoonRay reads
// whatever memory holds; no such index arises from in-range indices).
inline F3 at3(const rdl::Vec3fVector& v,const rdl::IntVector& ind,size_t k){const int i=ind[k];if(i<0||size_t(i)>=v.size())return {0,0,0};const auto& q=v[size_t(i)];return {q.x,q.y,q.z};}
// TriMesh.cc's unitNormal, calculateConcaveNGonArea, calculateConcaveNGonNormal, isPlanar and pointInTriangle.
inline F3 unitNormal(F3 a,F3 b,F3 c){
 auto det=[](F3 r0,F3 r1,F3 r2){return dt(r0,crs(r1,r2));};
 const float x=det({1,a.y,a.z},{1,b.y,b.z},{1,c.y,c.z}),y=det({a.x,1,a.z},{b.x,1,b.z},{c.x,1,c.z}),z=det({a.x,a.y,1},{b.x,b.y,1},{c.x,c.y,1});
 const float m=std::max(1e-6f,std::sqrt(x*x+y*y+z*z));return {x/m,y/m,z/m};}
inline float area(const rdl::Vec3fVector& v,const rdl::IntVector& ind,int n,size_t o){
 F3 t{0,0,0};for(int i=0;i<n;++i){const F3 c=crs(at(v,ind,o+loop(i,n)),at(v,ind,o+loop(i+1,n)));t={t.x+c.x,t.y+c.y,t.z+c.z};}
 return dt(t,unitNormal(at(v,ind,o),at(v,ind,o+1),at(v,ind,o+2)))/2.f;}
inline F3 normal(const rdl::Vec3fVector& v,const rdl::IntVector& ind,int n,size_t o){
 const float a=area(v,ind,n,o);F3 t{0,0,0};
 for(int i=0;i<n;++i){const F3 c=crs(at(v,ind,o+size_t(i)),at(v,ind,o+loop(i+1,n)));t={t.x+c.x,t.y+c.y,t.z+c.z};}
 if(a<0.f)t={-t.x,-t.y,-t.z};return nrm(t);}
inline bool planar(const rdl::Vec3fVector& v,const rdl::IntVector& ind,int n,size_t o){
 const F3 a=at(v,ind,o),b=at(v,ind,o+1),c=at(v,ind,o+2),N=crs(sub(a,b),sub(c,b));
 for(int i=3;i<n;++i)if(!(std::abs(dt(nrm(sub(a,at(v,ind,o+size_t(i)))),N))<=1e-6f))return false;
 return true;}
inline bool inside(F3 N,F3 p,F3 a,F3 b,F3 c){
 (void)N;const F3 v0=sub(b,a),v1=sub(c,a),v2=sub(p,a);
 const float d00=dt(v0,v0),d01=dt(v0,v1),d11=dt(v1,v1),d20=dt(v2,v0),d21=dt(v2,v1),den=d00*d11-d01*d01;
 const float y=(d11*d20-d01*d21)/den,w=(d00*d21-d01*d20)/den,u=1.0f-y-w;return u>=0&&y>=0&&w>=0;}
}
inline PolygonSplit splitPolygons(const rdl::Vec3fVector& v,const rdl::IntVector& counts,const rdl::IntVector& ind,const std::string& name){
 using namespace polygon;
 PolygonSplit out;size_t tris=0,quads=0,total=0;int most=0;
 for(int n:counts){X2_NEED(n>=3,name+": a face of fewer than three vertices");tris+=size_t(n-2);quads+=size_t((n-1)>>1);total+=size_t(n);most=std::max(most,n);}
 need(total==ind.size(),name+": face vertex counts do not match the vertex indices");
 for(int k:ind)need(k>=0&&size_t(k)<v.size(),"Vertex index out of bounds");
 auto tri=[&](size_t f,unsigned a,unsigned b,unsigned c){out.corners.insert(out.corners.end(),{a,b,c});out.vertices.insert(out.vertices.end(),{ind[a],ind[b],ind[c]});out.faceOf.push_back(unsigned(f));};
 out.quads=most<=4&&quads*4<tris*3;
 if(out.quads){
  size_t o=0;
  for(size_t f=0;f<counts.size();o+=size_t(counts[f]),++f){
   const int n=counts[f];const unsigned b=unsigned(o);
   const unsigned q[4]={b,b+1,b+2,n==3?b+2:b+3};
   for(unsigned k:q)out.quadVertices.push_back(ind[k]);
   tri(f,q[0],q[1],q[3]);out.second.push_back(0);if(n==4){tri(f,q[2],q[3],q[1]);out.second.push_back(1);}}
  return out;}
 // TriMesh::splitNGons as MoonRay runs it: the faces from the last, each one's triangles written backward from the
 // end of the very index buffer it reads. A fan reads its indices before it can overwrite them, but ear clipping does
 // not: when the faces before an n-gon are mostly triangles, its own triangles overwrite indices it reads next (and a
 // clipping that fails leaves them so for the fan it reverts to). MoonRay renders the indices it read, while its
 // face-varying values follow the corners it meant (indexRemapping): so do these.
 rdl::IntVector buf(ind.begin(),ind.end());buf.resize(3*tris);
 std::vector<unsigned> remap(3*tris);std::vector<unsigned> faceOf(tris);
 size_t in=ind.size();std::ptrdiff_t at=std::ptrdiff_t(3*tris)-1,atFace=std::ptrdiff_t(tris)-1;
 auto put=[&](size_t corner){remap[size_t(at)]=unsigned(corner);buf[size_t(at)]=buf[corner];--at;};
 auto fan=[&](size_t f,size_t n){for(size_t v=n;v>=3;--v){put(in+v-1);put(in+v-2);put(in);faceOf[size_t(atFace--)]=unsigned(f);}};
 for(std::ptrdiff_t fi=std::ptrdiff_t(counts.size())-1;fi>=0;--fi){
  const size_t f=size_t(fi),n=size_t(counts[f]);in-=n;
  const float a=area(v,buf,int(n),in);const bool some=!std::isnan(a)&&a>=1e-6f;
  const bool flat=n>3&&some?planar(v,buf,int(n),in):true;
  if(!some||!flat||n<=4){fan(f,n);continue;}
  const std::vector<unsigned> remapCopy=remap;const std::ptrdiff_t atCopy=at,atFaceCopy=atFace;
  const F3 N=normal(v,buf,int(n),in);
  std::vector<unsigned> local(n);for(size_t i=0;i<n;++i)local[i]=unsigned(i);
  bool ok=false;
  while(local.size()>3){
   ok=false;const int m=int(local.size());
   for(int i=m-1;i>=0;--i){
    const unsigned ia=local[size_t(i)],ib=local[loop(i-1,m)],ic=local[loop(i+1,m)];
    const F3 va=at3(v,buf,in+ia),vb=at3(v,buf,in+ib),vc=at3(v,buf,in+ic);
    if(dt(crs(sub(vc,va),sub(vb,va)),N)<0.f)continue;
    bool interior=false;
    for(int j=0;j<m;++j){const unsigned l=local[size_t(j)];if(l==ia||l==ib||l==ic)continue;if(inside(N,at3(v,buf,in+l),vb,va,vc)){interior=true;break;}}
    if(interior)continue;
    put(in+ic);put(in+ia);put(in+ib);faceOf[size_t(atFace--)]=unsigned(f);
    local.erase(local.begin()+i);ok=true;break;}
   if(!ok){remap=remapCopy;at=atCopy;atFace=atFaceCopy;break;}
  }
  if(ok){put(in+local[2]);put(in+local[1]);put(in+local[0]);faceOf[size_t(atFace--)]=unsigned(f);}
  else fan(f,n);
 }
 for(size_t t=0;t<tris;++t){
  out.corners.insert(out.corners.end(),{remap[3*t],remap[3*t+1],remap[3*t+2]});
  out.vertices.insert(out.vertices.end(),{buf[3*t],buf[3*t+1],buf[3*t+2]});
  out.faceOf.push_back(faceOf[t]);}
 for(int k:out.vertices)X2_NEED(k>=0&&size_t(k)<v.size(),name+": vertex index out of bounds");
 return out;
}
// reverse_normals (PolyMesh::tessellate's reverseOrientation, after the split): each face's corners reversed, a quad
// (q0 q1 q2 q3) as (q3 q2 q1 q0), which Embree splits as (q3 q2 q0) and (q1 q0 q2); a triangle (a b c) as (c b a).
inline void reverseSplit(PolygonSplit& s,const rdl::IntVector& counts,const rdl::IntVector& ind){
 std::vector<unsigned> corners;std::vector<int> vertices;std::vector<unsigned> faceOf;std::vector<unsigned char> second;
 auto tri=[&](unsigned f,unsigned a,unsigned b,unsigned c){corners.insert(corners.end(),{a,b,c});vertices.insert(vertices.end(),{ind[a],ind[b],ind[c]});faceOf.push_back(f);};
 if(s.quads){size_t o=0;
  for(size_t f=0;f<counts.size();o+=size_t(counts[f]),++f){const unsigned b=unsigned(o);const int n=counts[f];
   const unsigned q[4]={b,b+1,b+2,n==3?b+2:b+3};
   if(n==4){tri(unsigned(f),q[3],q[2],q[0]);second.push_back(0);}
   tri(unsigned(f),q[1],q[0],q[2]);second.push_back(1);}}
 else for(size_t t=0;t<s.faceOf.size();++t){
  corners.insert(corners.end(),{s.corners[3*t+2],s.corners[3*t+1],s.corners[3*t]});
  vertices.insert(vertices.end(),{s.vertices[3*t+2],s.vertices[3*t+1],s.vertices[3*t]});faceOf.push_back(s.faceOf[t]);}
 s.corners=std::move(corners);s.vertices=std::move(vertices);s.faceOf=std::move(faceOf);s.second=std::move(second);}
// A mesh's parts (RdlMeshGeometry's part_list): each face takes its part's assignment and a face in no part the
// mesh's own, the "" part (createPerFaceAssignmentId); a face with none is not rendered. Parts whose lists do not add
// up are ignored, as MoonRay warns and ignores them. One piece an assignment, its faces marked (none: every face).
struct Piece{int id;std::string suffix;std::vector<unsigned char> faces;};
// RdlInstancerGeometry (InstanceProceduralLeaf::instanceWithXforms): each instance a reference geometry under the
// instance's transform, expanded into a mesh of its own. The instance's transform samples are MoonRay's: its
// xform_list entry (or positions, orientations and scales), at the shutter's close its xform_list_close entry or the
// entry moved by its velocity, after the reference's own transform where use_reference_xforms asks, then the
// instancer's node transform at each end (Instance::transformPrimitive's appendXform); under motion the instance moves
// as MotionTransform::eval has it, its ends decomposed and slerped over the shutter.
inline void instancerItems(const rdl::SceneObject& o,const rdl::SceneVariables& sv,const MotionParams& mp,std::vector<RenderItem>& items){
 namespace sm=scene_rdl2::math;
 {std::set<std::string> allowed{"method","xform_list","xform_list_close","positions","orientations","scales","velocities","evaluation_frame",
   "ref_indices","disable_indices","use_reference_xforms","use_reference_attributes","references","primitive_attributes","node_xform","instance_level","side_type"};
  allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(o,allowed,{},{"node_xform"});}
 const auto& refs=o.get<rdl::SceneObjectVector>("references");need(!refs.empty(),o.getName()+": an instancer with no references");
 for(auto* r:refs){const std::string rc=r&&r->isA<rdl::Geometry>()?r->getSceneClass().getName():std::string();
  need(rc=="RdlMeshGeometry"||rc=="RdlCurveGeometry"||rc=="RdlPointGeometry",o.getName()+": an instanced "+(rc.empty()?std::string("object"):rc)+" is not rendered yet");}
 const int method=get<rdl::Int>(o,"method");need(method==0||method==2,o.getName()+": unknown instancing method");
 const auto& xforms=o.get<rdl::Mat4dVector>("xform_list");const auto& closes=o.get<rdl::Mat4dVector>("xform_list_close");
 const auto& positions=o.get<rdl::Vec3fVector>("positions");const auto& orientations=o.get<rdl::Vec4fVector>("orientations");const auto& scales=o.get<rdl::Vec3fVector>("scales");
 const auto& velocities=o.get<rdl::Vec3fVector>("velocities");const auto& indices=o.get<rdl::IntVector>("ref_indices");const auto& disabled=o.get<rdl::IntVector>("disable_indices");
 size_t count=method==0?positions.size():xforms.size();
 need(closes.empty()||(method==2&&closes.size()==xforms.size()),o.getName()+": xform_list_close requires the xform list method and as many entries");
 const bool affineMotion=mp.on&&!closes.empty();
 const bool velocity=mp.on&&!velocities.empty()&&velocities.size()==count;
 float dt0=0,dt1=0;
 if(velocity){const auto& steps=get<rdl::FloatVector>(sv,"motion_steps");need(steps.size()>=2,o.getName()+": velocities need two motion steps");
  const float fps=std::max(get<rdl::Float>(sv,"fps"),1.0f),frame=get<rdl::Float>(o,"evaluation_frame");dt0=(steps[0]-frame)/fps;dt1=(steps[1]-frame)/fps;}
 bool applyIndex=false,groupOnly=false;
 if(indices.size()==count)applyIndex=true;else if(!indices.empty()&&count==0){applyIndex=groupOnly=true;count=indices.size();}
 std::set<int> off;for(int i:disabled)if(i<int(count))off.insert(i);
 const bool useRefXforms=get<rdl::Bool>(o,"use_reference_xforms");
 const sm::Xform3f p0=sm::xform<sm::Xform3f>(sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN)));
 const sm::Xform3f p1=sm::xform<sm::Xform3f>(sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END)));
 const bool parentMoves=!(p0==p1);
 for(size_t i=0;i<count;++i){
  if(off.count(int(i)))continue;
  std::vector<sm::Xform3f> x;
  if(groupOnly)x.push_back(sm::Xform3f(sm::Mat3f(sm::one),sm::Vec3f(sm::zero)));
  else if(method==0){
   const sm::Vec3f scale=scales.size()==count?scales[i]:sm::Vec3f(sm::one);
   sm::Quaternion3f orient(sm::one);
   if(orientations.size()==count&&orientations[i].lengthSqr()>0.0f){orient.i=orientations[i][0];orient.j=orientations[i][1];orient.k=orientations[i][2];orient.r=orientations[i][3];orient=sm::normalize(orient);}
   const sm::Mat3f sr(sm::Mat3f::scale(scale)*sm::Mat3f(orient));
   if(velocity){x.emplace_back(sr,positions[i]+velocities[i]*dt0);x.emplace_back(sr,positions[i]+velocities[i]*dt1);}
   else x.emplace_back(sr,positions[i]);
  }else if(affineMotion){x.push_back(sm::xform<sm::Xform3f>(sm::toFloat(xforms[i])));x.push_back(sm::xform<sm::Xform3f>(sm::toFloat(closes[i])));}
  else if(velocity){sm::Xform3f a=sm::xform<sm::Xform3f>(sm::toFloat(xforms[i])),b=a;a.p+=velocities[i]*dt0;b.p+=velocities[i]*dt1;x.push_back(a);x.push_back(b);}
  else x.push_back(sm::xform<sm::Xform3f>(sm::toFloat(xforms[i])));
  int index=applyIndex?indices[size_t(i)]:0;if(index>int(refs.size())-1)index=0;
  const rdl::SceneObject* ref=refs[size_t(index)];
  if(useRefXforms){
   const sm::Xform3f r0=sm::xform<sm::Xform3f>(sm::toFloat(ref->get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN)));
   const sm::Xform3f r1=sm::xform<sm::Xform3f>(sm::toFloat(ref->get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END)));
   for(size_t j=0;j<x.size();++j)x[j]=(j?r1:r0)*x[j];}
  // MotionTransform: its two samples (one: both the same) times the instancer's at each end.
  sm::Xform3f s0=x[0],s1=x.size()>1?x[1]:x[0];
  s0=s0*p0;s1=s1*(parentMoves?p1:p0);
  RenderItem item{ref,&o,int(i)};item.count=count;item.suffix="@"+o.getName()+"#"+std::to_string(i);item.sample[0]=s0;item.sample[1]=s1;
  item.moves=mp.on&&!(s0==s1)&&!(mp.openDelta==0.0f&&mp.closeDelta==0.0f);
  if(!item.moves&&!(s0==s1)){
   // A static MotionTransform of two samples: their slerp at the shutter's open delta.
   sm::XformComponent3f c0,c1;sm::decompose(s0,c0);sm::decompose(s1,c1);if(sm::dot(c0.r,c1.r)<0.0f)c1.r=-c1.r;
   item.sample[0]=item.sample[1]=sm::slerp(c0,c1,mp.openDelta).combined();}
  items.push_back(item);}
}
// The instance's transform at a time in the shutter (MotionTransform::eval).
inline scene_rdl2::math::Xform3f instanceXformAt(const RenderItem& item,const MotionParams& mp,float t){
 namespace sm=scene_rdl2::math;
 if(!item.moves)return item.sample[0];
 sm::XformComponent3f c0,c1;sm::decompose(item.sample[0],c0);sm::decompose(item.sample[1],c1);if(sm::dot(c0.r,c1.r)<0.0f)c1.r=-c1.r;
 return sm::slerp(c0,c1,mp.openDelta+t*(mp.closeDelta-mp.openDelta)).combined();}
inline rdl::Mat4d instanceMatrix(const RenderItem& item){
 const auto& x=item.sample[0];
 return rdl::Mat4d(x.l.vx.x,x.l.vx.y,x.l.vx.z,0,x.l.vy.x,x.l.vy.y,x.l.vy.z,0,x.l.vz.x,x.l.vz.y,x.l.vz.z,0,x.p.x,x.p.y,x.p.z,1);}
// A moving instance's vertices over the shutter: the reference's own time steps in its space (meshMotionSteps with no
// transform), each moved by the instance's transform at its time; a turning instance is baked at 17 steps.
inline std::vector<std::vector<V3>> instanceMotionSteps(const rdl::SceneObject& o,const MotionParams& mp,const RenderItem& item,
                                                       bool staticTakesSecond=true,std::vector<float>* scales=nullptr,
                                                       const std::vector<float>* w0=nullptr,const std::vector<float>* w1=nullptr,
                                                       std::vector<std::vector<float>>* widths=nullptr){
 namespace sm=scene_rdl2::math;
 const rdl::Mat4d one(sm::one);
 std::vector<std::vector<float>> lw;
 const auto local=meshMotionSteps(o,mp,one,one,staticTakesSecond,nullptr,w0,w1,widths?&lw:nullptr);
 sm::XformComponent3f c0,c1;sm::decompose(item.sample[0],c0);sm::decompose(item.sample[1],c1);
 const bool turns=item.moves&&!(c0.r==c1.r||c0.r==-c1.r);
 const size_t n=std::max<size_t>(local.size(),item.moves?(turns?17:2):1);
 std::vector<std::vector<V3>> out(n,std::vector<V3>(local[0].size()));
 for(size_t j=0;j<n;++j){const float u=n>1?float(j)/float(n-1):0.0f;const sm::Xform3f x=instanceXformAt(item,mp,u);
  const float f=local.size()>1?u*float(local.size()-1):0.0f;const size_t k=std::min<size_t>(size_t(f),local.size()>1?local.size()-2:0);const float w=f-float(k);
  for(size_t i=0;i<local[0].size();++i){V3 p=local[k][i];if(local.size()>1){const V3& q=local[k+1][i];p=V3{(1-w)*p.x+w*q.x,(1-w)*p.y+w*q.y,(1-w)*p.z+w*q.z};}
   const sm::Vec3f r=sm::transformPoint(x,sm::Vec3f(p.x,p.y,p.z));out[j][i]=V3{r.x,r.y,r.z};}
  if(scales)scales->push_back((sm::length(x.l.vx)+sm::length(x.l.vy)+sm::length(x.l.vz))/3.f);
  if(widths&&!lw.empty()){std::vector<float> step(lw[0].size());
   for(size_t i=0;i<step.size();++i)step[i]=lw.size()>1?(1-w)*lw[k][i]+w*lw[k+1][i]:lw[0][i];
   widths->push_back(std::move(step));}}
 return out;}
inline void instanceSurface(MeshData& m,const RenderItem& item,const MotionParams& mp){
 namespace sm=scene_rdl2::math;
 sm::XformComponent3f c0,c1;sm::decompose(item.sample[0],c0);sm::decompose(item.sample[1],c1);
 const bool turns=item.moves&&!(c0.r==c1.r||c0.r==-c1.r);
 const size_t n=item.moves?(turns?17:2):1;
 const std::vector<V3> p0=m.positions,n0=m.normals,s0=m.dpds,t0=m.dpdt;
 auto moved=[&](const sm::Xform3f& x,std::vector<V3>& p,std::vector<V3>& nn,std::vector<V3>& ds,std::vector<V3>& dt){
  const double L[3][3]={{x.l.vx.x,x.l.vx.y,x.l.vx.z},{x.l.vy.x,x.l.vy.y,x.l.vy.z},{x.l.vz.x,x.l.vz.y,x.l.vz.z}};
  const double det=L[0][0]*(L[1][1]*L[2][2]-L[1][2]*L[2][1])-L[0][1]*(L[1][0]*L[2][2]-L[1][2]*L[2][0])+L[0][2]*(L[1][0]*L[2][1]-L[1][1]*L[2][0]);
  double inv[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){const int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,dd=(i+2)%3;inv[i][j]=(L[a][cc]*L[b][dd]-L[a][dd]*L[b][cc])/det;}
  p.resize(p0.size());for(size_t i=0;i<p0.size();++i){const sm::Vec3f r=sm::transformPoint(x,sm::Vec3f(p0[i].x,p0[i].y,p0[i].z));p[i]=V3{r.x,r.y,r.z};}
  // Row vectors: a normal times the inverse's transpose.
  nn.resize(n0.size());for(size_t i=0;i<n0.size();++i){const V3 v=n0[i];
   const double r[3]={v.x*inv[0][0]+v.y*inv[0][1]+v.z*inv[0][2],v.x*inv[1][0]+v.y*inv[1][1]+v.z*inv[1][2],v.x*inv[2][0]+v.y*inv[2][1]+v.z*inv[2][2]};
   const double l=std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);nn[i]=l>0?V3{float(r[0]/l),float(r[1]/l),float(r[2]/l)}:v;}
  auto vec=[&](const std::vector<V3>& in,std::vector<V3>& out){out.resize(in.size());
   for(size_t i=0;i<in.size();++i){const sm::Vec3f r=sm::transformVector(x,sm::Vec3f(in[i].x,in[i].y,in[i].z));out[i]=V3{r.x,r.y,r.z};}};
  vec(s0,ds);vec(t0,dt);};
 moved(instanceXformAt(item,mp,0.0f),m.positions,m.normals,m.dpds,m.dpdt);
 m.steps.clear();m.stepNormals.clear();m.stepDpds.clear();m.stepDpdt.clear();
 for(size_t j=1;j<n;++j){std::vector<V3> p,nn,ds,dt;moved(instanceXformAt(item,mp,float(j)/float(n-1)),p,nn,ds,dt);
  m.steps.push_back(std::move(p));if(!n0.empty())m.stepNormals.push_back(std::move(nn));if(!s0.empty())m.stepDpds.push_back(std::move(ds));if(!t0.empty())m.stepDpdt.push_back(std::move(dt));}
 m.stepShading=n>1;m.instanced=true;
 const auto w=instanceMatrix(item).inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
 for(int i=0;i<12;++i)m.worldToObject[i]=float(rows[i]);}
// The instancer's primitive attributes of one instance (its UserData of one value an instance), constant on its mesh:
// into a mesh's own (instanceAttributes), or a shared mesh's instance's (instanceValues).
inline void instanceValues(const rdl::SceneObject& inst,int instance,size_t count,std::vector<shade::PrimAttr>& prims,std::vector<float>& primData);
inline void instanceAttributes(const rdl::SceneObject& inst,int instance,size_t count,MeshData& m){instanceValues(inst,instance,count,m.prims,m.primData);}
// An instance of shared data placed (Mesh::instance): its transform, the inverse's transpose, whether reflected, and its
// worldToObject (the inverse's rows, as a mesh's own).
inline void placeInstance(Mesh& e,const rdl::Mat4d& x){
 for(int r=0;r<4;++r)for(int c=0;c<3;++c)e.objectToWorld[3*r+c]=float(x[r][c]);
 double M[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)M[i][j]=x[i][j];
 const double det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
 need(std::isfinite(det)&&std::abs(det)>1e-12,"Singular instance transform");
 // inv[i][j]: the inverse's; a normal (row vector) goes through its transpose, n' = n inverse^T.
 double inv[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){const int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,dd=(i+2)%3;inv[i][j]=(M[a][cc]*M[b][dd]-M[a][dd]*M[b][cc])/det;}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)e.normalToWorld[3*i+j]=float(inv[j][i]);
 e.instance=det<0?2u:1u;
 const auto w=x.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
 for(int i=0;i<12;++i)e.worldToObject[i]=float(rows[i]);}
// Whether instances share their reference's data (XPU2_SESSION_SHARED_INSTANCES=0: each baked into a mesh of its own).
inline bool sharedInstances(){const char* v=std::getenv("XPU2_SESSION_SHARED_INSTANCES");return !(v&&v[0]=='0');}
inline void instanceValues(const rdl::SceneObject& inst,int instance,size_t count,std::vector<shade::PrimAttr>& prims,std::vector<float>& primData){
 for(auto* object:inst.get<rdl::SceneObjectVector>("primitive_attributes")){
  if(!object||object->getSceneClass().getName()!="UserData")continue;
  auto add=[&](const std::string& key,unsigned type,const std::vector<float>& values,unsigned components){
   if(key.empty()||values.size()!=count*components)return;
   prims.push_back(shade::PrimAttr{shade::primKey(key.c_str(),type),components,1u,unsigned(primData.size())});
   primData.insert(primData.end(),values.begin()+std::ptrdiff_t(size_t(instance)*components),values.begin()+std::ptrdiff_t((size_t(instance)+1)*components));};
  {const auto& d=get<rdl::FloatVector>(*object,"float_values_0");add(get<rdl::String>(*object,"float_key"),shade::PrimFloat,std::vector<float>(d.begin(),d.end()),1);}
  {const auto& d=get<rdl::IntVector>(*object,"int_values");std::vector<float> f;for(int v:d)f.push_back(float(v));add(get<rdl::String>(*object,"int_key"),shade::PrimInt,f,1);}
  {const auto& d=get<rdl::RgbVector>(*object,"color_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.r);f.push_back(c.g);f.push_back(c.b);}add(get<rdl::String>(*object,"color_key"),shade::PrimRgb,f,3);}
  {const auto& d=get<rdl::Vec2fVector>(*object,"vec2f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);}add(get<rdl::String>(*object,"vec2f_key"),shade::PrimVec2f,f,2);}
  {const auto& d=get<rdl::Vec3fVector>(*object,"vec3f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);f.push_back(c.z);}add(get<rdl::String>(*object,"vec3f_key"),shade::PrimVec3f,f,3);}
 }}
// RdlPointGeometry (RdlPointProcedural, geom::Points): its points' centres, world space, over the shutter as a mesh's
// vertices (meshMotionSteps, whose static case keeps vertex_list_0 for points), and their radii (radius_list: none 0.5,
// one for all, else one a point; a node transform does not scale them).
inline void admitPoints(const rdl::SceneObject& o,const MotionParams& mp,MeshData& m,const std::vector<unsigned char>* mask=nullptr,
                        const rdl::Mat4d* instance=nullptr,const RenderItem* item=nullptr){
 namespace sm=scene_rdl2::math;
 {std::set<std::string> allowed{"vertex_list_0","vertex_list_1","velocity_list_0","velocity_list_1","accleration_list","radius_list","velocity_scale",
   "motion_blur_type","curved_motion_blur_sample_count","node_xform","primitive_attributes","side_type","part_list","part_indices","use_screen_space_radius"};
  allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(o,allowed,{},{"node_xform"});}
 const auto& v=o.get<rdl::Vec3fVector>("vertex_list_0");need(!v.empty(),o.getName()+": no points");
 const auto& r=o.get<rdl::FloatVector>("radius_list");
 // pickRate over RateCounts{parts, 0, 0, points, 0}: one value, one a part (part_indices), one a point; else 0.5.
 const auto& parts=get<rdl::StringVector>(o,"part_list");const auto& partIndices=get<rdl::IntVector>(o,"part_indices");
 m.points=unsigned(v.size());m.pointRadius.resize(v.size());
 for(size_t i=0;i<v.size();++i){
  float x=0.5f;
  if(r.size()==1)x=r[0];
  else if(!parts.empty()&&r.size()==parts.size()){need(i<partIndices.size()&&partIndices[i]>=0&&size_t(partIndices[i])<r.size(),o.getName()+": a point's part out of range");x=r[size_t(partIndices[i])];}
  else if(r.size()==v.size())x=r[i];
  need(std::isfinite(x)&&x>=0,o.getName()+": invalid radius");m.pointRadius[i]=x;}
 // Points keep vertex_list_0 in their static case (not vertex_list_1, as a mesh's does).
 // An instance (a shared primitive): its points through the instance's transform, its radii by the transform's mean
 // axis length (the instance's ray is moved into the points' own space).
 // A moving instance (or a reference with motion of its own): the reference's steps through the instance's transform
 // at each step's time; one radius list, so the instance keeps its size over the shutter.
 std::vector<float> instanceScales;
 auto steps=item&&(item->moves||meshHasMotion(o))?instanceMotionSteps(o,mp,*item,false,&instanceScales):
            instance?meshMotionSteps(o,mp,*instance,*instance,false):
                     meshMotionSteps(o,mp,o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN),o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END),false);
 for(float s:instanceScales)X2_NEED(std::abs(s-instanceScales[0])<=1e-6f*std::max(1.0f,instanceScales[0]),o.getName()+": an instance of points scaling over the shutter is not rendered yet");
 m.positions=std::move(steps[0]);for(size_t k=1;k<steps.size();++k)m.steps.push_back(std::move(steps[k]));
 if(instance){const rdl::Mat4d& x=*instance;const float s=(float(std::sqrt(x[0][0]*x[0][0]+x[0][1]*x[0][1]+x[0][2]*x[0][2]))+
   float(std::sqrt(x[1][0]*x[1][0]+x[1][1]*x[1][1]+x[1][2]*x[1][2]))+float(std::sqrt(x[2][0]*x[2][0]+x[2][1]*x[2][1]+x[2][2]*x[2][2])))/3.f;
  for(auto& r:m.pointRadius)r*=s;}
 const auto w=(instance?*instance:o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN)).inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
 for(int i=0;i<12;++i)m.worldToObject[i]=float(rows[i]);
 // Primitive attributes (UserData): constant, or one a point (vertex or varying; RateCounts has no uniform or face
 // varying for points).
 primitiveAttributes(o,m,0,v.size(),v.size(),0);
 // use_screen_space_radius (applyScreenSpaceRadius): each radius times its distance from the primary camera over the
 // camera's focal, times its min/max distance radius lerped between min_distance and max_distance (each one by default).
 if(get<rdl::Bool>(o,"use_screen_space_radius")){
  const rdl::SceneVariables& sv=o.getSceneClass().getSceneContext()->getSceneVariables();
  const rdl::SceneObject* camera=get<rdl::SceneObject*>(sv,"camera");
  bool hasFocal=false;if(camera)for(auto i=camera->getSceneClass().beginAttributes();i!=camera->getSceneClass().endAttributes();++i)if((*i)->getName()=="focal")hasFocal=true;
  if(camera&&hasFocal){
   const float focal=get<rdl::Float>(*camera,"focal");const rdl::Mat4d cm=camera->get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN);
   const sm::Vec3f cameraPosition=sm::Vec3f(float(cm[3][0]),float(cm[3][1]),float(cm[3][2]));
   const sm::Mat4f node=sm::toFloat(o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN));
   auto values=[&](const char* key){std::vector<float> out(v.size(),1.0f);
    for(const auto& a:m.prims)if(a.key==shade::primKey(key,shade::PrimFloat)){
     if(a.rate==shade::PrimConstant)std::fill(out.begin(),out.end(),m.primData[a.offset]);
     else for(size_t i=0;i<v.size();++i)out[i]=m.primData[a.offset+i];}
    return out;};
   const auto minDist=values("min_distance"),maxDist=values("max_distance"),minRadius=values("min_distance_radius"),maxRadius=values("max_distance_radius");
   for(size_t i=0;i<v.size();++i){
    const sm::Vec3f position=sm::transformPoint(node,sm::Vec3f(v[i].x,v[i].y,v[i].z));
    const float distance=sm::distance(position,cameraPosition);
    const float distT=sm::clamp((distance-minDist[i])/(maxDist[i]-minDist[i]),0.0f,1.0f);
    const float distFactor=(minRadius[i]*(1.0f-distT))+(maxRadius[i]*distT);
    m.pointRadius[i]=std::max(0.f,m.pointRadius[i]*(distance/focal)*distFactor);}}}
 // A part's points alone (createPerPointAssignmentId's points of its assignment).
 if(mask){std::vector<size_t> keep;for(size_t i=0;i<v.size();++i)if(i<mask->size()&&(*mask)[i])keep.push_back(i);
  need(!keep.empty(),o.getName()+": a part of no points");
  auto pick=[&](std::vector<V3>& a){std::vector<V3> b;b.reserve(keep.size());for(size_t i:keep)b.push_back(a[i]);a=std::move(b);};
  pick(m.positions);for(auto& s:m.steps)pick(s);
  {std::vector<float> rr;for(size_t i:keep)rr.push_back(m.pointRadius[i]);m.pointRadius=std::move(rr);}
  std::vector<float> data;
  for(auto& a:m.prims){const unsigned at=unsigned(data.size());
   if(a.rate==shade::PrimConstant)data.insert(data.end(),m.primData.begin()+std::ptrdiff_t(a.offset),m.primData.begin()+std::ptrdiff_t(a.offset+a.components));
   else for(size_t i:keep)data.insert(data.end(),m.primData.begin()+std::ptrdiff_t(a.offset+i*a.components),m.primData.begin()+std::ptrdiff_t(a.offset+(i+1)*a.components));
   a.offset=at;}
  m.primData=std::move(data);m.points=unsigned(keep.size());}
}
inline std::vector<Piece> meshPieces(const rdl::SceneObject& o,const rdl::Geometry& geom,const rdl::Layer& layer){
 const int whole=layer.getAssignmentId(&geom,"");
 // Points (createPerPointAssignmentId): each point the assignment of its part_indices' part, the whole's otherwise.
 if(o.getSceneClass().getName()=="RdlPointGeometry"&&!get<rdl::StringVector>(o,"part_list").empty()){
  const auto& parts=get<rdl::StringVector>(o,"part_list");const auto& partIndices=get<rdl::IntVector>(o,"part_indices");
  const size_t points=o.get<rdl::Vec3fVector>("vertex_list_0").size();
  std::vector<int> id(points,whole);
  for(size_t i=0;i<partIndices.size()&&i<points;++i){const int p=partIndices[i];if(p>=0&&size_t(p)<parts.size())id[i]=layer.getAssignmentId(&geom,parts[size_t(p)]);}
  std::vector<Piece> out;std::map<int,size_t> slot;
  for(size_t i=0;i<points;++i){const int a=id[i];if(a<0)continue;auto it=slot.find(a);
   if(it==slot.end()){it=slot.emplace(a,out.size()).first;out.push_back({a,"#"+std::to_string(a),std::vector<unsigned char>(points,0)});}
   out[it->second].faces[i]=1;}
  if(out.size()==1&&out[0].id==whole&&std::find(id.begin(),id.end(),-1)==id.end())out[0]={whole,"",{}};
  return out;}
 if(o.getSceneClass().getName()!="RdlMeshGeometry"||get<rdl::StringVector>(o,"part_list").empty())return {{whole,"",{}}};
 const auto& parts=get<rdl::StringVector>(o,"part_list");const auto& counts=get<rdl::IntVector>(o,"part_face_count_list");
 const auto& indices=get<rdl::IntVector>(o,"part_face_indices");const size_t faces=get<rdl::IntVector>(o,"face_vertex_count").size();
 size_t expected=0;for(int c:counts)expected+=size_t(std::max(c,0));
 if(parts.size()!=counts.size()||expected!=indices.size())return {{whole,"",{}}};
 std::vector<int> faceId(faces,whole);size_t at=0;
 for(size_t i=0;i<parts.size();++i){const int id=layer.getAssignmentId(&geom,parts[i]);
  for(int k=0;k<counts[i];++k,++at){const int f=indices[at];X2_NEED(f>=0&&size_t(f)<faces,o.getName()+": part face index out of bounds");faceId[size_t(f)]=id;}}
 std::vector<Piece> out;std::map<int,size_t> slot;
 for(size_t f=0;f<faces;++f){const int a=faceId[f];if(a<0)continue;auto it=slot.find(a);
  if(it==slot.end()){it=slot.emplace(a,out.size()).first;out.push_back({a,"#"+std::to_string(a),std::vector<unsigned char>(faces,0)});}
  out[it->second].faces[f]=1;}
 // Every face the mesh's own: the mesh as it is.
 if(out.size()==1&&out[0].id==whole&&std::find(faceId.begin(),faceId.end(),-1)==faceId.end())out[0]={whole,"",{}};
 return out;
}
// MoonRay's MeshLight (light_mesh_build.h, MeshLight.cc): the faces of its geometry, which must stand outside the
// layer (Scene::generateMeshLightGeometry; one inside it leaves the light without faces, off), of its listed parts
// (RenderContext's MeshLightLayer assigns those alone; removeUnassignedFaces drops the rest) or all of it. Geometry
// outside the layer is never tessellated (PolyMesh::tessellate, which splits its polygons, runs for the layer's
// alone), so the light reads the mesh's polygon index buffer as it is, in steps of four for a QuadMesh (no face above
// four vertices and fewer quads than triangles) or three for a TriMesh: a face of another size shifts every face
// after it, and MoonRay emits from the faces those steps make. The light's local space is the geometry's world space; its
// transform (node_xform, no turn) carries it on. A map shader is not read yet; subdivision falls back.
inline bool admitMeshLight(const rdl::SceneObject& o,float sceneScale,const rdl::Layer& layer,const rdl::SceneVariables& sv,const CanonicalView& frame,
                           lights::Light& out,std::vector<MeshLightEntry>& tables,const std::vector<MeshLightEntry>* previous,
                           const Plan* plan=nullptr){
 std::set<std::string> allowed=lightCommonAttributes();allowed.insert({"normalized","apply_scene_scale","geometry","parts","map_shader"});
 allowed.insert(clearRadiusAttributes().begin(),clearRadiusAttributes().end());
 attributes(o,allowed,{},{"node_xform"});
 if(!get<rdl::Bool>(o,"on"))return false;
 const rdl::SceneObject* g=get<rdl::SceneObject*>(o,"geometry");
 if(!g||!g->isA<rdl::Geometry>()||layer.contains(g->asA<rdl::Geometry>()))return false;
 need(!get<rdl::SceneObject*>(o,"map_shader"),o.getName()+": a mesh light's map shader is not rendered yet");
 need(g->getSceneClass().getName()=="RdlMeshGeometry",o.getName()+": a mesh light's geometry must be an RdlMeshGeometry");
 attributes(*g,{"vertex_list_0","face_vertex_count","vertices_by_index","is_subd","node_xform","smooth_normal","normal_list","uv_list","primitive_attributes","mesh_resolution","adaptive_error",
  "part_list","part_face_count_list","part_face_indices","visible_in_camera","visible_shadow","visible_diffuse_reflection","visible_diffuse_transmission","visible_glossy_reflection",
  "visible_glossy_transmission","visible_mirror_reflection","visible_mirror_transmission","label"});
 const bool subd=get<rdl::Bool>(*g,"is_subd");
 need(!subd||plan,o.getName()+": a subdivided mesh light falls back");
 lights::Light l{};
 auto c=color(get<rdl::Rgb>(o,"color"));const float intensity=get<rdl::Float>(o,"intensity"),exposure=get<rdl::Float>(o,"exposure");
 need(surface::finite(intensity)&&intensity>=0&&surface::finite(exposure)&&std::abs(exposure)<=32,o.getName()+": invalid light intensity");
 V3 radiance=surface::mul(c,intensity*std::pow(2.f,exposure));
 if(radiance.x==0&&radiance.y==0&&radiance.z==0)return false;  // a black light is off
 l.visibleInCamera=lightVisibleInCamera(o,sv);l.fallback={1,1,1};l.filter=0;l.image=nullptr;l.visibilityMask=lightVisibilityMask(o);l.presenceShadows=lightPresenceShadows(o,sv);
 lightShadowDistances(o,l,true);
 // Its transform: node_xform as it is (LocalParamLight with no turn), a uniform scale, at the shutter's open.
 auto m=xformAt(o,0.0f);
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)need(std::isfinite(m[i][j])&&std::abs(m[i][j])<=10000,o.getName()+": light transform bound");
 need(std::abs(m[0][3])<=1e-6&&std::abs(m[1][3])<=1e-6&&std::abs(m[2][3])<=1e-6&&std::abs(m[3][3]-1)<=1e-6,o.getName()+": projective light transform unsupported");
 double rows[4][3];for(int i=0;i<4;++i)for(int j=0;j<3;++j)rows[i][j]=m[i][j];
 auto length=[&](int i){return std::sqrt(rows[i][0]*rows[i][0]+rows[i][1]*rows[i][1]+rows[i][2]*rows[i][2]);};
 const double x=length(0),y=length(1),z=length(2);
 need(std::abs(x-y)<=0.0001&&std::abs(x-z)<=0.0001,o.getName()+": non-uniform light scale unsupported");
 const double scale=(x+y+z)/3;need(scale>1e-12,o.getName()+": degenerate light transform");
 const double det=rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])-rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])+rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]);
 need(std::abs(std::abs(det/(scale*scale*scale))-1)<=0.01,o.getName()+": invalid light transform");
 double inv[3][3];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,d=(i+2)%3;inv[i][j]=(rows[a][cc]*rows[b][d]-rows[a][d]*rows[b][cc])/det;}
 for(int i=0;i<3;++i){l.local2Render[i]={float(rows[i][0]),float(rows[i][1]),float(rows[i][2])};
  l.render2Local[i]={float(inv[i][0]),float(inv[i][1]),float(inv[i][2])};
  l.render2LocalRot[i]={float(inv[i][0]*scale),float(inv[i][1]*scale),float(inv[i][2]*scale)};}
 l.position={float(rows[3][0]),float(rows[3][1]),float(rows[3][2])};
 l.direction={float(rows[2][0]/scale),float(rows[2][1]/scale),float(rows[2][2]/scale)};l.scale=float(scale);
 // Its faces: built anew only when the geometry or the parts change.
 const auto& parts=get<rdl::StringVector>(o,"parts");
 std::string key=g->getName();for(const auto& p:parts)key+="|"+p;
 auto sig=signature(*g,frame);std::shared_ptr<lights::MeshLightTables> data;
 const std::vector<MeshLightEntry>* sets[2]={&tables,previous};
 for(const auto* set:sets)if(set&&!data)for(const auto& e:*set)if(e.key==key&&e.signature==sig){data=e.tables;break;}
 if(!data){
  const auto& v=g->get<rdl::Vec3fVector>("vertex_list");const auto& counts=g->get<rdl::IntVector>("face_vertex_count");const auto& indices=g->get<rdl::IntVector>("vertices_by_index");
  if(v.empty()||counts.empty())return false;
  const auto transform=affine(*g);
  // The faces its parts keep (below), for a subdivision surface each tessellated quad by its control face.
  auto keptFaces=[&](){
   std::vector<unsigned char> keep(counts.size(),parts.empty()?1:0);
   if(!parts.empty()){
    const auto& names=get<rdl::StringVector>(*g,"part_list");const auto& partCounts=get<rdl::IntVector>(*g,"part_face_count_list");const auto& partFaces=get<rdl::IntVector>(*g,"part_face_indices");
    size_t expected=0;for(int k:partCounts)expected+=size_t(std::max(k,0));
    if(names.size()==partCounts.size()&&expected==partFaces.size()){size_t at=0;
     for(size_t i=0;i<names.size();++i){const bool listed=std::find(parts.begin(),parts.end(),names[i])!=parts.end();
      for(int k=0;k<partCounts[i];++k,++at){const int f=partFaces[at];X2_NEED(f>=0&&size_t(f)<counts.size(),g->getName()+": part face index out of bounds");if(listed)keep[size_t(f)]=1;}}}}
   return keep;};
  if(subd){
   // MoonRay's OpenSubdiv tessellation of the geometry (its quads, world space), as MeshLight::setMesh takes them.
   const auto s=tessellateSubdivision(*g,layer,*plan,transform);
   const auto keep=keptFaces();
   std::vector<unsigned> corners,prims;
   for(size_t q=0;q*4<s->quads.size();++q){const int f=q<s->controlFace.size()?s->controlFace[q]:-1;
    if(f<0||size_t(f)>=keep.size()||!keep[size_t(f)])continue;
    prims.push_back(unsigned(q));for(int k=0;k<4;++k)corners.push_back(s->quads[4*q+size_t(k)]);}
   for(const auto& p:s->positions)X2_NEED(surface::finite(p),o.getName()+": invalid mesh light vertex");
   data=lights::buildMeshLight(s->positions,corners,4u,prims);
   if(!data)return false;
  }else{
  std::vector<V3> world;world.reserve(v.size());
  for(auto p:v){V3 q{float(p.x*transform[0][0]+p.y*transform[1][0]+p.z*transform[2][0]+transform[3][0]),float(p.x*transform[0][1]+p.y*transform[1][1]+p.z*transform[2][1]+transform[3][1]),
    float(p.x*transform[0][2]+p.y*transform[1][2]+p.z*transform[2][2]+transform[3][2])};need(surface::finite(q),o.getName()+": invalid mesh light vertex");world.push_back(q);}
  // The faces its parts keep: all of them with no parts listed; else those of the listed parts (a face in none, or in
  // a mesh whose part lists do not add up, falls to the unassigned "" part and is culled).
  std::vector<unsigned char> keep(counts.size(),parts.empty()?1:0);
  if(!parts.empty()){
   const auto& names=get<rdl::StringVector>(*g,"part_list");const auto& partCounts=get<rdl::IntVector>(*g,"part_face_count_list");const auto& partFaces=get<rdl::IntVector>(*g,"part_face_indices");
   size_t expected=0;for(int k:partCounts)expected+=size_t(std::max(k,0));
   if(names.size()==partCounts.size()&&expected==partFaces.size()){size_t at=0;
    for(size_t i=0;i<names.size();++i){const bool listed=std::find(parts.begin(),parts.end(),names[i])!=parts.end();
     for(int k=0;k<partCounts[i];++k,++at){const int f=partFaces[at];X2_NEED(f>=0&&size_t(f)<counts.size(),g->getName()+": part face index out of bounds");if(listed)keep[size_t(f)]=1;}}}
  }
  std::vector<unsigned> raw,corners,prims;size_t tris=0,quads=0;int most=0;size_t o=0;
  for(size_t f=0;f<counts.size();o+=size_t(counts[f]),++f){if(!keep[f])continue;const int n=counts[f];X2_NEED(n>=3,g->getName()+": a face of fewer than three vertices");
   tris+=size_t(n-2);quads+=size_t((n-1)>>1);most=std::max(most,n);
   for(int k=0;k<n;++k){const int i=indices[o+size_t(k)];X2_NEED(i>=0&&size_t(i)<v.size(),g->getName()+": vertex index out of bounds");raw.push_back(unsigned(i));}}
  const unsigned stride=most<=4&&quads*4<tris*3?4u:3u;
  for(size_t f=0;(f+1)*stride<=raw.size();++f){prims.push_back(unsigned(f));for(unsigned j=0;j<stride;++j)corners.push_back(raw[f*stride+j]);}
  data=lights::buildMeshLight(world,corners,stride,prims);
  if(!data)return false;  // no faces or no area: MoonRay warns and turns the light off
  }
 }
 if(std::none_of(tables.begin(),tables.end(),[&](const MeshLightEntry& e){return e.tables==data;}))tables.push_back({key,std::move(sig),data});
 // computeLightRadiance over the faces' area, local units.
 const double area=data->area;
 if(get<rdl::Bool>(o,"normalized")){double inverse=1/area;if(get<rdl::Bool>(o,"apply_scene_scale"))inverse*=1/(double(sceneScale)*sceneScale)/3.14159265358979323846;radiance=surface::mul(radiance,float(inverse));}
 if(radiance.x==0&&radiance.y==0&&radiance.z==0)return false;
 l.type=lights::Mesh;l.mesh=&data->view;l.invArea=float(1/area);l.radiance=radiance;
 out=l;return surface::finite(radiance);
}
// The plan of a scene (scene admission). With XPU2_SPLIT_UNITS the session's unit sees the declarations alone and
// xpu2_scene.cc (XPU2_SCENE_UNIT) the definitions, so the admission is compiled once, in its own unit.
#if defined(XPU2_SPLIT_UNITS)
#define X2_SPLIT_INLINE
#else
#define X2_SPLIT_INLINE inline
#endif
// A camera-only edit (the session's fast path, realtime): the camera, render space and render origin made again from the
// scene's camera as it stands, the rest of the plan as it was prepared (its meshes are in world space). False where
// something else follows the camera and the scene must be prepared again: a bake camera, a pixel sample map, a shutter
// that changed (geometry in motion is read over it), volumes and light filters (evaluated at the render origin),
// adaptive tessellation (cameraDependent).
inline void admitCamera(const rdl::SceneObject& camera,const rdl::SceneVariables& sv,Plan& plan,const Plan* previous);
// why, where given: what kept the camera from being taken alone (the session's log).
inline bool updateCamera(Plan& plan,const rdl::SceneContext& context,std::string* why=nullptr){
 const auto& sv=context.getSceneVariables();
 auto* camera=get<rdl::SceneObject*>(sv,"camera");
 auto refuse=[&](const char* reason){if(why)*why=reason;return false;};
 if(!camera||!plan.paths)return refuse("no camera, or not the path tracer");
 if(plan.camera.type==surface::CameraBake)return refuse("a bake camera");
 if(plan.cameraDependent)return refuse("geometry tessellated for the camera (adaptive tessellation)");
 if(!plan.filterLists.empty())return refuse("light filters");
 if(!plan.pixelSamples.empty()||!get<rdl::String>(*camera,"pixel_sample_map").empty())return refuse("a pixel sample map");
 if(!(motionParams(sv,camera)==plan.motion))return refuse("the camera's motion blur changed");
 if(plan.motion.on&&get<rdl::Float>(*camera,"mb_shutter_bias")!=plan.shutterBias)return refuse("the shutter bias changed");
 const auto cm=rigid(*camera);const auto w=cm.inverse();
 {const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)plan.worldToRender[i]=float(rows[i]);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)plan.world2render[4*i+j]=w[i][j];}
 admitCamera(*camera,sv,plan,nullptr);
 plan.renderOrigin=V3{float(cm[3][0]),float(cm[3][1]),float(cm[3][2])};
 // A volume whose settings are mapped (a texture, a map) holds them as at the render origin, the camera's place
 // (originAttenuation, mappedAt): made again where the camera now is (the card's copy goes again, GpuLane::updateCamera).
 // An unmapped one's are its constants. A car's mapped glass had every move of the camera prepare the whole scene.
 for(auto& s:plan.volumes)if(s.mapped){
  const volume::Mapped m=mappedAt(plan,s,plan.renderOrigin);
  s.originAttenuation=(s.flags&volume::MatchAlbedo)?m.albedoColor:m.attenuationColor;s.originIntensity=m.attenuationIntensity;s.originFactor=m.attenuationFactor;}
 return true;
}
// A placement-only edit (the session's fast path, realtime): geometries moved and nothing else of them changed. Each of
// their meshes is made an instance of the data prepared for it at the last full preparation (the plan given, that
// preparation's), placed from where its geometry stood then to where it stands now: its data in world space, as every
// mesh's in a plan (a tessellation's is taken back from render space), the instance's transform the old node_xform's
// inverse then the new one (row vectors); its worldToObject the new node_xform's inverse as a full preparation makes it.
// meshW2R: the preparation's world to render, not used for the meshes (a render space turned or moved with the camera
// put every moved mesh elsewhere). moves: each moved geometry's node_xform then and now. The meshes and their structures on the
// card are taken over as they are; only where they are placed is new. false, with why, where a full preparation is
// needed: a moving or volume-bounding mesh, points or curves, an instance already, one left to the CPU lane.
// instancePlaces: a moved instancer's instances' transforms now, by instance (as prepare makes them, InstanceGeometry's
// xform_list entry by its reference's and its own node_xform), which its shared instances are placed by.
// A BaseVolume's baked extinction (bakeVolume's grid, in its mesh's own space) made again where its volume now is: each
// voxel's object point (object to index's inverse) where the mesh puts it (the volume's world to grid's inverse), its
// extinction there as bakeVolume makes it, into a grid of its own (the plan's grid left as it is). cancel: stops it
// early (a newer move), the grid then none. The grid and its view, or none where it cannot be made.
inline std::pair<std::shared_ptr<VdbGridData>,volume::Grid> rebakeVolume(const Plan& plan,unsigned volumeIndex,const std::atomic<bool>* cancel){
 const volume::Shader& s=plan.volumes.at(volumeIndex);
 const unsigned k=s.densityGrid;
 if(k==volume::NoGrid||k>=plan.volumeGrids.size())return {};
 auto fromRows=[](const float* r){rdl::Mat4d m(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);for(int i=0;i<3;++i){for(int c=0;c<3;++c)m[c][i]=r[4*i+c];m[3][i]=r[4*i+3];}return m;};
 const volume::Grid was0=plan.volumeGridViews[k];
 rdl::Mat4d objectToIndex;for(int i=0;i<4;++i)for(int j=0;j<4;++j)objectToIndex[i][j]=was0.worldToIndex[4*i+j];
 const rdl::Mat4d indexToWorld=objectToIndex.inverse()*fromRows(s.worldToGrid).inverse();
 const int dims[3]={was0.dims[0],was0.dims[1],was0.dims[2]};
 auto g=std::make_shared<VdbGridData>();g->key=plan.volumeGrids[k]->key;g->dense.assign(size_t(dims[0])*size_t(dims[1])*size_t(dims[2])*3,0.0f);
 const unsigned threads=std::max(1u,std::min(16u,std::thread::hardware_concurrency()));
 std::vector<std::thread> pool;std::atomic<int> nextSlice{0};std::atomic<bool> failed{false};
 for(unsigned t=0;t<threads;++t)pool.emplace_back([&]{
  try{for(int kk=nextSlice++;kk<dims[2]&&!(cancel&&cancel->load());kk=nextSlice++)for(int j=0;j<dims[1];++j)for(int i=0;i<dims[0];++i){
   const V3 pw{float(i*indexToWorld[0][0]+j*indexToWorld[1][0]+kk*indexToWorld[2][0]+indexToWorld[3][0]),
               float(i*indexToWorld[0][1]+j*indexToWorld[1][1]+kk*indexToWorld[2][1]+indexToWorld[3][1]),
               float(i*indexToWorld[0][2]+j*indexToWorld[1][2]+kk*indexToWorld[2][2]+indexToWorld[3][2])};
   const volume::Mapped m=mappedAt(plan,s,pw);
   const V3 color=volume::attenuationColorFrom(s,(s.flags&volume::MatchAlbedo)?m.albedoColor:m.attenuationColor,-1.0f);
   const V3 e=volume::cmax0(volume::scaled(volume::times(volume::scaled(V3{1,1,1},m.attenuationIntensity),color),m.attenuationFactor));
   float* out=g->dense.data()+3*((size_t(kk)*size_t(dims[1])+size_t(j))*size_t(dims[0])+size_t(i));out[0]=e.x;out[1]=e.y;out[2]=e.z;}}
  catch(...){failed=true;}});
 for(auto& t:pool)t.join();
 if(failed||(cancel&&cancel->load()))return {};
 volume::Grid view=was0;view.dense=g->dense.data();
 return {std::move(g),view};}
// A moved geometry's volume (the one a mesh of it bounds), placed with it (placeMoved): its grids' world to grid and grid
// to world, its emissive regions' maps, and what its maps give where they are read (world points): a mapped volume's
// settings at the render origin made again, and a BaseVolume's baked extinction (bakeVolume's grid, in the mesh's own
// space) made again at the same voxels from where they now are, into a grid of its own (the base plan keeps its grid).
// was, now: the geometry's node_xform then and now. false, with why, for what is not placed so: a volume moving during
// the shutter, a VDB's baked opacity map, a mapped emission's region. deferred: where given, a baked grid is not made
// again here but its volume listed, for a worker (rebakeVolume; the session's).
inline bool placeVolume(Plan& plan,int volumeIndex,const rdl::Mat4d& was,const rdl::Mat4d& now,std::string& why,
                        std::vector<unsigned>* deferred=nullptr){
 volume::Shader& s=plan.volumes.at(size_t(volumeIndex));
 if(s.moving){why="a volume moves during the shutter";return false;}
 if(s.bakedGrid!=volume::NoGrid){why="a VDB volume's opacity is baked from its maps";return false;}
 for(const auto& r:plan.emissiveRegions)if(int(r.volume)==volumeIndex&&(s.mapped&(1u<<volume::MapEmissionColor))){why="a mapped volume's emission is a light";return false;}
 const rdl::Mat4d d=was.inverse()*now,di=d.inverse();
 // The rows of an affine transform (column vectors, as worldToGrid's) and the row-vector matrix they are made from.
 auto fromRows=[](const float* r){rdl::Mat4d m(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);for(int i=0;i<3;++i){for(int c=0;c<3;++c)m[c][i]=r[4*i+c];m[3][i]=r[4*i+3];}return m;};
 auto toRows=[](const rdl::Mat4d& m,float* r){for(int i=0;i<3;++i){for(int c=0;c<3;++c)r[4*i+c]=float(m[c][i]);r[4*i+3]=float(m[3][i]);}};
 toRows(di*fromRows(s.worldToGrid),s.worldToGrid);toRows(di*fromRows(s.worldToGrid1),s.worldToGrid1);
 toRows(fromRows(s.gridToWorld)*d,s.gridToWorld);toRows(fromRows(s.gridToWorld1)*d,s.gridToWorld1);
 // An emissive region's maps (four rows of three, row vectors).
 auto from43=[](const float* r){rdl::Mat4d m(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);for(int i=0;i<4;++i)for(int j=0;j<3;++j)m[i][j]=r[3*i+j];return m;};
 auto to43=[](const rdl::Mat4d& m,float* r){for(int i=0;i<4;++i)for(int j=0;j<3;++j)r[3*i+j]=float(m[i][j]);};
 for(auto& r:plan.emissiveRegions)if(int(r.volume)==volumeIndex){to43(from43(r.distToWorld)*d,r.distToWorld);to43(di*from43(r.worldToDist),r.worldToDist);}
 if(!s.mapped)return true;
 const unsigned k=s.densityGrid;
 const bool baked=k!=volume::NoGrid&&k<plan.volumeGrids.size()&&plan.volumeGrids[k]->key.rfind("baked|",0)==0;
 if(!baked){
  const volume::Mapped m=mappedAt(plan,s,plan.renderOrigin);
  s.originAttenuation=(s.flags&volume::MatchAlbedo)?m.albedoColor:m.attenuationColor;s.originIntensity=m.attenuationIntensity;s.originFactor=m.attenuationFactor;
  return true;}
 // The baked grid again (rebakeVolume): here, or on a worker while the old grid, moved with it, stands in (deferred: the
 // session swaps the new one in once it is made, the move itself not waiting on it).
 if(deferred){deferred->push_back(unsigned(volumeIndex));return true;}
 auto made=rebakeVolume(plan,unsigned(volumeIndex),nullptr);
 if(!made.first){why="a baked volume could not be made again";return false;}
 plan.volumeGrids[k]=std::move(made.first);plan.volumeGridViews[k]=made.second;
 return true;}
inline bool placeMoved(Plan& plan,const std::map<std::string,std::pair<rdl::Mat4d,rdl::Mat4d>>& moves,
                       const std::map<std::string,std::vector<rdl::Mat4d>>& instancePlaces,const double meshW2R[16],std::string& why,
                       std::vector<unsigned>* deferredBakes=nullptr){
 (void)meshW2R;
 size_t placed=0;
 std::set<int> placedVolumes;  // a volume once, however many of the geometry's meshes bound it
 for(auto& e:plan.meshes){
  if(e.owner.empty())continue;
  const auto f=moves.find(e.owner);if(f==moves.end())continue;
  const MeshData& d=*e.data;
  const rdl::Mat4d move=f->second.first.inverse()*f->second.second;
  if(e.ownerInstancer){
   // An instance of shared data its instancer places: where the instancer now puts it.
   const auto p=instancePlaces.find(e.owner);
   if(!e.instance||p==instancePlaces.end()){why=e.owner+"'s instances are not placed by their transforms";return false;}
   if(e.ownerIndex<0||size_t(e.ownerIndex)>=p->second.size()){why=e.owner+" has another count of instances";return false;}
   placeInstance(e,p->second[size_t(e.ownerIndex)]);
   ++placed;continue;}
  // An instance of its own data (placed anew by a preparation, MeshData::placeable): moved on from where it is placed.
  rdl::Mat4d placedNow(scene_rdl2::math::one);
  if(e.instance){
   if(!d.placeable){why=e.owner+" is an instance already";return false;}
   for(int r=0;r<4;++r)for(int c=0;c<3;++c)placedNow[r][c]=e.objectToWorld[3*r+c];}
  if(e.volume>=0&&placedVolumes.insert(e.volume).second&&!placeVolume(plan,e.volume,f->second.first,f->second.second,why,deferredBakes)){why=e.owner+": "+why;return false;}
  if(!d.steps.empty()){why=e.owner+" moves during the shutter";return false;}
  if(d.points||d.curveBasis){why=e.owner+" is points or curves";return false;}
  if(e.cpuOnly){why=e.owner+" is left to the CPU lane";return false;}
  placeInstance(e,placedNow*move);
  const auto w=f->second.second.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)e.worldToObject[i]=float(rows[i]);
  ++placed;}
 // A moved geometry with no mesh of its own here (an instancer's, or one the plan left out): read again whole.
 for(const auto& m:moves){bool found=false;for(const auto& e:plan.meshes)if(e.owner==m.first){found=true;break;}
  if(!found){why=m.first+" has no mesh of its own in the plan";return false;}}
 return placed>0||moves.empty();}
// The light sets' trees (Scene::updateLightList's LightAccelerator::buildSamplingTree), for adaptive light sampling: made
// again from the plan's lights (none kept).
inline void buildLightTrees(Plan& p){
 p.lightTree.clear();p.lightTreeRoots.clear();
 if(!(p.paths&&p.lightSamplingMode==1))return;
 const unsigned words=p.lightSetWords;
 for(size_t k=0;k*words<p.lightSetBits.size();++k){std::vector<unsigned> set;
  for(unsigned i=0;i<p.lights.size();++i)if(p.lightSetBits[k*words+(i>>5)]>>(i&31)&1u)set.push_back(i);
  std::vector<const lights::Light*> close(p.lightClose.size());for(size_t q=0;q<close.size();++q)close[q]=p.lightClose[q].get();
  p.lightTreeRoots.push_back(lights::buildLightTree(p.lights,set,p.lightTree,&close));}}
// A light-only edit (the session's fast path, realtime): each edited light admitted again from the scene as a full
// preparation admits it (admitLight), in its place among the plan's lights, and the light trees made again; meshes,
// materials and the rest as they are. false, with why, where a full preparation is needed: a mesh light or portal, a
// light turned off or new, motion blur, light filters.
inline bool updateLights(Plan& plan,const rdl::SceneContext& context,const std::vector<const rdl::SceneObject*>& edited,std::string& why){
 if(edited.empty())return true;
 if(!plan.paths){why="not the path tracer";return false;}
 if(plan.motion.on){why="motion blur";return false;}
 if(!plan.filterLists.empty()){why="light filters";return false;}
 if(plan.lightNames.size()!=plan.lights.size()){why="lights the plan cannot name";return false;}
 const auto& sv=context.getSceneVariables();
 const std::vector<std::shared_ptr<lights::ImageData>> earlier=plan.lightImages;
 for(const rdl::SceneObject* o:edited){
  const std::string cls=o->getSceneClass().getName();
  if(cls=="MeshLight"||cls=="PortalLight"||!isLight(cls)){why=o->getName()+" is a "+cls;return false;}
  const auto at=std::find(plan.lightNames.begin(),plan.lightNames.end(),o->getName());
  if(at==plan.lightNames.end()){why=o->getName()+" is not among the plan's lights";return false;}
  const size_t i=size_t(at-plan.lightNames.begin());
  lights::Light l;
  if(!admitLight(*o,get<rdl::Float>(sv,"scene_scale"),l,sv,plan.lightImages,&earlier)){why=o->getName()+" is off";return false;}
  l.lpeLabel=plan.lights[i].lpeLabel;
  plan.lights[i]=l;}
 buildLightTrees(plan);
 return true;}
// Where prepare's time goes (the session's log): laps between its stages, the latest prepare's.
struct PrepLaps{std::vector<std::pair<std::string,double>> laps;std::chrono::steady_clock::time_point last=std::chrono::steady_clock::now();
 std::map<std::string,double> parts;std::mutex lock;  // parts within the laps, summed (PrepPart)
 void lap(const char* what){const auto now=std::chrono::steady_clock::now();laps.push_back({what,std::chrono::duration<double>(now-last).count()});last=now;}
 void add(const char* what,double s){std::lock_guard<std::mutex> g(lock);parts[what]+=s;}};
inline PrepLaps& prepLaps(){static PrepLaps l;return l;}
// The tessellation handler's cache (INDEXER.md section 16): a displaced mesh MoonRay tessellates (subdivision or
// PolyMesh) kept on disk as its prepared data, by a hash of what shaped it (the mesh's and its displacement's
// signatures, their textures' files, the level and, for adaptive tessellation, the camera's frustums), so a scene opened
// again or a session started again reuses it. xpu2-geometry in the kit's Cache folder (MOONRAY_MODO_CACHE, its preview
// preferences; else %LOCALAPPDATA%/MoonRayForModo), 20 GB at most (XPU2_SESSION_GEOMETRY_CACHE_GB), the oldest let go;
// XPU2_SESSION_GEOMETRY_CACHE=0: none.
struct GeometryKey{uint64_t a=0x9E3779B97F4A7C15ull,b=0xC2B2AE3D27D4EB4Full;
 void add(const void* data,size_t n){const unsigned char* p=static_cast<const unsigned char*>(data);size_t i=0;
  for(;i+8<=n;i+=8){uint64_t w;std::memcpy(&w,p+i,8);a=(a^w)*0x100000001B3ull;a^=a>>29;b=(b+w)*0x9E3779B97F4A7C15ull;b^=b>>31;}
  uint64_t w=0;std::memcpy(&w,p+i,n-i);a=(a^w^n)*0x100000001B3ull;b=(b+w+n)*0x9E3779B97F4A7C15ull;a^=b>>17;b^=a>>23;}
 void add(const std::string& s){add(s.data(),s.size());}
 std::string hex()const{char t[40];std::snprintf(t,sizeof t,"%016llx%016llx",(unsigned long long)a,(unsigned long long)b);return t;}};
inline bool geometryCacheOn(){const char* v=std::getenv("XPU2_SESSION_GEOMETRY_CACHE");return !(v&&v[0]=='0');}
inline std::filesystem::path geometryCacheFolder(){
 if(const char* chosen=std::getenv("MOONRAY_MODO_CACHE");chosen&&*chosen)return std::filesystem::u8path(chosen)/"xpu2-geometry";
 const char* l=std::getenv("LOCALAPPDATA");
 return std::filesystem::path(l?l:".")/"MoonRayForModo"/"xpu2-geometry";}
// What the cache did in this preparation (the session's log).
struct GeometryCacheCounts{std::atomic<unsigned> reused{0},saved{0};std::atomic<size_t> reusedBytes{0};};
inline GeometryCacheCounts& geometryCacheCounts(){static GeometryCacheCounts c;return c;}
constexpr uint32_t GeometryCacheMagic=0x58324743u,GeometryCacheVersion=1u;  // "X2GC"
template<class T>inline void writeVector(std::ofstream& f,const std::vector<T>& v){const uint64_t n=v.size();f.write(reinterpret_cast<const char*>(&n),8);
 if(n)f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(n*sizeof(T)));}
template<class T>inline bool readVector(std::ifstream& f,std::vector<T>& v,uint64_t limit){uint64_t n=0;f.read(reinterpret_cast<char*>(&n),8);
 if(!f||n>limit/sizeof(T))return false;v.resize(size_t(n));if(n)f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T)));return bool(f);}
// A mesh's prepared data from the cache, or null.
inline std::shared_ptr<MeshData> loadGeometry(const std::string& key){
 namespace sf=std::filesystem;const sf::path path=geometryCacheFolder()/(key+".x2g");std::error_code ec;
 if(!sf::exists(path,ec))return nullptr;const uint64_t size=uint64_t(sf::file_size(path,ec));if(ec)return nullptr;
 std::ifstream f(path,std::ios::binary);uint32_t head[2]={0,0};f.read(reinterpret_cast<char*>(head),8);
 if(!f||head[0]!=GeometryCacheMagic||head[1]!=GeometryCacheVersion)return nullptr;
 auto m=std::make_shared<MeshData>();uint32_t rates[3]={0,0,0};f.read(reinterpret_cast<char*>(rates),12);f.read(reinterpret_cast<char*>(m->worldToObject),sizeof m->worldToObject);
 m->normalRate=rates[0];m->uvRate=rates[1];m->instanced=rates[2]!=0;std::vector<char> dk;
 if(!readVector(f,m->positions,size)||!readVector(f,m->triangles,size)||!readVector(f,m->normals,size)||!readVector(f,m->uvs,size)||
    !readVector(f,m->prims,size)||!readVector(f,m->primData,size)||!readVector(f,m->polyVertices,size)||!readVector(f,m->dpds,size)||
    !readVector(f,m->dpdt,size)||!readVector(f,dk,size))return nullptr;
 for(const auto& tr:m->triangles)if(tr.a>=m->positions.size()||tr.b>=m->positions.size()||tr.c>=m->positions.size())return nullptr;
 m->displacementKey.assign(dk.begin(),dk.end());
 sf::last_write_time(path,sf::file_time_type::clock::now(),ec);  // the most recently used, last to be let go
 geometryCacheCounts().reused++;geometryCacheCounts().reusedBytes+=size_t(size);
 return m;}
// The cache's writers, joined before the session ends (finishGeometryCache): a session that exited while one wrote left
// the file unfinished, and the next session tessellated again.
inline std::mutex& cacheWritersLock(){static std::mutex m;return m;}
inline std::vector<std::thread>& cacheWriters(){static std::vector<std::thread> w;return w;}
inline void finishGeometryCache(){std::vector<std::thread> w;{std::lock_guard<std::mutex> g(cacheWritersLock());w.swap(cacheWriters());}for(auto& t:w)if(t.joinable())t.join();}
// A mesh's prepared data into the cache, on a thread of its own (written under another name, then renamed, so that a
// file the cache holds is whole), the oldest files let go past the cache's size.
inline void saveGeometry(const std::string& key,std::shared_ptr<const MeshData> m){
 if(!m->steps.empty()||m->points||m->curveBasis||m->gate)return;
 geometryCacheCounts().saved++;
 std::thread writer([key,m]{namespace sf=std::filesystem;std::error_code ec;const sf::path folder=geometryCacheFolder();sf::create_directories(folder,ec);
  const sf::path path=folder/(key+".x2g"),part=folder/(key+".part");
  {std::ofstream f(part,std::ios::binary|std::ios::trunc);if(!f)return;const uint32_t head[2]={GeometryCacheMagic,GeometryCacheVersion};f.write(reinterpret_cast<const char*>(head),8);
   const uint32_t rates[3]={m->normalRate,m->uvRate,m->instanced?1u:0u};f.write(reinterpret_cast<const char*>(rates),12);f.write(reinterpret_cast<const char*>(m->worldToObject),sizeof m->worldToObject);
   writeVector(f,m->positions);writeVector(f,m->triangles);writeVector(f,m->normals);writeVector(f,m->uvs);writeVector(f,m->prims);writeVector(f,m->primData);
   writeVector(f,m->polyVertices);writeVector(f,m->dpds);writeVector(f,m->dpdt);writeVector(f,std::vector<char>(m->displacementKey.begin(),m->displacementKey.end()));
   if(!f){f.close();sf::remove(part,ec);return;}}
  sf::rename(part,path,ec);if(ec){sf::remove(part,ec);return;}
  double gb=20;if(const char* v=std::getenv("XPU2_SESSION_GEOMETRY_CACHE_GB"))gb=std::max(0.1,std::atof(v));
  std::vector<std::pair<sf::file_time_type,sf::path>> files;uint64_t total=0;
  for(const auto& e:sf::directory_iterator(folder,ec))if(e.path().extension()==".x2g"){total+=uint64_t(e.file_size(ec));files.push_back({e.last_write_time(ec),e.path()});}
  std::sort(files.begin(),files.end());
  for(const auto& [when,file]:files){if(double(total)<=gb*double(1ull<<30)||file==path)break;total-=uint64_t(sf::file_size(file,ec));sf::remove(file,ec);}
 });
 std::lock_guard<std::mutex> g(cacheWritersLock());cacheWriters().push_back(std::move(writer));}
// A mesh's key: what shaped its tessellation.
inline std::string geometryKey(const rdl::SceneObject& o,const std::vector<std::shared_ptr<const ix::Buffer>>& sig,const std::string& displacementKey,
                               const Plan& plan,bool shared,const rdl::SceneVariables& sv){
 GeometryKey k;const uint32_t version=GeometryCacheVersion;k.add(&version,4);k.add(displacementKey);k.add(&shared,1);
 for(const auto& b:sig)if(b){k.add(b->source.data(),b->source.size());}
 const unsigned char displace=get<rdl::Bool>(sv,"enable_displacement")?1:0;k.add(&displace,1);
 // Displaced from textures made smaller (texture overflow mode): not what a session with the whole textures may take.
 if(const unsigned side=texture::overflowSide().load())k.add(&side,sizeof side);
 if(texture::overflowCompress().load()){const unsigned char compressed=1;k.add(&compressed,1);}
 // The kit's MoonRay Source Overrides (their switches can change what a displacement's maps give).
 for(const char* name:{"MOONRAY_MODO_TOON_TERMINATOR","MOONRAY_MODO_TOON_ALBEDO","MOONRAY_MODO_USD_SPECULAR","MOONRAY_MODO_OPAQUE_ALPHA",
                       "MOONRAY_MODO_TRANSMISSION_ROUGHNESS"}){const char* v=std::getenv(name);k.add(std::string(name)+"="+(v?v:""));}
 // The textures' files as they are now (a file changed under the same name is another tessellation).
 for(const auto& f:plan.textures){std::error_code ec;const std::string n=f->name.string();k.add(n);
  const auto size=std::filesystem::file_size(n,ec);const auto when=std::filesystem::last_write_time(n,ec).time_since_epoch().count();k.add(&size,sizeof size);k.add(&when,sizeof when);}
 // MoonRay's tessellation itself (xpu2_geom.dll beside the session).
 {wchar_t self[32768];const DWORD n=GetModuleFileNameW(nullptr,self,32768);std::error_code ec;
  const auto dll=std::filesystem::path(std::wstring(self,n)).parent_path()/"xpu2_geom.dll";const auto size=std::filesystem::file_size(dll,ec);
  const auto when=std::filesystem::last_write_time(dll,ec).time_since_epoch().count();k.add(&size,sizeof size);k.add(&when,sizeof when);}
 // Adaptive tessellation follows the camera: its frustums and render space.
 if(get<rdl::Float>(o,"adaptive_error")>0.0f){k.add(plan.frustumC2S.data(),plan.frustumC2S.size()*sizeof(float));k.add(plan.frustumPlanes.data(),plan.frustumPlanes.size()*sizeof(float));
  k.add(plan.frustumViewports.data(),plan.frustumViewports.size()*sizeof(int));k.add(plan.world2render,sizeof plan.world2render);}
 return k.hex();}
// A mesh MoonRay tessellates on a worker of its own (the tessellation handler, INDEXER.md section 16): the scene's other
// meshes are prepared meanwhile, and its plan mesh (index mesh) takes the result after the loop, with its budget and
// cache. Its displacement is its own (SubdDisplacement, a copy of the material pool: the plan's grows as the loop goes).
struct TessellationJob{size_t mesh=0;size_t estimate=0;std::string cacheKey,displacementKey;std::shared_ptr<SubdDisplacement> displacement;
 std::future<std::shared_ptr<const MeshData>> result;};
// How many tessellations run at once: one unless XPU2_SESSION_TESSELLATION_JOBS says more (experimental). MoonRay's
// tessellation of displaced meshes goes through one displacement hook at a time (xpu2_geom's displaceMutex: the scene's
// displacement objects are shared and configured per tessellation), and MoonRay keeps thread-local state for at most four
// threads it did not start (ThreadLocalState's MAX_OVERFLOW_POOL_SIZE): four workers of std::async ran one after another
// and then failed that assertion. Running meshes at once needs the hook per mesh and a fixed pool of workers first.
inline size_t tessellationJobs(){if(const char* v=std::getenv("XPU2_SESSION_TESSELLATION_JOBS")){const long n=std::atol(v);if(n>0)return size_t(n);}
 return 1;}
// The tessellation handler's routing by device (INDEXER.md section 16): the triangle meshes' data the GPU would hold
// (each shared data once), and past the card's budget the largest left to the CPU lane alone, every plan mesh of that
// data with it (Mesh::cpuOnly: a stand-in on the GPU, a sample that meets it rendered by the CPU from its identity, the
// image unchanged). Volume bounds, curves and points stay as they are. A moving mesh holds each of its time steps.
inline void routeMeshes(Plan& plan,size_t gpuBytes){
 std::map<const MeshData*,size_t> held;
 for(const auto& m:plan.meshes){const MeshData& d=*m.data;if(m.volume>=0||d.points||d.curveBasis)continue;
  held[&d]=(1+d.steps.size())*d.positions.size()*sizeof(V3)+d.triangles.size()*sizeof(Tri);}
 size_t total=0;for(const auto& [d,b]:held)total+=b;
 std::vector<std::pair<size_t,const MeshData*>> largest;for(const auto& [d,b]:held)largest.push_back({b,d});
 std::sort(largest.begin(),largest.end(),[](const auto& a,const auto& b){return a.first>b.first;});
 std::set<const MeshData*> left;
 for(const auto& [b,d]:largest){if(total<=gpuBytes)break;left.insert(d);total-=b;}
 plan.gpuBytes=total;plan.cpuOnlyMeshes=0;
 for(auto& m:plan.meshes)if(left.count(m.data.get())){m.cpuOnly=true;++plan.cpuOnlyMeshes;}}
inline PrepPart::~PrepPart(){prepLaps().add(what,double(nowNs()-began)*1e-9);}
X2_SPLIT_INLINE PlanView prepare(const rdl::SceneContext& context,const CanonicalView& frame,PlanView previous={},Limits limits={},const fs::path& assetRoot={},Domain domain=Domain::Planar,bool allowAdaptive=false,bool paths=false);
X2_SPLIT_INLINE PlanView prepare(CanonicalSession& session,const CanonicalView& frame,PlanView previous={},Limits limits={},const fs::path& assetRoot={},Domain domain=Domain::Planar,bool allowAdaptive=false,bool paths=false);
#if !defined(XPU2_SPLIT_UNITS) || defined(XPU2_SCENE_UNIT)
X2_SPLIT_INLINE PlanView prepare(const rdl::SceneContext& context,const CanonicalView& frame,PlanView previous,Limits limits,const fs::path& assetRoot,Domain domain,bool allowAdaptive,bool paths){
 // The coarse first pass's tessellation for as long as this plan is prepared (coarse_geometry.h).
 struct CoarseFor{explicit CoarseFor(bool on){coarseTessellation()=on;}~CoarseFor(){coarseTessellation()=false;}} coarseFor(limits.coarse);
 need(bool(frame),"Missing canonical snapshot");{auto& l=prepLaps();l.laps.clear();l.parts.clear();l.last=std::chrono::steady_clock::now();}subdTimes().reset();
 {auto& c=geometryCacheCounts();c.reused=0;c.saved=0;c.reusedBytes=0;}
 const Limits hard{};
 need(limits.vertices>0&&limits.vertices<=hard.vertices&&limits.triangles>0&&limits.triangles<=hard.triangles&&
      limits.bytes>0&&limits.bytes<=HardGeometryBytes&&limits.objects>0&&limits.objects<=hard.objects&&limits.pixels>0&&limits.pixels<=hard.pixels,"Invalid preparation limits");
 auto plan=std::make_shared<Plan>();plan->source=frame;plan->domain=domain;need(domain==Domain::Planar||domain==Domain::DiffuseMesh||gpuTextures(domain),"Unknown scene domain");bool spatial=isSpatial(domain);
 const auto& sv=context.getSceneVariables();
 // The render outputs (RenderOutputDriver): the AOVs the samples gather, before the materials take their labels.
 if(paths){std::shared_ptr<aov::Plan> outputs;
  try{outputs=aov::buildPlan(context);}catch(const std::exception& e){throw Unsupported(std::string("Render outputs: ")+e.what());}
  for(const auto& p:outputs->prims)plan->aovPrimKeys.push_back(shade::primKey(p.first.c_str(),p.second));
  plan->aovs=outputs;}
 // A material's AOV labels (aov::Plan::labelTable) in the pool, where an AOV reads labels.
 auto labelTable=[&](const rdl::SceneObject& m,shade::MaterialRecord& rec){
  if(!plan->aovs||!plan->aovs->labelsNeeded())return;const auto t=plan->aovs->labelTable(m);rec.labels=unsigned(plan->materialPool.size());
  for(unsigned w:t){float f;std::memcpy(&f,&w,sizeof f);plan->materialPool.push_back(f);}};
 // Bound native geometry reads BEFORE copying prepared values.
 size_t objects=0,verts=0,tris=0;
 std::map<std::string,const rdl::SceneObject*> sorted;
 for(auto i=context.beginSceneObject();i!=context.endSceneObject();++i){auto* o=i->second;need(++objects<=limits.objects,"Object budget");sorted.emplace(o->getName(),o);
  if(o->getSceneClass().getName()=="RdlMeshGeometry"){
   auto& v=o->get<rdl::Vec3fVector>("vertex_list");auto& t=o->get<rdl::IntVector>("face_vertex_count");auto& ind=o->get<rdl::IntVector>("vertices_by_index");
   size_t faces=0,corners=0;for(int n:t){faces+=n>=3?size_t(n-2):0;corners+=n>0?size_t(n):0;}need(corners==ind.size(),o->getName()+": face vertex counts ("+std::to_string(corners)+" corners in "+std::to_string(t.size())+" faces) do not match the vertex indices ("+std::to_string(ind.size())+")");
   need(v.size()<=limits.vertices-verts&&faces<=limits.triangles-tris,"Geometry count budget");verts+=v.size();tris+=faces;
  }
 }
 need(verts*sizeof(V3)+tris*sizeof(Tri)<=limits.bytes,geometryBudgetText(verts*sizeof(V3)+tris*sizeof(Tri),limits.bytes));
 attributes(sv,{"enable_displacement","dicing_camera","resumable_output","checkpoint_active","checkpoint_interval","checkpoint_bg_write","resume_render","exr_header_attributes","camera","layer","image_width","image_height","pixel_samples","sampling_mode","min_adaptive_samples","max_adaptive_samples","target_adaptive_error","frame","lock_frame_noise","pixel_filter","pixel_filter_width","bsdf_samples","light_samples","max_depth","max_diffuse_depth","enable_dof","enable_motion_blur","sample_clamping_value","shadow_terminator_fix","two_stage_output","russian_roulette_threshold","lights_visible_in_camera","light_sampling_mode","light_sampling_quality","max_glossy_depth","max_mirror_depth","sample_clamping_depth","scene_scale",
  "max_hair_depth","max_presence_depth","presence_threshold","presence_quality","enable_presence_shadows","roughness_clamping_factor","transparency_threshold","bssrdf_samples","max_subsurface_per_path",
  "enable_subsurface_scattering","max_volume_depth","volume_quality","volume_shadow_quality","volume_illumination_samples",
  "volume_opacity_threshold","volume_overlap_mode","volume_attenuation_factor","volume_contribution_factor","volume_phase_attenuation_factor",
  // What changes only MoonRay's scheduling or its texture cache's size, and motion blur's steps and frame rate.
  "progressive_tile_order","checkpoint_tile_order","batch_tile_order","texture_cache_size","fps","motion_steps","max_geometry_resolution",
  // The windows (rezed: the resolution divisor), the pixels rendered and the texture blur (the path tracer's camera).
  "res","aperture_window","region_window","sub_viewport","debug_pixel","texture_blur",
  // What only deep outputs read, which are refused.
  "deep_id_attribute_names","deep_format","deep_curvature_tolerance","deep_z_tolerance","deep_vol_compression_res","deep_max_layers","deep_layer_bias"});
 // The frame: the rezed region window (SceneVariables::getRezedWidth and getRezedHeight).
 const float resolution=get<rdl::Float>(sv,"res");need(std::isfinite(resolution)&&resolution>0,"Invalid resolution divisor");
 int width=int(sv.getRezedRegionWindow().width()),height=int(sv.getRezedRegionWindow().height()),grid=get<rdl::Int>(sv,"pixel_samples");
 if(!paths)need(resolution==1&&width==get<rdl::Int>(sv,"image_width")&&height==get<rdl::Int>(sv,"image_height"),"Windows unsupported");
 need(width>0&&height>0&&size_t(width)*height<=limits.pixels,"Image budget");need(grid>0&&grid<=64,"Sample budget");plan->width=width;plan->height=height;plan->spp=unsigned(grid*grid);
 int samplingMode=get<rdl::Int>(sv,"sampling_mode");
 // The kit's XPU2 sampling controls rule its previews (its UserData "/modo/xpu2/generations" of form 2): a scene's
 // adaptive sampling is then MoonRay's alone, and XPU2 samples from Pixel Samples and the kit's controls.
 {const auto k=sorted.find("/modo/xpu2/generations");
  if(samplingMode==2&&k!=sorted.end()&&k->second->getSceneClass().getName()=="UserData"){
   const auto& v=get<rdl::FloatVector>(*k->second,"float_values_0");if(v.size()>=8&&v[0]==2.0f)samplingMode=0;}}
 need(samplingMode==0||(samplingMode==2&&allowAdaptive),"Adaptive sampling unsupported");
 if(samplingMode==2){
  // The same clamping as MoonRay's RenderContext: at least 2 samples, a maximum no
  // lower than the minimum, and the target error divided by 10,000.
  plan->adaptive=true;plan->minSamples=unsigned(std::max(2,get<rdl::Int>(sv,"min_adaptive_samples")));
  plan->maxSamples=unsigned(std::max(int(plan->minSamples),get<rdl::Int>(sv,"max_adaptive_samples")));
  plan->targetError=std::max(0.000001f,get<rdl::Float>(sv,"target_adaptive_error")/10000.0f);
  need(plan->maxSamples<=16384,"Adaptive sample budget");plan->spp=plan->maxSamples;
 }
 need(paths||!get<rdl::Bool>(sv,"enable_motion_blur"),"Motion blur unsupported");need(paths||!get<rdl::Bool>(sv,"enable_dof"),"DOF unsupported");
 {const auto v=sv.getRezedSubViewport();plan->viewport[0]=v.min().x;plan->viewport[1]=v.min().y;plan->viewport[2]=v.max().x;plan->viewport[3]=v.max().y;
  need(paths||(v.min().x==0&&v.min().y==0&&v.max().x==width&&v.max().y==height),"Sub viewport unsupported");
  need(v.max().x>v.min().x&&v.max().y>v.min().y,"Empty sub viewport");}
 {// PathIntegrator::initPrimaryRay's texture differential scale: texture_blur, over the resolution divisor, never below one.
  const float primaryScale=1.0f/std::min(std::sqrt(float(plan->spp)),2.0f),scale=get<rdl::Float>(sv,"texture_blur")+1.0f;
  plan->textureDiffScale=std::max((primaryScale<1?2.0f*scale-1.0f:scale)/resolution,1.0f);
  need(std::isfinite(plan->textureDiffScale),"Invalid texture blur");need(paths||plan->textureDiffScale==1.0f,"Texture blur unsupported");}
 int depth=get<rdl::Int>(sv,"max_depth"),diffuse=get<rdl::Int>(sv,"max_diffuse_depth");
 if(paths){
  need(spatial,"The path tracer renders spatial scenes");
  int bsdfSamples=get<rdl::Int>(sv,"bsdf_samples"),lightSamples=get<rdl::Int>(sv,"light_samples"),clampDepth=get<rdl::Int>(sv,"sample_clamping_depth");
  float rr=get<rdl::Float>(sv,"russian_roulette_threshold"),clamp=get<rdl::Float>(sv,"sample_clamping_value");
  // MoonRay's bsdf_samples and light_samples are square roots: a lobe takes bsdf_samples squared samples at the first hit.
  need(bsdfSamples>=1&&bsdfSamples<=8&&lightSamples>=1&&lightSamples<=8,"Sample splitting budget (1..8)");
  need(depth>=0&&depth<=16&&diffuse>=0&&diffuse<=16&&clampDepth>=0&&clampDepth<=16,"Bounce budget");
  need(std::isfinite(rr)&&rr>=0&&rr<=1&&std::isfinite(clamp)&&clamp>=0,"Invalid roulette/clamping");
  {const int mode=get<rdl::Int>(sv,"light_sampling_mode");const float quality=get<rdl::Float>(sv,"light_sampling_quality");
   need((mode==0||mode==1)&&std::isfinite(quality),"Invalid light sampling mode");plan->lightSamplingMode=unsigned(mode);plan->lightSamplingQuality=quality;}
  plan->paths=true;plan->bsdfSamples=unsigned(bsdfSamples*bsdfSamples);plan->lightSamples=unsigned(lightSamples*lightSamples);plan->maxDepth=unsigned(depth);
  plan->maxDiffuseDepth=unsigned(diffuse);plan->rrThreshold=rr;plan->clampDepth=unsigned(clampDepth);
  // PathIntegrator::initPrimaryRay: the clamp is divided by the BSDF sample count, so the look does not change with it.
  plan->clampValue=clamp/float(bsdfSamples*bsdfSamples);
  int glossy=get<rdl::Int>(sv,"max_glossy_depth"),mirror=get<rdl::Int>(sv,"max_mirror_depth"),hair=get<rdl::Int>(sv,"max_hair_depth"),presenceDepth=get<rdl::Int>(sv,"max_presence_depth");
  need(glossy>=0&&glossy<=16&&mirror>=0&&mirror<=16&&hair>=0&&hair<=16&&presenceDepth>=0&&presenceDepth<=64,"Bounce budget");
  {const float threshold=get<rdl::Float>(sv,"presence_threshold"),quality=get<rdl::Float>(sv,"presence_quality");
   need(std::isfinite(threshold)&&std::isfinite(quality),"Invalid presence settings");plan->presenceThreshold=threshold;plan->presenceQuality=quality;}
  plan->maxGlossyDepth=unsigned(glossy);plan->maxMirrorDepth=unsigned(mirror);plan->maxHairDepth=unsigned(hair);plan->maxPresenceDepth=unsigned(presenceDepth);
  int stf=get<rdl::Int>(sv,"shadow_terminator_fix");need(stf>=0&&stf<=4,"Invalid shadow terminator fix");plan->shadowTerminatorFix=unsigned(stf);
  float roughnessClamping=get<rdl::Float>(sv,"roughness_clamping_factor"),transparency=get<rdl::Float>(sv,"transparency_threshold");
  need(std::isfinite(roughnessClamping)&&roughnessClamping>=0&&std::isfinite(transparency),"Invalid roughness clamping/transparency threshold");
  plan->roughnessClamping=roughnessClamping;plan->transparencyThreshold=transparency;
  // PathIntegrator: bssrdf_samples is a square root (at least two samples); the subsurface integrator keeps
  // at most sixteen exit samples a hit. More asked for take sixteen: the same image on average, as noisy as 4 x 4, where
  // more than 4 used to send the whole scene to MoonRay XPU.
  int bssrdf=get<rdl::Int>(sv,"bssrdf_samples"),maxSss=get<rdl::Int>(sv,"max_subsurface_per_path");
  need(bssrdf>=1&&maxSss>=0&&maxSss<=16,"Subsurface sample budget");
  if(bssrdf>4){std::printf("MoonRay XPU2: Subsurface Samples %d: XPU2 takes 4 x 4 at a hit (raise Pixel Samples for less noise)%c",bssrdf,char(10));bssrdf=4;}
  plan->bssrdfSamples=unsigned(std::max(2,bssrdf*bssrdf));plan->maxSubsurfacePerPath=unsigned(maxSss);
  plan->enableSss=get<rdl::Bool>(sv,"enable_subsurface_scattering")?1u:0u;
  // The volume settings (RenderContext's integrator parameters): the transmittance threshold from the opacity one, the
  // factors clamped to [0, 1]. Volume quality only steps heterogeneous volumes (none are admitted).
  int maxVolumeDepth=get<rdl::Int>(sv,"max_volume_depth"),illumination=get<rdl::Int>(sv,"volume_illumination_samples"),overlap=get<rdl::Int>(sv,"volume_overlap_mode");
  need(maxVolumeDepth>=0&&maxVolumeDepth<=16&&illumination>=0&&illumination<=64&&overlap>=0&&overlap<=2,"Volume settings budget");
  auto unit=[](float f){return std::isfinite(f)?std::min(std::max(f,0.0f),1.0f):0.0f;};
  plan->maxVolumeDepth=unsigned(maxVolumeDepth);plan->volumeIlluminationSamples=unsigned(illumination);plan->volumeOverlapMode=unsigned(overlap);
  plan->volumeTransmittanceThreshold=unit(1.0f-get<rdl::Float>(sv,"volume_opacity_threshold"));
  plan->volumeAttenuationFactor=unit(get<rdl::Float>(sv,"volume_attenuation_factor"));
  plan->volumeContributionFactor=unit(get<rdl::Float>(sv,"volume_contribution_factor"));
  plan->volumePhaseAttenuationFactor=unit(get<rdl::Float>(sv,"volume_phase_attenuation_factor"));
  {const float q=get<rdl::Float>(sv,"volume_quality"),sq=get<rdl::Float>(sv,"volume_shadow_quality");
   need(std::isfinite(q)&&q>0.0f&&std::isfinite(sq)&&sq>0.0f,"Invalid volume quality");plan->volumeQuality=q;plan->volumeShadowQuality=sq;}
 }else{
 need(get<rdl::Int>(sv,"bsdf_samples")==1&&get<rdl::Int>(sv,"light_samples")==1,"Primary splitting unsupported");
 if(domain==Domain::SurfaceTextures)need(depth==0&&diffuse==0,"Spatial textures require direct-only transport until secondary derivatives are qualified");
 if(spatial){need(depth>=0&&depth<=7&&diffuse>=0&&diffuse<=7,"Spatial bounce budget (0..7 indirect)");plan->surfaceLimit=1+unsigned(std::min(depth,diffuse));}
 else need(depth==2&&diffuse==1,"Only the tested single diffuse event supported");
 need(get<rdl::Float>(sv,"sample_clamping_value")==0&&get<rdl::Float>(sv,"russian_roulette_threshold")==0,"Clamping/roulette unsupported");
 }
 need(paths||get<rdl::Int>(sv,"shadow_terminator_fix")==0,"Shadow terminator correction unsupported");
 // lights_visible_in_camera: each light's visible_in_camera reads it (lightVisibleInCamera).
 float frameNumber=get<rdl::Float>(sv,"frame");need(std::isfinite(frameNumber)&&frameNumber>=0&&double(frameNumber)<=4294967295.&&std::floor(frameNumber)==frameNumber,"Invalid integer frame");
 plan->seed=surface::frameSeed(unsigned(frameNumber),0,get<rdl::Bool>(sv,"lock_frame_noise"));
 int filter=get<rdl::Int>(sv,"pixel_filter");plan->filterWidth=get<rdl::Float>(sv,"pixel_filter_width");need(filter>=0&&filter<=5&&surface::finite(plan->filterWidth)&&plan->filterWidth>0&&plan->filterWidth<=64,"Unsupported pixel filter");plan->filter=surface::Filter(filter);
 auto* camera=get<rdl::SceneObject*>(sv,"camera");auto* layerObject=get<rdl::SceneObject*>(sv,"layer");
 need(camera&&(paths||camera->getSceneClass().getName()=="PerspectiveCamera"),"Perspective camera required");need(layerObject&&layerObject->isA<rdl::Layer>(),"Layer required");auto& layer=*layerObject->asA<rdl::Layer>();
 // The lens's depth of field, bokeh and shutter count only with depth of field or motion blur, which are refused.
 if(!paths)attributes(*camera,{"node_xform","focal","film_width_aperture","near","far","dof_aperture","dof_focus_distance","bokeh_sides","bokeh_angle",
  "bokeh_weight_location","bokeh_weight_strength","mb_shutter_open","mb_shutter_close","mb_shutter_bias"});auto cm=plan->paths?rigid(*camera):translation(*camera);need(plan->paths||cm[3][2]>0,"Camera must face the planar surface from positive Z");
 {const auto w=cm.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
  for(int i=0;i<12;++i)plan->worldToRender[i]=float(rows[i]);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)plan->world2render[4*i+j]=w[i][j];}
 // A bake camera's mesh, whose maps are made as it is read.
 const rdl::SceneObject* bakeGeometry=nullptr;
 if(paths){
  // Motion blur's steps and shutter (motionParams), and the samples' shutter bias.
  sceneTimeRescale=timeRescale(sv,camera);
  plan->motion=motionParams(sv,camera);plan->shutterBias=plan->motion.on?get<rdl::Float>(*camera,"mb_shutter_bias"):0.0f;
  need(std::isfinite(plan->shutterBias),camera->getName()+": invalid shutter bias");
  // The path tracer's camera: every MoonRay camera type, its windows and lens (admitCamera, camera_core.h).
  admitCamera(*camera,sv,*plan,previous.get());
  // The pixel sample map (uniform sampling alone: adaptive sampling takes priority over it).
  {const std::string map=get<rdl::String>(*camera,"pixel_sample_map");
   if(!map.empty()&&!plan->adaptive){
    const auto values=readPixelSampleMap(map,plan->width,plan->height);const unsigned original=plan->spp;
    const float most=*std::max_element(values.begin(),values.end());
    plan->pixelSamples.resize(values.size());
    for(size_t i=0;i<values.size();++i){const unsigned n=unsigned(std::floor(float(original)*std::max(0.0f,values[i])));plan->pixelSamples[i]=n?n:1u;}
    // FrameState's samples a pixel: the original count times the map's largest value.
    plan->spp=unsigned(float(original)*most);
    for(unsigned n:plan->pixelSamples)X2_NEED(n<=plan->spp&&n<=16384,camera->getName()+": pixel sample map budget");}}
  // A uniform frame by sampling generations (XPU2_SESSION_UNIFORM_BUDGET, on unless 0): uniform's time, spent where
  // it shows. Every pixel is seeded with an eighth of its samples (16 at least), whose rate prices uniform's samples
  // as uniform's time; each generation after hands the next pass to the tiles whose error is above a target set so
  // the samples the time left buys would just be spent (async_adaptive.h Cellular, async_render.h Adaptive), and
  // the frame ends when the time is up (at most twice uniform's samples). Not with a pixel sample map, for a bake
  // camera, nor at 16 samples a pixel or fewer (the seed would be all of them).
  // The kit's render settings for them (XPU2 Sampling Generations): UserData "/modo/xpu2/generations", float_values_0
  // = {form (1; 2: they rule, a scene's adaptive sampling left to MoonRay, above), on, samples a pixel on average (0: the scene's), spend (0: uniform's time, 1: exactly the samples), noise
  // target (target_adaptive_error's units; 0: none, or an adaptive scene's own), first generation (0: an eighth, 16
  // at least; adaptive: its minimum), most samples a pixel (0: 16 times the average; adaptive: its maximum), CPU
  // workers}. Without it, the defaults. XPU2_SESSION_UNIFORM_BUDGET=0 turns generations off (MoonRay's uniform
  // exactly, for the rounds against MoonRay's renders).
  {struct{bool on=true;unsigned samples=0;bool bySamples=false;float noise=0;unsigned seed=0,most=0;bool cpu=true,realtime=false;}g;
   const auto found=sorted.find("/modo/xpu2/generations");
   if(found!=sorted.end()&&found->second->getSceneClass().getName()=="UserData"){
    const auto& v=get<rdl::FloatVector>(*found->second,"float_values_0");
    if(v.size()>=8&&(v[0]==1.0f||v[0]==2.0f)){g.on=v[1]!=0;g.samples=unsigned(std::clamp(v[2],0.0f,4096.0f));g.bySamples=v[3]!=0;
     g.noise=std::isfinite(v[4])?std::max(0.0f,v[4]):0.0f;g.seed=unsigned(std::clamp(v[5],0.0f,16384.0f));
     g.most=unsigned(std::clamp(v[6],0.0f,16384.0f));g.cpu=v[7]!=0;if(v.size()>=9)g.realtime=v[8]!=0;}}
   const char* off=std::getenv("XPU2_SESSION_UNIFORM_BUDGET");if(off&&off[0]=='0')g.on=false;
   plan->generationsCpu=g.cpu;plan->realtime=g.realtime;
   const unsigned long long pixels=(unsigned long long)(plan->viewport[2]-plan->viewport[0])*(unsigned long long)(plan->viewport[3]-plan->viewport[1]);
   if(allowAdaptive&&g.on&&plan->pixelSamples.empty()&&plan->camera.type!=surface::CameraBake){
    if(!plan->adaptive){
     const unsigned n=g.samples?g.samples:plan->spp;
     if(n>16||g.samples){
      plan->adaptive=true;plan->minSamples=std::min(n,g.seed?g.seed:std::max(16u,n/8));
      plan->maxSamples=std::min(16384u,std::max(n,g.most?g.most:n*16));
      plan->noiseTarget=g.noise/10000.0f;plan->targetError=plan->noiseTarget>0?plan->noiseTarget:1.0f;
      // spp is the average: the ray differentials (texture footprints, curves) and the frame's progress are uniform's.
      plan->spp=n;plan->sampleBudget=(unsigned long long)n*pixels;plan->budgetBySamples=g.bySamples;}
    }else{
     // An adaptive scene: its own target error, minimum and maximum unless the settings give theirs; with samples, a
     // budget besides, the frame ending at whichever it reaches first.
     if(g.noise>0)plan->targetError=g.noise/10000.0f;
     if(g.seed)plan->minSamples=std::max(2u,std::min(16384u,g.seed));
     if(g.most)plan->maxSamples=std::max(plan->minSamples,std::min(16384u,g.most));
     plan->maxSamples=std::max(plan->maxSamples,plan->minSamples);plan->spp=plan->maxSamples;
     if(g.samples){plan->sampleBudget=(unsigned long long)g.samples*pixels;plan->budgetBySamples=g.bySamples;plan->noiseTarget=plan->targetError;}}}}
  if(plan->camera.type==surface::CameraBake){
   bakeGeometry=get<rdl::SceneObject*>(*camera,"geometry");
   need(bakeGeometry&&bakeGeometry->getSceneClass().getName()=="RdlMeshGeometry",camera->getName()+": the bake camera bakes a mesh (RdlMeshGeometry)");
  }
 }else{
 float focal=get<rdl::Float>(*camera,"focal"),film=get<rdl::Float>(*camera,"film_width_aperture");
 auto& cam=plan->camera;cam.nearClip=get<rdl::Float>(*camera,"near");cam.farClip=get<rdl::Float>(*camera,"far");
 need(surface::finite(focal)&&surface::finite(film)&&focal>0&&film>0&&surface::finite(cam.nearClip)&&surface::finite(cam.farClip)&&cam.nearClip>0&&cam.farClip>cam.nearClip,"Invalid lens");
 for(int i=0;i<4;++i){cam.rasterToCamera.m[i][i]=1;for(int j=0;j<4;++j)cam.cameraToRender.m[i][j]=float(cm[i][j]);}
 cam.rasterToCamera.m[0][0]=cam.rasterToCamera.m[1][1]=film/(focal*width);
 cam.rasterToCamera.m[3][0]=-film/(2*focal);cam.rasterToCamera.m[3][1]=-film/(2*focal)*height/width;
 }
 std::map<std::string,unsigned> materialIds;std::set<const rdl::SceneObject*> allGeometry,setGeometry;const rdl::SceneObject* sky=nullptr;
 std::map<const rdl::SceneObject*,uint64_t> familyCache;  // indexedFamilies' objects
 std::map<const rdl::SceneObject*,unsigned> lightIds;
 // The materials' lobe light sets, gathered as they are admitted (dwa_admission.h).
 std::vector<const rdl::SceneObject*> lobeLightSets;lobeLightSetTable=&lobeLightSets;
 struct LobeTableGuard{~LobeTableGuard(){lobeLightSetTable=nullptr;}} lobeTableGuard;
 // Lights moving under motion blur, and their records at the shutter's close.
 std::map<const rdl::SceneObject*,lights::LightMotion> movingLights;std::map<const rdl::SceneObject*,std::shared_ptr<const lights::Light>> closeLights;
 unsigned outputCount=0;bool beauty=false,alpha=false;
 std::map<const rdl::SceneObject*,texture::Binding> maps;std::shared_ptr<OIIO::TextureSystem> texSystem;
 // The meshes to displace (plan mesh, edge vertex count, shader) once their maps' textures are read.
 struct Displaced{size_t mesh;int edgeVertices;displace::Shader shader;};std::vector<Displaced> displaced;
 // Each subdivision surface tessellated once, whatever its parts (tessellateSubdivision).
 std::map<const rdl::SceneObject*,std::shared_ptr<const SubdSurface>> subdCache;
 // Under motion blur: each time step's tessellation.
 std::map<const rdl::SceneObject*,std::vector<std::shared_ptr<const SubdSurface>>> subdMotionCache;
 // Each displaced polygon mesh MoonRay's PolyMesh makes, once whatever its parts (tessellatePolygons).
 std::map<const rdl::SceneObject*,std::shared_ptr<const PolySurface>> polyCache;
 // An instancer's prototypes, tessellated once in their own space for every instance (MoonRay's shared primitives).
 std::map<const rdl::SceneObject*,std::shared_ptr<const SubdSurface>> subdRefCache;std::map<const rdl::SceneObject*,std::shared_ptr<const PolySurface>> polyRefCache;
 // The mapped volumes (admitMappedVolume), baked once the textures are read (bakeVolume): volume, mesh, shader, the
 // mesh's node transform, whether a polygon mesh.
 struct MappedVolume{unsigned volume;size_t mesh;const rdl::SceneObject* shader;rdl::Mat4d nodeXform;bool polygons;const rdl::SceneObject* vdb=nullptr;};
 std::vector<MappedVolume> mappedVolumes;
 TextureContext textureContext{assetRoot,&texSystem,&plan->textures,previous?&previous->textures:nullptr};
 for(auto& pair:sorted){auto& o=*pair.second;if(paths||o.getSceneClass().getName()!="ImageMap")continue;
  need(!spatial||domain==Domain::SurfaceTextures,"Spatial map derivatives not implemented; use reference");
  need(!assetRoot.empty()&&maps.size()<8,"Textured preparation requires frozen asset admission; at most 8 maps");
  attributes(o,{"texture","wrap_around","gamma","scale","offset","mip_bias"});
  need(get<rdl::Int>(o,"texture_coordinates")==0&&!get<rdl::Bool>(o,"alpha_only"),"Only RGB mesh-ST ImageMap supported");
  texture::Binding binding;
  auto texturePath=get<rdl::String>(o,"texture");texture::admittedPath(fs::path(texturePath),assetRoot);
  // A retained File holds a Windows read lease, so unchanged assets are still
  // immutable and the same host mip cache can survive camera/material edits.
  for(auto& admitted:maps)if(admitted.second.file->name.string()==texturePath)binding.file=admitted.second.file;
  if(!binding.file&&previous)for(auto* set:{&previous->diffuseMaps,&previous->emissionMaps})for(auto& old:*set)if(old.second.file->name.string()==texturePath)binding.file=old.second.file;
  if(binding.file)texSystem=binding.file->system;else {if(!texSystem)texSystem=texture::system();binding.file=std::make_shared<texture::File>(fs::path(texturePath),assetRoot,texSystem);}
  auto scale=get<rdl::Vec2f>(o,"scale"),offset=get<rdl::Vec2f>(o,"offset");binding.sx=scale.x;binding.sy=scale.y;binding.ox=offset.x;binding.oy=offset.y;
  binding.wrap=get<rdl::Bool>(o,"wrap_around");binding.mipBias=get<rdl::Float>(o,"mip_bias");int gamma=get<rdl::Int>(o,"gamma");need(gamma>=0&&gamma<=2,"Invalid gamma mode");binding.gamma=gamma!=0;
  need(std::isfinite(scale.x)&&std::isfinite(scale.y)&&std::abs(scale.x)<=256&&std::abs(scale.y)<=256&&std::isfinite(offset.x)&&std::isfinite(offset.y)&&std::abs(offset.x)<=256&&std::abs(offset.y)<=256&&std::isfinite(binding.mipBias)&&binding.mipBias>=-1&&binding.mipBias<=32,"Texture transform/bias budget");maps.emplace(&o,std::move(binding));
 }

 // The materials the layer's assignments reach, through layered and switched materials: the only ones shaded, as
 // MoonRay shades only what is assigned (a material left in the scene unassigned is not looked at).
 std::set<const rdl::SceneObject*> usedMaterials;
 if(paths){std::vector<const rdl::SceneObject*> todo;
  for(int id=0;id<int(layer.getAssignmentCount());++id)if(const rdl::SceneObject* m=layer.lookupMaterial(id))todo.push_back(m);
  while(!todo.empty()){const rdl::SceneObject* m=todo.back();todo.pop_back();if(!m||!usedMaterials.insert(m).second)continue;
   for(auto i=m->getSceneClass().beginAttributes();i!=m->getSceneClass().endAttributes();++i){const rdl::Attribute& a=**i;
    if(a.getType()==rdl::TYPE_SCENE_OBJECT){const rdl::SceneObject* s=m->get(rdl::AttributeKey<rdl::SceneObject*>(a));if(s&&s->isA<rdl::Material>())todo.push_back(s);}
    else if(a.getType()==rdl::TYPE_SCENE_OBJECT_VECTOR){for(const rdl::SceneObject* s:m->get(rdl::AttributeKey<rdl::SceneObjectVector>(a)))if(s&&s->isA<rdl::Material>())todo.push_back(s);}}}}
 std::vector<const rdl::SceneObject*> switches;
 // The scene's lights are those of the layer's light sets (Scene::updateLightList); any other light is not there.
 std::set<const rdl::SceneObject*> layerLights;
 for(int id=0;id<int(layer.getAssignmentCount());++id)if(const rdl::LightSet* set=layer.lookupLightSet(id))for(const rdl::SceneObject* l:set->getLights())layerLights.insert(l);
 // A mesh light's geometry outside the layer is the light's alone (Scene::generateMeshLightGeometry): not rendered.
 std::set<const rdl::SceneObject*> meshLightGeometry;
 if(paths)for(auto& pair:sorted)if(pair.second->getSceneClass().getName()=="MeshLight")
  if(const rdl::SceneObject* g=get<rdl::SceneObject*>(*pair.second,"geometry"))if(g->isA<rdl::Geometry>()&&!layer.contains(g->asA<rdl::Geometry>()))meshLightGeometry.insert(g);
 prepLaps().lap("outputs, camera, sampling");
 for(auto& pair:sorted){auto& o=*pair.second;auto cls=o.getSceneClass().getName();
  if(cls=="SceneVariables"||&o==camera||maps.count(&o))continue;
  if(paths&&o.isA<rdl::Material>()&&!usedMaterials.count(&o))continue;
  // On the path tracer a map is admitted where a material is bound to it (map_admission.h).
  if(paths&&(o.isA<rdl::Map>()||o.isA<rdl::NormalMap>()))continue;
  // A displacement is admitted with the assignment that binds it (admitDisplacement).
  if(paths&&o.isA<rdl::Displacement>())continue;
  if(cls=="UsdPreviewSurface"&&paths){
   // The whole material (shade_material.h); maps on its diffuse and emissive colours.
   attributes(o,{"diffuseColor","emissiveColor","useSpecularWorkflow","specularColor","metallic","roughness","clearcoat","clearcoatRoughness","opacity","opacityThreshold","ior","normal","displacement","occlusion"},{"diffuseColor","emissiveColor","specularColor","metallic","roughness","clearcoat","clearcoatRoughness","opacity","ior","normal"});
   shade::UsdPreviewSurfaceParams q{};
   auto c3=[&](const char* n){auto c=get<rdl::Rgb>(o,n);V3 v{c.r,c.g,c.b};need(surface::finite(v),o.getName()+": invalid colour");return v;};
   auto f1=[&](const char* n){float f=get<rdl::Float>(o,n);need(std::isfinite(f),o.getName()+": invalid value");return f;};
   q.diffuseColor=c3("diffuseColor");q.emissiveColor=c3("emissiveColor");q.specularColor=c3("specularColor");
   int workflow=get<rdl::Int>(o,"useSpecularWorkflow");need(workflow==0||workflow==1,o.getName()+": invalid workflow");q.useSpecularWorkflow=float(workflow);
   q.metallic=f1("metallic");q.roughness=f1("roughness");q.clearcoat=f1("clearcoat");q.clearcoatRoughness=f1("clearcoatRoughness");
   q.opacity=f1("opacity");q.opacityThreshold=f1("opacityThreshold");q.ior=f1("ior");need(q.ior>0,o.getName()+": invalid ior");
   auto n=get<rdl::Vec3f>(o,"normal");q.normal={n.x,n.y,n.z};need(surface::finite(q.normal),o.getName()+": invalid normal");
   shade::MaterialRecord rec;rec.kind=shade::MatUsdPreviewSurface;rec.offset=unsigned(plan->materialPool.size());rec.count=shade::UsdPreviewSurfaceFloats;
   // MoonRay reads MOONRAY_MODO_USD_SPECULAR (UsdPreviewSurface.cc) from its environment; so does the renderer here.
   const char* modoSpecular=std::getenv("MOONRAY_MODO_USD_SPECULAR");
   if(modoSpecular&&modoSpecular[0]==49)rec.flags|=shade::MatModoUsdSpecular;
   const float* qf=reinterpret_cast<const float*>(&q);plan->materialPool.insert(plan->materialPool.end(),qf,qf+shade::UsdPreviewSurfaceFloats);
   // Its bound attributes and the maps behind them (map_admission.h).
   {MapProgramBuilder program{plan->materialPool,textureContext};std::vector<shade::MapBinding> bindings;using U=shade::UsdPreviewSurfaceParams;
    program.bindingsOf(o,{{"diffuseColor",X2_OFF(U,diffuseColor)},{"emissiveColor",X2_OFF(U,emissiveColor)},{"specularColor",X2_OFF(U,specularColor)},{"metallic",X2_OFF(U,metallic)},{"roughness",X2_OFF(U,roughness)},{"clearcoat",X2_OFF(U,clearcoat)},{"clearcoatRoughness",X2_OFF(U,clearcoatRoughness)},{"opacity",X2_OFF(U,opacity)},{"ior",X2_OFF(U,ior)},{"normal",X2_OFF(U,normal)}},bindings);
    rec.bindings=unsigned(plan->materialPool.size());rec.bindingCount=unsigned(bindings.size());
    for(auto& b:bindings){const float* bf=reinterpret_cast<const float*>(&b);plan->materialPool.insert(plan->materialPool.end(),bf,bf+sizeof(b)/sizeof(float));}
    rec.maps=program.finish();if(program.polyVertices)rec.flags|=shade::MatPolyVertices;}
   labelTable(o,rec);
   materialIds[o.getName()]=unsigned(plan->materials.size());plan->materials.push_back({q.diffuseColor,q.emissiveColor});plan->materialRecords.push_back(rec);
  }else if(paths&&(cls=="SwitchMaterial"||cls=="RaySwitchMaterial")){
   // Their records once every material is read (shade_material.h SwitchRecord).
   {std::set<std::string> all;for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)all.insert((*i)->getName());attributes(o,all);}
   materialIds[o.getName()]=unsigned(plan->materials.size());plan->materials.push_back({V3{0,0,0},V3{0,0,0}});plan->materialRecords.push_back(shade::MaterialRecord{});switches.push_back(&o);
  }else if(paths&&(dwaClass(cls)||dwaCombinator(cls)||hairCombinator(cls))){
   // The DwaBase family and its combinators (shade_dwabase.h, shade_dwalayer.h, dwa_admission.h): each material its
   // own tree, its inputs' trees inside it.
   shade::MaterialRecord rec=admitDwaTree(o,plan->materialPool,plan->dwaSlots,textureContext,&plan->glitter,&plan->randomWalk);
   labelTable(o,rec);
   materialIds[o.getName()]=unsigned(plan->materials.size());plan->materials.push_back({V3{0,0,0},V3{0,0,0}});plan->materialRecords.push_back(rec);
  }else if(cls=="UsdPreviewSurface"){
   attributes(o,{"diffuseColor","useSpecularWorkflow","specularColor","emissiveColor"},{"diffuseColor","emissiveColor"});
   need(get<rdl::Int>(o,"useSpecularWorkflow")==1&&zero(get<rdl::Rgb>(o,"specularColor")),o.getName()+": Fresnel/specular/metal is not diffuse-only");
   need(get<rdl::Float>(o,"opacity")==1&&get<rdl::Float>(o,"clearcoat")==0,o.getName()+": opacity/coat unsupported");
   materialIds[o.getName()]=unsigned(plan->materials.size());plan->materials.push_back({color(get<rdl::Rgb>(o,"diffuseColor")),color(get<rdl::Rgb>(o,"emissiveColor"))});
   for(auto name:{"diffuseColor","emissiveColor"}){auto* binding=o.getBinding(*o.getSceneClass().getAttribute(name));if(binding){need(maps.count(binding),"Unsupported material map");(std::string(name)=="diffuseColor"?plan->diffuseMaps:plan->emissionMaps).emplace(materialIds.at(o.getName()),maps.at(binding));}}
  }else if(paths&&cls=="MeshLight"){
   lights::Light l;if(layerLights.count(&o)&&admitMeshLight(o,get<rdl::Float>(sv,"scene_scale"),layer,sv,frame,l,plan->meshLightTables,previous?&previous->meshLightTables:nullptr,plan.get())){
    l.rayTerminator=get<rdl::Bool>(o,"ray_termination")?1u:0u;
    if(plan->aovs)l.lpeLabel=plan->aovs->lpeLabel(aov::attribute<rdl::String>(o,"label",""));need(plan->lights.size()<65536,"At most 65536 lights");lightIds[&o]=unsigned(plan->lights.size());plan->lights.push_back(l);plan->lightNames.push_back(o.getName());}
  }else if(paths&&isLight(cls)){
   lights::Light l;if(layerLights.count(&o)&&admitLight(o,get<rdl::Float>(sv,"scene_scale"),l,sv,plan->lightImages,previous?&previous->lightImages:nullptr)){
    if(plan->aovs)l.lpeLabel=plan->aovs->lpeLabel(aov::attribute<rdl::String>(o,"label",""));need(plan->lights.size()<65536,"At most 65536 lights");lightIds[&o]=unsigned(plan->lights.size());plan->lights.push_back(l);plan->lightNames.push_back(o.getName());
    if(plan->motion.on&&get<rdl::Bool>(o,"mb")){
     // LocalParamLight::updateParamAndTransforms: the light at the shutter's close; moving where its transform differs.
     need(cls!="PortalLight",o.getName()+": a portal in motion is not rendered yet");
     lights::Light c;std::vector<std::shared_ptr<lights::ImageData>> unused;
     need(admitLight(o,get<rdl::Float>(sv,"scene_scale"),c,sv,unused,&plan->lightImages,1.0f),o.getName()+": the light is off at the shutter's close");
     namespace sm=scene_rdl2::math;
     const bool oriented=l.type==lights::Env||l.type==lights::Distant;
     const float s0=oriented?1.0f:l.scale,s1=oriented?1.0f:c.scale;
     auto rotation=[&](const lights::Light& x,float s){std::array<sm::Vec3f,3> r;for(int i=0;i<3;++i)r[size_t(i)]=sm::Vec3f(x.local2Render[i].x/s,x.local2Render[i].y/s,x.local2Render[i].z/s);return r;};
     const auto r0=rotation(l,s0),r1=rotation(c,s1);
     lights::LightMotion m;
     const sm::Vec3f p0(l.position.x,l.position.y,l.position.z),p1(c.position.x,c.position.y,c.position.z);
     if(!oriented&&!sm::isEqual(p0,p1))m.flags|=lights::MbTranslation;
     if(!sm::isEqual(r0[0],r1[0])||!sm::isEqual(r0[1],r1[1])||!sm::isEqual(r0[2],r1[2]))m.flags|=lights::MbRotation;
     if(!oriented&&!sm::isEqual(s0,s1))m.flags|=lights::MbScale;
     if(m.flags){
      sm::Quaternion3f q0=sm::normalize(sm::Quaternion3f(r0[0],r0[1],r0[2])),q1=sm::normalize(sm::Quaternion3f(r1[0],r1[1],r1[2]));
      if(sm::dot(q0,q1)<0)q1*=-1.f;
      const float qa[4]={q0.r,q0.i,q0.j,q0.k},qb[4]={q1.r,q1.i,q1.j,q1.k};
      for(int k=0;k<4;++k){m.orientation[0][k]=qa[k];m.orientation[1][k]=qb[k];}
      m.position[0]=l.position;m.position[1]=c.position;m.scale[0]=s0;m.scale[1]=s1;
      movingLights[&o]=m;closeLights[&o]=std::make_shared<const lights::Light>(c);}}}
  }else if(cls=="EnvLight"){
   need(!sky,"Multiple environment lights unsupported");sky=&o;attributes(o,{"color","intensity","exposure","sample_upper_hemisphere_only"});need(!get<rdl::Bool>(o,"sample_upper_hemisphere_only"),"Hemisphere environment unsupported");
   auto c=color(get<rdl::Rgb>(o,"color"));float intensity=get<rdl::Float>(o,"intensity"),exposure=get<rdl::Float>(o,"exposure");need(surface::finite(intensity)&&intensity>=0&&surface::finite(exposure)&&std::abs(exposure)<=32,"Invalid light intensity");plan->sky=surface::mul(c,intensity*std::pow(2.f,exposure));need(surface::finite(plan->sky),"Environment radiance overflow");
  }else if(cls=="RdlMeshGeometry"||(paths&&isCurves(cls))||(paths&&cls=="RdlInstancerGeometry")||(paths&&isPoints(cls))||(paths&&cls=="VdbGeometry")){
   // The path tracer renders the layer's geometry, as MoonRay does; the rest (left out of the layer) is not rendered.
   if(!meshLightGeometry.count(&o)&&(!paths||layer.contains(o.asA<rdl::Geometry>())))allGeometry.insert(&o);}
  else if(cls=="GeometrySet"){
   attributes(o,{"geometries"});for(auto* geom:o.asA<rdl::GeometrySet>()->getGeometries())need(setGeometry.insert(geom).second||paths,"Repeated geometry sets unsupported");
  }else if(cls=="Layer"){
   need(&o==&layer,"Additional layers unsupported");attributes(o,{"geometries","parts","surface_shaders","lightsets","displacements","volume_shaders","lightfiltersets","shadowsets","shadowreceiversets"});
  }else if(cls=="LightSet")attributes(o,{"lights"});
  else if(cls=="RenderOutput"){
   // Paths: every output as MoonRay's output driver reads it (aov_plan.h; the session leaves what XPU2 lacks to MoonRay).
   if(paths){++outputCount;continue;}
   attributes(o,{"file_name","result","channel_format","compression"});int result=get<rdl::Int>(o,"result");need(result==0||result==1,"Unsupported AOV");need(get<rdl::Int>(o,"channel_format")==0,"Only float32 outputs supported");++outputCount;(result?alpha:beauty)=true;
  }else if(paths&&(cls=="PerspectiveCamera"||cls=="OrthographicCamera")){
   // Another camera: a projector, read by the maps that name it (ProjectCameraMap_v2).
  }else if(paths&&(cls=="BaseVolume"||cls=="VdbVolume"||cls=="CutoutVolume")){
   // Admitted with the assignment that binds it (admitVolume, admitMappedVolume, admitVdbVolume).
  }else if(paths&&(cls=="ShadowSet"||cls=="ShadowReceiverSet")){
   // Read by the layer's assignments (shadow linking).
  }else if(paths&&(o.isA<rdl::LightFilter>()||cls=="LightFilterSet")){
   // Read through the lights that name them and the layer's assignments (FilterAdmission).
  }else if(paths&&cls=="UserData"){
   // Read where a geometry lists it among its primitive attributes.
  }else if(paths&&cls=="Metadata"){
   // An output's EXR header attributes (aov_write.h).
  }else if(paths&&o.isA<rdl::DisplayFilter>()){
   // Run over the outputs by MoonRay's own filters (aov_display.h).
  }else throw Unsupported(o.getName()+": unsupported class "+cls);
 }
 // Portals (Scene::updateLightList): each takes its reference's copy, which then lights no shading point (it
 // leaves the light sets) but is still seen by the camera. A portal whose reference is not a light here is off.
 if(paths){std::vector<lights::Light> kept;std::map<const rdl::SceneObject*,unsigned> keptIds;std::map<unsigned,unsigned> renumber;
  // The moving lights by their final index (after the portals' renumbering below).
  auto finishMotion=[&]{plan->lightClose.assign(plan->lights.size(),nullptr);
   for(auto& pair:movingLights){auto it=lightIds.find(pair.first);if(it==lightIds.end())continue;lights::LightMotion m=pair.second;m.light=it->second;
    plan->lightMotion.push_back(m);plan->lightClose[it->second]=closeLights.at(pair.first);}};
  for(auto& pair:sorted){auto it=lightIds.find(pair.second);if(it==lightIds.end())continue;lights::Light& l=plan->lights[it->second];
   if(l.type==lights::Portal){auto ref=lightIds.find(get<rdl::SceneObject*>(*pair.second,"light"));if(ref==lightIds.end()||plan->lights[ref->second].type==lights::Portal)continue;l.refIndex=ref->second;}
   renumber[it->second]=unsigned(kept.size());keptIds[pair.second]=unsigned(kept.size());kept.push_back(l);}
  for(auto& l:kept)if(l.type==lights::Portal){l.refIndex=renumber.at(l.refIndex);kept[l.refIndex].hasPortal=1;}
  for(auto& l:kept)if(l.type==lights::Portal){auto copy=std::make_shared<lights::Light>(kept[l.refIndex]);plan->portalRefs.push_back(copy);l.ref=copy.get();}
  plan->lights=std::move(kept);lightIds=std::move(keptIds);finishMotion();
  // The lobe light sets the materials named, over the final lights.
  {const unsigned words=std::max(1u,unsigned((plan->lights.size()+31)/32));
   for(const rdl::SceneObject* ls:lobeLightSets){std::vector<unsigned> bits(words,0u);
    for(auto* light:ls->asA<rdl::LightSet>()->getLights()){auto it=lightIds.find(light);if(it!=lightIds.end())bits[it->second>>5]|=1u<<(it->second&31);}
    plan->lobeLightSetBits.insert(plan->lobeLightSetBits.end(),bits.begin(),bits.end());}}
  for(size_t k=0;k<plan->lights.size();++k)plan->lights[k].index=unsigned(k);}
 // The kit's MoonRay source switches (MoonRay Source Overrides), read from the environment as MoonRay reads them:
 // every material record carries the shading ones, a DwaToonMaterial's its smooth shadow origins.
 {auto on=[](const char* n){const char* v=std::getenv(n);return v&&v[0]=='1';};
  const unsigned global=(on("MOONRAY_MODO_TOON_TERMINATOR")?unsigned(shade::MatModoToonTerminator):0u)|
                        (on("MOONRAY_MODO_TRANSMISSION_ROUGHNESS")?unsigned(shade::MatModoTransmissionRoughness):0u);
  for(const auto& [name,index]:materialIds){auto& rec=plan->materialRecords[index];rec.flags|=global;
   const auto found=sorted.find(name);
   if(found!=sorted.end()&&found->second->getSceneClass().getName()=="DwaToonMaterial"&&on("MOONRAY_MODO_TOON_TERMINATOR"))rec.flags|=shade::MatModoToonMaterial;}
  plan->modoSwitches=(on("MOONRAY_MODO_TOON_ALBEDO")?1u:0u)|(on("MOONRAY_MODO_FOG_ALPHA")?2u:0u)|(on("MOONRAY_MODO_CRYPTO_VOLUMES")?4u:0u)|(on("MOONRAY_MODO_CAMERA_LIGHT_LPE")?8u:0u);}
 // The switches' records: copies of their materials' (a switch's switch first; a material not shaded fails the scene).
 {std::map<const rdl::SceneObject*,int> state;
  std::function<void(const rdl::SceneObject*)> fill=[&](const rdl::SceneObject* s){
   int& st=state[s];if(st==2)return;need(st==0,s->getName()+": switch materials in a cycle");st=1;
   auto sub=[&](const rdl::SceneObject* m){shade::MaterialRecord rec{};if(!m)return rec;
    need(materialIds.count(m->getName()),s->getName()+": its material "+m->getName()+" is not shaded");
    const std::string mc=m->getSceneClass().getName();if(mc=="SwitchMaterial"||mc=="RaySwitchMaterial")fill(m);
    return plan->materialRecords[materialIds.at(m->getName())];};
   shade::SwitchRecord r{};
   if(s->getSceneClass().getName()=="SwitchMaterial"){const int choice=get<rdl::Int>(*s,"choice");
    if(choice>=0&&choice<64)r.def=sub(get<rdl::SceneObject*>(*s,("material"+std::to_string(choice)).c_str()));}
   else{r.byRay=1;r.cutout=get<rdl::Bool>(*s,"cutout_camera_rays")?1:0;r.def=sub(get<rdl::SceneObject*>(*s,"default_material"));
    r.camera=sub(get<rdl::SceneObject*>(*s,"camera_ray_material"));r.mirror=sub(get<rdl::SceneObject*>(*s,"indirect_mirror_ray_material"));
    r.glossy=sub(get<rdl::SceneObject*>(*s,"indirect_glossy_ray_material"));r.diffuse=sub(get<rdl::SceneObject*>(*s,"indirect_diffuse_ray_material"));}
   shade::MaterialRecord rec{};rec.kind=shade::MatSwitch;rec.offset=unsigned(plan->materialPool.size());rec.count=shade::SwitchRecordFloats;
   for(const auto* sr:{&r.def,&r.camera,&r.mirror,&r.glossy,&r.diffuse})rec.flags|=sr->flags&shade::MatPolyVertices;
   const float* f=reinterpret_cast<const float*>(&r);plan->materialPool.insert(plan->materialPool.end(),f,f+shade::SwitchRecordFloats);
   plan->materialRecords[materialIds.at(s->getName())]=rec;st=2;};
  for(auto* s:switches)fill(s);}
 // The path tracer's caller writes the outputs it asks for: a beauty, an alpha, or both.
 need((paths||sky)&&(paths?outputCount>=1:outputCount==2&&beauty&&alpha),"One environment and beauty/alpha outputs required");
 for(auto* g:meshLightGeometry)setGeometry.erase(g);
 // MoonRay's GeometryManager renders the geometry of the scene's geometry sets (collectPrimitives), shaded by the layer:
 // a mesh of the layer in no set is not rendered.
 if(paths)for(auto it=allGeometry.begin();it!=allGeometry.end();)it=setGeometry.count(*it)?std::next(it):allGeometry.erase(it);
 need(!allGeometry.empty()&&(paths||allGeometry==setGeometry),"Geometry sets must contain exactly the rendered meshes");
 prepLaps().lap("objects (lights, volumes, sets)");
 // Each mesh's LightFilterSet (none: every filter its lights name), for the light filter tables below.
 std::vector<const rdl::LightFilterSet*> meshFilterSets;
 // The render items: each geometry of the layer, and each instance of an instancer (instancerItems).
 std::vector<RenderItem> items;
 // Shared instances' data, by reference and part (o's name and its piece's suffix): made for the first instance.
 std::map<std::string,std::shared_ptr<const MeshData>> sharedMeshes;
 // Tessellations on workers (TessellationJob), their geometry estimated while they run (at most half the budget at once).
 std::vector<std::shared_ptr<TessellationJob>> tessellations;std::shared_ptr<TessellationJob> launched;
 auto waitForRoom=[&](size_t estimate){
  for(;;){size_t running=0,bytes=0;std::shared_ptr<TessellationJob> oldest;
   for(auto& k:tessellations)if(k->result.valid()&&k->result.wait_for(std::chrono::seconds(0))!=std::future_status::ready){++running;bytes+=k->estimate;if(!oldest)oldest=k;}
   if(!oldest||(running<tessellationJobs()&&bytes+estimate<=limits.bytes/2))return;
   oldest->result.wait();}};
 for(auto& pair:sorted){const rdl::SceneObject& g=*pair.second;if(!allGeometry.count(&g))continue;
  if(g.getSceneClass().getName()=="RdlInstancerGeometry")instancerItems(g,sv,plan->motion,items);else items.push_back(RenderItem{&g});}
 for(const RenderItem& item:items){auto& o=*item.geom;
  auto* geom=o.asA<rdl::Geometry>();
  const auto pieces=[&]{PrepPart t{"geometries' pieces"};return meshPieces(o,*geom,layer);}();need(!pieces.empty(),o.getName()+": no assignment in the layer");
  for(const auto& piece:pieces){const int id=piece.id;need(id>=0,o.getName()+": no assignment in the layer");
  // Where a render item's time goes (the plan's parts): a kept mesh's and a made one's, and their lookups.
  const long long pieceBegan=nowNs();
  const std::vector<unsigned char>* faceMask=piece.faces.empty()?nullptr:&piece.faces;const std::string meshName=o.getName()+piece.suffix+item.suffix;
  auto* mat=layer.lookupMaterial(id);auto* lights=layer.lookupLightSet(id);
  // A volume shader: the mesh bounds a volume (with or without a surface of its own).
  const rdl::SceneObject* volumeShader=paths?layer.lookupVolumeShader(id):nullptr;
  need((mat&&materialIds.count(mat->getName()))||(volumeShader&&!mat),o.getName()+": missing diffuse material");need(lights,"Missing light set");
  const auto& ls=lights->getLights();
  if(paths){
   // Its light set as a bitset over the lights, shared with any mesh whose set is the same.
   const unsigned words=std::max(1u,unsigned((plan->lights.size()+31)/32));plan->lightSetWords=words;std::vector<unsigned> bits(words,0u);
   for(auto* light:ls){auto it=lightIds.find(light);if(it!=lightIds.end()&&!plan->lights[it->second].hasPortal)bits[it->second>>5]|=1u<<(it->second&31);else need(!light->get<rdl::Bool>("on")||isLight(light->getSceneClass().getName()),o.getName()+": unsupported light");}
   unsigned set=unsigned(plan->lightSetBits.size()/words);
   for(unsigned k=0;k*words<plan->lightSetBits.size();++k)if(std::equal(bits.begin(),bits.end(),plan->lightSetBits.begin()+std::ptrdiff_t(k*words))){set=k;break;}
   if(set*words==plan->lightSetBits.size())plan->lightSetBits.insert(plan->lightSetBits.end(),bits.begin(),bits.end());
   plan->meshLights.push_back(set);meshFilterSets.push_back(layer.lookupLightFilterSet(id));
   // Shadow linking (GeometryManager's ShadowLinking setter): the assignment's shadow set and shadow receiver set.
   plan->meshAssignment.push_back(id);plan->meshShadowLink.push_back(~0u);
   const rdl::ShadowSet* shadowSet=layer.lookupShadowSet(id);const rdl::ShadowReceiverSet* receiverSet=layer.lookupShadowReceiverSet(id);
   if(shadowSet||receiverSet){Plan::ShadowLink link;link.lights=unsigned(plan->shadowLinkBits.size());
    std::vector<unsigned> skip(words,0u);
    if(shadowSet)for(auto* light:shadowSet->getLights()){auto it=lightIds.find(light);if(it!=lightIds.end())skip[it->second>>5]|=1u<<(it->second&31);}
    plan->shadowLinkBits.insert(plan->shadowLinkBits.end(),skip.begin(),skip.end());
    link.receivers=unsigned(plan->shadowReceivers.size());
    if(receiverSet){link.complement=get<rdl::Bool>(*receiverSet,"complement")?1u:0u;
     for(auto* so:receiverSet->getGeometries()){const auto* rg=so?so->asA<rdl::Geometry>():nullptr;if(!rg)continue;
      const int rid=layer.getAssignmentId(rg,"");
      if(rid!=-1)plan->shadowReceivers.push_back(rid);
      else for(auto it=layer.begin(rg);it!=layer.end(rg);++it)plan->shadowReceivers.push_back(*it);}}
    link.receiverCount=unsigned(plan->shadowReceivers.size())-link.receivers;
    plan->meshShadowLink.back()=unsigned(plan->shadowLinks.size());plan->shadowLinks.push_back(link);}}
  else need(ls.size()==1&&ls[0]==sky,"Per-object light linking unsupported");
  // MoonRay displaces meshes alone (PolyMesh, OpenSubdivMesh): on curves and points a displacement is left out.
  const bool undisplaced=isCurves(o.getSceneClass().getName())||isPoints(o.getSceneClass().getName());
  const rdl::SceneObject* displacementShader=paths&&get<rdl::Bool>(sv,"enable_displacement")&&!undisplaced?layer.lookupDisplacement(id):nullptr;
  const bool subdGeometry=o.getSceneClass().getName()=="RdlMeshGeometry"&&get<rdl::Bool>(o,"is_subd");
  // Whether any part of the geometry is displaced: MoonRay then tessellates and displaces the mesh whole, each part
  // read from that one surface (its shared vertices moved by the displaced faces, its smooth normals over them all).
  bool objectDisplaced=false;
  if(paths&&get<rdl::Bool>(sv,"enable_displacement"))for(const auto& p:pieces)if(layer.lookupDisplacement(p.id))objectDisplaced=true;
  // A displaced polygon mesh beyond the triangle port (mesh_displacement.h: triangles alone, tessellated uniformly to
  // 64 at most, whole, facing as given, without polygon maps): MoonRay's own PolyMesh tessellates and displaces it.
  bool polyGeometry=false;
  if(objectDisplaced&&!subdGeometry&&o.getSceneClass().getName()=="RdlMeshGeometry"){
   const auto& fc=o.get<rdl::IntVector>("face_vertex_count");
   polyGeometry=item.instancer||std::any_of(fc.begin(),fc.end(),[](int c){return c!=3;})||faceMask||get<rdl::Float>(o,"adaptive_error")>0.0f||get<rdl::Bool>(o,"reverse_normals")||
    !(get<rdl::Float>(o,"mesh_resolution")<=64.0f)||(mat&&(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices));}
  need((paths||!layer.lookupDisplacement(id))&&(paths||!layer.lookupVolumeShader(id))&&(paths||!layer.lookupLightFilterSet(id))&&(paths||!layer.lookupShadowSet(id))&&(paths||!layer.lookupShadowReceiverSet(id)),o.getName()+": displacement/volume/filter/shadow links unsupported");
  int volumeIndex=-1;std::shared_ptr<MeshData> vdbBox;
  // A CutoutVolume: its indirect volume, held out on camera rays and read with no ray depth (none: no volume at all).
  const rdl::SceneObject* volumeInner=volumeShader;bool volumeCutout=false;
  if(volumeShader&&volumeShader->getSceneClass().getName()=="CutoutVolume"){
   {std::set<std::string> all;for(auto i=volumeShader->getSceneClass().beginAttributes();i!=volumeShader->getSceneClass().endAttributes();++i)all.insert((*i)->getName());
    attributes(*volumeShader,all);}
   volumeInner=get<rdl::SceneObject*>(*volumeShader,"indirect_volume");volumeCutout=true;
   need(!volumeInner||volumeInner->getSceneClass().getName()=="BaseVolume"||volumeInner->getSceneClass().getName()=="VdbVolume",
        volumeShader->getName()+": a cutout of a "+(volumeInner?volumeInner->getSceneClass().getName():std::string("volume"))+" is not rendered");}
  if(volumeShader){
   need(!isCurves(o.getSceneClass().getName())&&!isPoints(o.getSceneClass().getName()),o.getName()+": a volume bound to curves or points is not rendered");
   need(plan->volumes.size()<volume::MaxVolumes,"At most 32 volumes");
   volumeIndex=int(plan->volumes.size());
   if(o.getSceneClass().getName()=="VdbGeometry"){need(volumeInner,o.getName()+": a VDB geometry under a cutout of no volume");
    plan->volumes.push_back(admitVdbVolume(o,*volumeInner,plan->meshLights.back(),*plan,previous.get(),vdbBox));
    admitVdbMaps(plan->volumes.back(),*volumeInner,*plan,textureContext);}
   else if(!volumeInner){
    // A cutout of no volume: nothing (extinct, albedo and emission all zero).
    volume::Shader z{};z.kind=volume::KindBase;z.densityGrid=z.emissionGrid=z.bakedGrid=volume::NoGrid;z.program=shade::NoMapProgram;z.lightSet=plan->meshLights.back();
    for(unsigned k=0;k<volume::MappedCount;++k)z.regs[k]=shade::MapNoRegister;
    plan->volumes.push_back(z);plan->volumes.back().visibility=geometryVisibility(o)|0x300u;}
   else if(volumeInner->getSceneClass().getName()=="BaseVolume"&&volumeMapped(*volumeInner)){
    plan->volumes.push_back(admitMappedVolume(*volumeInner,plan->meshLights.back(),*plan,textureContext));plan->volumes.back().visibility=geometryVisibility(o)|0x300u;
    const auto w=affine(o).inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
    for(int i=0;i<12;++i)plan->volumes.back().worldToGrid[i]=float(rows[i]);}
   else{plan->volumes.push_back(admitVolume(*volumeInner,plan->meshLights.back()));plan->volumes.back().visibility=geometryVisibility(o)|0x300u;}
   if(volumeCutout){plan->volumes.back().cutout=1;plan->volumes.back().flags|=volume::DepthFixed;}
   plan->volumes.back().surfaceOpacityThreshold=get<rdl::Float>(*volumeShader,"surface_opacity_threshold");
   // MOONRAY_MODO_CRYPTO_VOLUMES (modoCryptoVolumesBuild): the geometry's IDs from its primitive attributes, the object's
   // from the first deep ID name (modo_object_id without one).
   {const char* sw=std::getenv("MOONRAY_MODO_CRYPTO_VOLUMES");
    if(sw&&sw[0]=='1'){
     const std::string key=plan->aovs&&!plan->aovs->cryptoIdAttribute.empty()?plan->aovs->cryptoIdAttribute:std::string("modo_object_id");
     auto read=[&](const std::string& k,float& value){
      for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)if((*i)->getName()=="primitive_attributes"){
       for(auto* u:o.get<rdl::SceneObjectVector>("primitive_attributes"))if(u&&u->getSceneClass().getName()=="UserData"&&get<rdl::String>(*u,"float_key")==k){
        const auto& f=get<rdl::FloatVector>(*u,"float_values_0");if(!f.empty()){value=f[0];return true;}}}
      return false;};
     auto& vs=plan->volumes.back();vs.cryptoValid=read(key,vs.cryptoId)?1u:0u;read("modo_material_id",vs.cryptoMaterialId);read("modo_asset_id",vs.cryptoAssetId);}}
   plan->volumes.back().lpeLabel=plan->aovs?plan->aovs->lpeLabel(aov::attribute<rdl::String>(*volumeShader,"label","")):-1;
  }else need(o.getSceneClass().getName()!="VdbGeometry",o.getName()+": a VDB geometry without a volume shader");
  // A displacement (mesh_displacement.h): MoonRay tessellates the mesh uniformly to its mesh resolution (adaptive
  // tessellation needs the camera's frustum, not ported) and displaces it.
  int edgeVertices=0;displace::Shader displacement;
  if(displacementShader&&!polyGeometry){
   need(subdGeometry||get<rdl::Float>(o,"adaptive_error")<=0.0f,o.getName()+": adaptive tessellation is not rendered yet");
   const float resolution=get<rdl::Float>(o,"mesh_resolution");need(std::isfinite(resolution)&&resolution<=64.0f,o.getName()+": a mesh resolution above 64");
   edgeVertices=std::max(0,std::max(1,int(resolution))-1);
   displacement=admitDisplacement(*displacementShader,plan->materialPool,textureContext);
   // MoonRay tessellates a displaced mesh into quads: polygon maps would read four polyvertices.
   need(!mat||!(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices),o.getName()+": polygon maps on a displaced mesh are not rendered yet");
  }
  // A static instance of a polygon or subdivision mesh shares its reference's data: one copy in the reference's own
  // space for every instance of it, placed by each instance's transform as MoonRay's instances are (its hits through
  // IntersectionInit's transforms), with the instance's attributes its own. Baked into a mesh of its own as before: an
  // instance that moves, bounds a volume, is single-sided, is the bake camera's mesh, or whose triangles XPU2 displaces.
  const std::string sharedKey=o.getName()+piece.suffix;
  const bool shareInstance=paths&&item.instancer&&!item.moves&&o.getSceneClass().getName()=="RdlMeshGeometry"&&volumeIndex<0&&mat&&
   !meshHasMotion(o)&&&o!=bakeGeometry&&get<rdl::Int>(o,"side_type")!=1&&!(displacementShader&&!subdGeometry&&!polyGeometry)&&sharedInstances();
  const long long lookupBegan=nowNs();
  auto sig=signature(o,frame);std::shared_ptr<const MeshData> data;bool reusedData=false,sharedHit=false;launched.reset();
  // On a worker (TessellationJob): a still, single-part mesh of its own, no volume's bound nor the bake camera's.
  const bool onWorker=paths&&!item.instancer&&!item.moves&&!meshHasMotion(o)&&pieces.size()==1&&volumeIndex<0&&&o!=bakeGeometry&&
   o.getSceneClass().getName()=="RdlMeshGeometry"&&tessellationJobs()>1;
  if(shareInstance){auto it=sharedMeshes.find(sharedKey);if(it!=sharedMeshes.end()){data=it->second;sharedHit=true;}}
  // Adaptive tessellation (a subdivision surface, or a displaced polygon mesh MoonRay's PolyMesh tessellates) follows the
  // dicing camera: the render camera's move tessellates it again. A dicing camera of its own (the scene's, or the mesh's;
  // a realtime preview holds one while the view moves little) is a part of what shapes it instead: the render camera
  // moves with the camera-only fast path, and the mesh is kept while that dicing camera is the same (its signature).
  const bool adaptiveMesh=o.getSceneClass().getName()=="RdlMeshGeometry"&&(get<rdl::Bool>(o,"is_subd")||polyGeometry)&&get<rdl::Float>(o,"adaptive_error")>0.0f;
  const rdl::SceneObject* dicingCamera=get<rdl::SceneObject*>(sv,"dicing_camera");if(!dicingCamera)dicingCamera=get<rdl::SceneObject*>(o,"dicing_camera");
  if(adaptiveMesh&&!dicingCamera)plan->cameraDependent=true;
  if(adaptiveMesh&&dicingCamera)for(auto& b:deepSignature(*dicingCamera,frame))sig.push_back(b);
  // A displaced mesh MoonRay tessellates (subdivision, PolyMesh) is kept while nothing that shapes it has changed: the
  // mesh, each part's displacement and its maps through and through (deepSignature), which displacement is whose, and
  // whether polyvertices are made. A camera, a light or another material changed, it is not tessellated again (its
  // vertices are kept in world space). Triangles XPU2 displaces itself (mesh_displacement.h) are kept the same way, by
  // their tessellation's edge vertices besides.
  std::string displacementKey;
  if(objectDisplaced&&(subdGeometry||polyGeometry||displacementShader)){
   for(const auto& p:pieces)if(const rdl::SceneObject* ds=layer.lookupDisplacement(p.id)){
    displacementKey+=std::to_string(p.id)+":"+ds->getName()+";";
    for(auto& b:deepSignature(*ds,frame))sig.push_back(b);}
   const bool polys=(plan->aovs&&plan->aovs->wireframe)||(mat&&(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices));
   displacementKey+=polys?"polyvertices":"";
   if(!subdGeometry&&!polyGeometry)displacementKey+="triangles:"+std::to_string(edgeVertices);}
  // The fur file can change under the same name: fur is read again each time.
  const bool curveGeometry=isCurves(o.getSceneClass().getName());
  // A shared instance's data too (in its own space, its instances' attributes and places their own): the 4 M triangles of
  // a car's four tyres were made again for any edit of anything (3.3 s), and every instance's geometry after them.
  if(!data&&previous&&previous->domain==domain&&(!item.instancer||shareInstance)&&o.getSceneClass().getName()!="ModoFurGeometry"&&(!objectDisplaced||!displacementKey.empty())&&&o!=bakeGeometry&&
   !(adaptiveMesh&&!dicingCamera))for(auto& old:previous->meshes)if(old.data->name==meshName&&old.data->signature==sig&&(!shareInstance||old.data->instanced)&&old.data->displacementKey==displacementKey&&
   !old.data->coarse&&old.data->polyVertices.empty()==!(plan->aovs&&plan->aovs->wireframe)&&((o.getSceneClass().getName()!="RdlMeshGeometry"&&o.getSceneClass().getName()!="RdlCurveGeometry"&&!isPoints(o.getSceneClass().getName()))||!meshHasMotion(o)||previous->motion==plan->motion)){data=old.data;++plan->reusedMeshes;reusedData=true;break;}
  // A plain mesh whose node_xform alone changed since it was made (MeshData::shape): its data taken over, placed anew as
  // an instance from where it was made to where it is (below, once its plan entry is made). A car turned whole, then a
  // mesh edited, made every mesh again (2.2 s).
  bool placedAnew=false;rdl::Mat4d placedFrom(scene_rdl2::math::one);
  if(!data&&previous&&previous->domain==domain&&paths&&!item.instancer&&!objectDisplaced&&!volumeShader&&&o!=bakeGeometry&&
     o.getSceneClass().getName()=="RdlMeshGeometry"&&!get<rdl::Bool>(o,"is_subd")&&!meshHasMotion(o)){
   const auto shape=shapeSignature(o,frame);
   for(auto& old:previous->meshes)if(old.data->placeable&&old.data->name==meshName&&old.data->shape==shape&&old.data->steps.empty()&&!old.data->coarse&&
    old.data->displacementKey==displacementKey&&old.data->polyVertices.empty()==!(plan->aovs&&plan->aovs->wireframe)){
    data=old.data;++plan->reusedMeshes;reusedData=true;placedAnew=true;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)placedFrom[i][j]=old.data->placedBy[4*i+j];break;}}
  prepLaps().add("meshes' signatures and lookups",double(nowNs()-lookupBegan)*1e-9);
  if(!data&&vdbBox){
   // A VDB volume's boundary box (admitVdbVolume), named as the mesh.
   vdbBox->name=meshName;vdbBox->signature=std::move(sig);data=std::move(vdbBox);++plan->preparedMeshes;}
  if(!data&&isPoints(o.getSceneClass().getName())){
   // Points (RdlPointProcedural): their centres through the node transform (and over the shutter), radii as given.
   auto m=std::make_shared<MeshData>();m->name=meshName;m->signature=std::move(sig);
   if(item.instancer){
    const rdl::Mat4d x=instanceMatrix(item);admitPoints(o,plan->motion,*m,faceMask,&x,&item);instanceAttributes(*item.instancer,item.instance,item.count,*m);m->instanced=true;}
   else admitPoints(o,plan->motion,*m,faceMask);
   data=std::move(m);++plan->preparedMeshes;}
  if(!data&&curveGeometry){auto m=std::make_shared<MeshData>();m->name=meshName;m->signature=std::move(sig);
   // An instance's curves (a shared primitive): through the instance's transform.
   const auto transform=item.instancer?instanceMatrix(item):affine(o);
   {const auto w=transform.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
    for(int i=0;i<12;++i)m->worldToObject[i]=float(rows[i]);}
   admitCurves(o,transform,*m,&plan->motion,item.instancer?&item:nullptr);
   if(item.instancer){instanceAttributes(*item.instancer,item.instance,item.count,*m);m->instanced=true;}
   data=std::move(m);++plan->preparedMeshes;}
  // The tessellation cache (geometryKey): a displaced mesh MoonRay tessellates, as it was prepared before (another
  // session, the scene opened again). Not one that moves, an instance with data of its own, the bake camera's mesh, a
  // volume's bound or one with a dicing camera of its own.
  std::string cacheKey;bool cacheHit=false;
  if(!data&&paths&&geometryCacheOn()&&!coarseTessellation()&&objectDisplaced&&(subdGeometry||polyGeometry)&&o.getSceneClass().getName()=="RdlMeshGeometry"&&
     (!item.instancer||shareInstance)&&!item.moves&&!meshHasMotion(o)&&&o!=bakeGeometry&&volumeIndex<0&&
     !get<rdl::SceneObject*>(sv,"dicing_camera")&&!get<rdl::SceneObject*>(o,"dicing_camera")){
   cacheKey=geometryKey(o,sig,displacementKey,*plan,shareInstance,sv);
   if(auto m=loadGeometry(cacheKey)){m->name=meshName;m->signature=sig;m->displacementKey=displacementKey;if(shareInstance)m->instanced=true;
    data=std::move(m);cacheHit=true;++plan->preparedMeshes;}}
  // All geometry attribute/binding identities match before validation is reused.
  // Material/light assignments are checked above for EVERY epoch.
  if(!data){
  if(paths){std::set<std::string> allowed{"vertex_list_0","face_vertex_count","vertices_by_index","is_subd","node_xform","smooth_normal","normal_list","uv_list","primitive_attributes","mesh_resolution","adaptive_error",
   "part_list","part_face_count_list","part_face_indices","side_type","reverse_normals",
   "subd_scheme","subd_boundary","subd_fvar_linear","dicing_camera","subd_crease_indices","subd_crease_sharpnesses","subd_corner_indices","subd_corner_sharpnesses",
   // Motion blur (meshMotionSteps).
   "vertex_list_1","velocity_list_0","velocity_list_1","accleration_list","velocity_scale","motion_blur_type","curved_motion_blur_sample_count"};
   allowed.insert(geometryVisibilityAttributes().begin(),geometryVisibilityAttributes().end());attributes(o,allowed,{},{"node_xform"});}
  else attributes(o,{"vertex_list_0","face_vertex_count","vertices_by_index","is_subd","node_xform","smooth_normal"});
  const bool meshMotion=paths&&o.getSceneClass().getName()=="RdlMeshGeometry"&&(meshHasMotion(o)||item.moves);
  need(paths||!item.instancer||!displacementShader||!get<rdl::Bool>(sv,"enable_displacement"),o.getName()+": a displaced instance is not rendered yet");
  need(!item.instancer||!(subdGeometry||polyGeometry)||!meshHasMotion(o),o.getName()+": an instanced tessellated mesh of its own motion is not rendered yet");
  need(!meshMotion||&o!=bakeGeometry,o.getName()+": a baked mesh in motion is not rendered yet");
  // A displaced mesh MoonRay is about to tessellate uniformly (subdivision or PolyMesh): its size told from its faces and
  // mesh resolution first (about r x r times its triangles, 18 bytes a triangle with its share of vertices), so one the
  // machine cannot hold is refused now, not after its tessellation (an 8 M-triangle tyre took 30 s to be refused).
  if(paths&&objectDisplaced&&(subdGeometry||polyGeometry)&&get<rdl::Float>(o,"adaptive_error")<=0.0f){
   const auto& fc=o.get<rdl::IntVector>("face_vertex_count");size_t control=0;for(int n:fc)control+=n>=3?size_t(n-2):0;
   const double r=std::max(1.0,double(get<rdl::Float>(o,"mesh_resolution")));
   const size_t estimate=size_t(double(control)*r*r*18.0);
   need(plan->bytes+estimate<=limits.bytes+limits.bytes/4,geometryBudgetText(plan->bytes+estimate,limits.bytes)+" (estimated before tessellating "+o.getName()+")");}
  if(paths&&get<rdl::Bool>(o,"is_subd")){
   // A subdivision surface: MoonRay's own tessellation (tessellateSubdivision), once for all its parts.
   // A displaced subdivision surface: MoonRay's displaceMesh in the tessellation, each vertex's value from its
   // assignment's displacement (subdDisplace), the textures its maps read decoded now.
   SubdDisplacement subdDisplacement;SubdDisplaceCall displaceCall;
   if(objectDisplaced){
    for(const auto& p:meshPieces(o,*geom,layer)){const rdl::SceneObject* ds=layer.lookupDisplacement(p.id);
     if(ds&&!subdDisplacement.shaders.count(ds))subdDisplacement.shaders.emplace(ds,admitDisplacement(*ds,plan->materialPool,textureContext));}
    for(auto& f:plan->textures){if(!f->pyramid)f->pyramid=std::make_shared<texture::Pyramid>(f->name.string(),f->alpha);subdDisplacement.images.push_back(&f->pyramid->image);}
    subdDisplacement.pool=plan->materialPool.data();subdDisplacement.worldToRender=plan->worldToRender;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)subdDisplacement.w2r[i][j]=plan->world2render[4*i+j];
    subdDisplacement.r2w=subdDisplacement.w2r.inverse();
    {const auto w=affine(o).inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
     for(int i=0;i<12;++i)subdDisplacement.worldToObject[i]=float(rows[i]);}
    displaceCall.fn=&subdDisplace;displaceCall.user=&subdDisplacement;displaceCall.with=&subdDisplaceWith;}
   const auto transform=affine(o);
   const bool polys=(plan->aovs&&plan->aovs->wireframe)||(mat&&(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices));
   if(item.instancer){
    // An instance: the prototype tessellated once in its own space, moved by the instance's transform.
    auto& surface=subdRefCache[&o];
    if(!surface){surface=tessellateSubdivision(o,layer,*plan,rdl::Mat4d(scene_rdl2::math::one),nullptr,displaceCall,true);subdDisplacement.check(o);}
    auto m=subdivisionMesh(*surface,faceMask,meshName,std::move(sig),rdl::Mat4d(scene_rdl2::math::one),polys);
    if(shareInstance)m->instanced=true;  // in its own space, placed by each instance (placeInstance)
    else{instanceSurface(*m,item,plan->motion);instanceAttributes(*item.instancer,item.instance,item.count,*m);}
    data=std::move(m);++plan->preparedMeshes;
   }else if(meshMotion){
    // Under motion blur (SubdivisionMesh::transformPrimitive's transformVertexBuffer on the control cage): the cage at
    // each time step, each tessellated; the steps' tessellations have one topology (uniform tessellation).
    auto& surfaces=subdMotionCache[&o];
    if(surfaces.empty()){
     const auto steps=meshMotionSteps(o,plan->motion,o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN),o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END));
     for(const auto& step:steps){surfaces.push_back(tessellateSubdivision(o,layer,*plan,transform,&step,displaceCall));subdDisplacement.check(o);}
     for(size_t k=1;k<surfaces.size();++k)need(surfaces[k]->quads==surfaces[0]->quads&&surfaces[k]->positions.size()==surfaces[0]->positions.size(),
      o.getName()+": a subdivision surface whose tessellation changes over the shutter is not rendered yet");}
    auto m=subdivisionMesh(*surfaces[0],faceMask,meshName,std::move(sig),transform,polys);
    for(size_t k=1;k<surfaces.size();++k){m->steps.push_back(surfaces[k]->positions);m->stepNormals.push_back(surfaces[k]->normals);
     m->stepDpds.push_back(surfaces[k]->dpds);m->stepDpdt.push_back(surfaces[k]->dpdt);}
    m->subdSteps=surfaces.size()>1;
    data=std::move(m);++plan->preparedMeshes;
   }else if(onWorker&&!subdCache.count(&o)){
    // On a worker: its own displacement (the material pool copied), the mesh made there too.
    auto job=std::make_shared<TessellationJob>();job->displacement=copyDisplacement(subdDisplacement);
    if(objectDisplaced){job->displacement->ownPool.assign(plan->materialPool.begin(),plan->materialPool.end());job->displacement->pool=job->displacement->ownPool.data();
     displaceCall.user=job->displacement.get();}
    {const auto& fc=o.get<rdl::IntVector>("face_vertex_count");size_t control=0;for(int n:fc)control+=n>=3?size_t(n-2):0;
     const double r=std::max(1.0,double(get<rdl::Float>(o,"mesh_resolution")));job->estimate=size_t(double(control)*r*r*18.0);}
    waitForRoom(job->estimate);
    const rdl::SceneObject* object=&o;const rdl::Layer* layerOf=&layer;const Plan* planOf=plan.get();auto keep=job->displacement;
    const std::vector<unsigned char> mask=faceMask?*faceMask:std::vector<unsigned char>();const bool masked=faceMask!=nullptr;
    job->result=std::async(std::launch::async,[object,layerOf,planOf,keep,displaceCall,transform,mask,masked,meshName,sig,polys]()->std::shared_ptr<const MeshData>{
     auto surface=tessellateSubdivision(*object,*layerOf,*planOf,transform,nullptr,displaceCall);keep->check(*object);
     return subdivisionMesh(*surface,masked?&mask:nullptr,meshName,sig,transform,polys);});
    data=std::make_shared<MeshData>();launched=job;++plan->preparedMeshes;
   }else{
   auto& surface=subdCache[&o];if(!surface){surface=tessellateSubdivision(o,layer,*plan,transform,nullptr,displaceCall);subdDisplacement.check(o);}
   data=subdivisionMesh(*surface,faceMask,meshName,std::move(sig),transform,polys);++plan->preparedMeshes;}
  }else if(polyGeometry){
   // A displaced polygon mesh: MoonRay's PolyMesh (tessellatePolygons), once for all its parts and time steps, each
   // vertex's displacement from its assignment's (subdDisplace), the textures its maps read decoded now.
   SubdDisplacement polyDisplacement;SubdDisplaceCall displaceCall;
   for(const auto& p:meshPieces(o,*geom,layer)){const rdl::SceneObject* ds=layer.lookupDisplacement(p.id);
    if(ds&&!polyDisplacement.shaders.count(ds))polyDisplacement.shaders.emplace(ds,admitDisplacement(*ds,plan->materialPool,textureContext));}
   {PrepPart t{"texture pyramids (for displacement)"};
   for(auto& f:plan->textures){if(!f->pyramid)f->pyramid=std::make_shared<texture::Pyramid>(f->name.string(),f->alpha);polyDisplacement.images.push_back(&f->pyramid->image);}}
   polyDisplacement.pool=plan->materialPool.data();polyDisplacement.worldToRender=plan->worldToRender;
   for(int i=0;i<4;++i)for(int j=0;j<4;++j)polyDisplacement.w2r[i][j]=plan->world2render[4*i+j];
   polyDisplacement.r2w=polyDisplacement.w2r.inverse();
   {const auto w=affine(o).inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
    for(int i=0;i<12;++i)polyDisplacement.worldToObject[i]=float(rows[i]);}
   displaceCall.fn=&subdDisplace;displaceCall.user=&polyDisplacement;displaceCall.with=&subdDisplaceWith;
   const auto transform=affine(o);
   const bool polys=(plan->aovs&&plan->aovs->wireframe)||(mat&&(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices));
   if(onWorker&&!meshMotion&&!polyCache.count(&o)){
    // On a worker: its own displacement (the material pool copied), the mesh made there too.
    auto job=std::make_shared<TessellationJob>();job->displacement=copyDisplacement(polyDisplacement);
    job->displacement->ownPool.assign(plan->materialPool.begin(),plan->materialPool.end());job->displacement->pool=job->displacement->ownPool.data();
    displaceCall.user=job->displacement.get();
    {const auto& fc=o.get<rdl::IntVector>("face_vertex_count");size_t control=0;for(int n:fc)control+=n>=3?size_t(n-2):0;
     const double r=std::max(1.0,double(get<rdl::Float>(o,"mesh_resolution")));job->estimate=size_t(double(control)*r*r*18.0);}
    waitForRoom(job->estimate);
    const rdl::SceneObject* object=&o;const rdl::Layer* layerOf=&layer;const Plan* planOf=plan.get();auto keep=job->displacement;
    const std::vector<unsigned char> mask=faceMask?*faceMask:std::vector<unsigned char>();const bool masked=faceMask!=nullptr;
    job->result=std::async(std::launch::async,[object,layerOf,planOf,keep,displaceCall,transform,mask,masked,meshName,sig,polys]()->std::shared_ptr<const MeshData>{
     auto surface=tessellatePolygons(*object,*layerOf,*planOf,transform,nullptr,displaceCall);keep->check(*object);
     return polygonMesh(*surface,masked?&mask:nullptr,meshName,sig,transform,polys);});
    data=std::make_shared<MeshData>();launched=job;++plan->preparedMeshes;
   }else{
   auto& surface=item.instancer?polyRefCache[&o]:polyCache[&o];
   if(!surface&&item.instancer){surface=tessellatePolygons(o,layer,*plan,rdl::Mat4d(scene_rdl2::math::one),nullptr,displaceCall,true);polyDisplacement.check(o);}
   if(!surface){std::vector<std::vector<V3>> steps;
    if(meshMotion)steps=meshMotionSteps(o,plan->motion,o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN),o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END));
    surface=tessellatePolygons(o,layer,*plan,transform,steps.empty()?nullptr:&steps,displaceCall);polyDisplacement.check(o);}
   if(item.instancer){
    // An instance: the prototype tessellated and displaced once in its own space, moved by the instance's transform.
    auto m=polygonMesh(*surface,faceMask,meshName,std::move(sig),rdl::Mat4d(scene_rdl2::math::one),polys);
    if(shareInstance)m->instanced=true;  // in its own space, placed by each instance (placeInstance)
    else{instanceSurface(*m,item,plan->motion);instanceAttributes(*item.instancer,item.instance,item.count,*m);}
    data=std::move(m);}
   else{PrepPart t{"polygon mesh from the tessellation"};data=polygonMesh(*surface,faceMask,meshName,std::move(sig),transform,polys);}
   ++plan->preparedMeshes;}
  }else{
  need(!get<rdl::Bool>(o,"is_subd"),o.getName()+": subdivision unsupported");
  // Supplied normals are the shading normals; on the path tracer a mesh without them smooths its own (PolyMeshCalcNv).
  if(spatial)need(!get<rdl::Bool>(o,"smooth_normal")||paths,"Spatial smooth normals unsupported");
  auto transform=shareInstance?rdl::Mat4d(scene_rdl2::math::one):item.instancer?instanceMatrix(item):spatial?affine(o):translation(o);if(!spatial)need(transform[3][2]==0,o.getName()+": only coplanar geometry supported");
   need(paths||!faceMask,o.getName()+": parts are rendered by the path tracer only");
   auto m=std::make_shared<MeshData>();m->name=meshName;m->signature=std::move(sig);
   {const auto w=transform.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
    for(int i=0;i<12;++i)m->worldToObject[i]=float(rows[i]);}
   // Where it is made (MeshData::placeable): a plain spatial mesh of its own, still and not displaced.
   if(paths&&spatial&&!shareInstance&&!item.instancer&&!objectDisplaced&&!volumeShader&&!meshHasMotion(o)){
    m->placeable=true;m->shape=shapeSignature(o,frame);for(int i=0;i<4;++i)for(int j=0;j<4;++j)m->placedBy[4*i+j]=transform[i][j];}
   const auto& v=o.get<rdl::Vec3fVector>("vertex_list");const auto& counts=o.get<rdl::IntVector>("face_vertex_count");const auto& indices=o.get<rdl::IntVector>("vertices_by_index");need(!v.empty()&&!counts.empty(),"Empty mesh unsupported");
   m->positions.reserve(v.size());m->triangles.reserve(counts.size());
   for(auto p:v){need(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&(spatial||p.z==0),"Invalid spatial/planar vertex");
    V3 point{float(p.x*transform[0][0]+p.y*transform[1][0]+p.z*transform[2][0]+transform[3][0]),float(p.x*transform[0][1]+p.y*transform[1][1]+p.z*transform[2][1]+transform[3][1]),float(p.x*transform[0][2]+p.y*transform[1][2]+p.z*transform[2][2]+transform[3][2])};
    need(surface::finite(point)&&(!spatial||(std::abs(point.x)<=10000&&std::abs(point.y)<=10000&&std::abs(point.z)<=10000)),"Transformed vertex bound");m->positions.push_back(point);}
   if(meshMotion){
    // Motion blur: the vertices at the shutter's time steps (the first, the open's, in place of the positions).
    auto steps=item.instancer?instanceMotionSteps(o,plan->motion,item):
                 meshMotionSteps(o,plan->motion,o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_BEGIN),o.get<rdl::Mat4d>("node_xform",rdl::TIMESTEP_END));
    m->positions=std::move(steps[0]);for(size_t k=1;k<steps.size();++k)m->steps.push_back(std::move(steps[k]));}
   prepLaps().add("made: positions",double(nowNs()-pieceBegan)*1e-9);
   PolygonSplit split;
   const bool reversed=paths&&get<rdl::Bool>(o,"reverse_normals");
   if(paths){PrepPart t{"made: split"};split=splitPolygons(v,counts,indices,o.getName());
    // MoonRay's QuadMesh hands polygon maps four polyvertices; a displaced mesh is tessellated from its faces.
    need(!displacementShader||(!split.quads&&split.faceOf.size()==counts.size()),o.getName()+": a displacement on a mesh with faces of more than three vertices is not rendered yet");
    need(!displacementShader||!faceMask,o.getName()+": a displacement on a mesh with parts is not rendered yet");}
   else for(size_t i=0;i<counts.size();++i){need(counts[i]==3,"Non-triangle mesh");split.corners.insert(split.corners.end(),{unsigned(i*3),unsigned(i*3+1),unsigned(i*3+2)});split.faceOf.push_back(unsigned(i));}
   if(reversed){
    // Its smooth normals are made from the faces as given (initAttributesAndDisplace), then negated with the rest.
    PolygonSplit given=split;reverseSplit(split,counts,indices);split.quadVertices=given.quadVertices;
    need(!displacementShader,o.getName()+": a displaced mesh with reversed normals falls back");}
   const long long keptBegan=nowNs();
   std::vector<unsigned> kept;  // the split's triangles kept, in order
   for(size_t i=0;i<split.faceOf.size();++i){if(faceMask&&!(*faceMask)[split.faceOf[i]])continue;unsigned ids[3];for(int j=0;j<3;++j){auto k=paths?split.vertices[i*3+size_t(j)]:indices[split.corners[i*3+size_t(j)]];need(k>=0&&size_t(k)<v.size(),"Vertex index out of bounds");ids[j]=unsigned(k);}
    auto a=m->positions[ids[0]],b=m->positions[ids[1]],c=m->positions[ids[2]];double area=(double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x);
    if(spatial){V3 n;const bool some=surface::normalize(surface::cross(surface::add(b,surface::mul(a,-1)),surface::add(c,surface::mul(a,-1))),n);
     if(!some&&paths)continue;need(some,"Degenerate triangle");}
    else need(std::isfinite(area)&&area>0,"Degenerate/reversed triangle unsupported");m->triangles.push_back({ids[0],ids[1],ids[2]});kept.push_back(unsigned(i));}
   need(!m->triangles.empty(),o.getName()+": no triangle of any area");
   prepLaps().add("made: triangles kept",double(nowNs()-keptBegan)*1e-9);const long long attributesBegan=nowNs();
   // MoonRay's polyvertices, where the wireframe AOV or a polygon map reads them.
   if(paths&&((plan->aovs&&plan->aovs->wireframe)||(mat&&(plan->materialRecords.at(materialIds.at(mat->getName())).flags&shade::MatPolyVertices)))){
    m->polyVertices.reserve(kept.size()*4);
    for(unsigned t:kept){
     if(split.quads){const size_t f=split.faceOf[t];for(int j=0;j<4;++j)m->polyVertices.push_back(unsigned(split.quadVertices[f*4+size_t(j)]));}
     else{for(int j=0;j<3;++j)m->polyVertices.push_back(unsigned(split.vertices[size_t(t)*3+size_t(j)]));m->polyVertices.push_back(~0u);}}}
   // Face-varying values a kept triangle's corner, uniform ones its face's (QuadMesh/TriMesh splitNGons).
   auto byCorner=[&](const std::vector<float>& d,unsigned n){std::vector<float> r(kept.size()*3*n);float* w=r.data();
    for(unsigned t:kept)for(int j=0;j<3;++j){const float* from=d.data()+split.corners[size_t(t)*3+size_t(j)]*n;for(unsigned k=0;k<n;++k)*w++=from[k];}return r;};
   auto byFace=[&](const std::vector<float>& d,unsigned n){std::vector<float> r(kept.size()*n);float* w=r.data();
    for(unsigned t:kept){const float* from=d.data()+size_t(split.faceOf[t])*n;for(unsigned k=0;k<n;++k)*w++=from[k];}return r;};
   if(paths){
    // pickRate: face-varying when there is one value a face-vertex, else one a vertex.
    auto rate=[&](size_t n)->unsigned{if(n==0)return shade::RateNone;if(n==indices.size())return shade::RateFaceVarying;if(n==v.size())return shade::RateVertex;throw Unsupported(o.getName()+": unsupported attribute rate");};
    const auto& nl=o.get<rdl::Vec3fVector>("normal_list");m->normalRate=rate(nl.size());
    // No normals and smooth_normal: each vertex's the normalized sum of its triangles' (PolyMeshCalcNv).
    // A QuadMesh sums each quad's normal (QuadMeshCalcNv), a TriMesh each triangle's.
    if(nl.empty()&&get<rdl::Bool>(o,"smooth_normal")){
     if(split.quads)m->normals=displace::smoothNormals(m->positions,split.quadVertices,4);
     // Over the whole mesh, whatever its parts (MoonRay's normals are the mesh's own).
     else m->normals=displace::smoothNormals(m->positions,split.vertices,3);
     // A moving mesh's second normal sample, from its second time step.
     if(!m->steps.empty())m->normals1=split.quads?displace::smoothNormals(m->steps[0],split.quadVertices,4):displace::smoothNormals(m->steps[0],split.vertices,3);
     m->normalRate=shade::RateVertex;}
    if(!nl.empty()){
     // Normals transform by the inverse transpose of the node transform (row vectors: n * inverse(M)^T).
     double M[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)M[i][j]=transform[i][j];
     double det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
     double inv[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,d=(i+2)%3;inv[i][j]=(M[a][cc]*M[b][d]-M[a][d]*M[b][cc])/det;}
     m->normals.reserve(nl.size());
     // X2_NEED: a message made for every normal, in case, was most of a dense mesh's time.
     for(auto n:nl){double x=n.x*inv[0][0]+n.y*inv[0][1]+n.z*inv[0][2],y=n.x*inv[1][0]+n.y*inv[1][1]+n.z*inv[1][2],z=n.x*inv[2][0]+n.y*inv[2][1]+n.z*inv[2][2];
      double l=std::sqrt(x*x+y*y+z*z);X2_NEED(std::isfinite(l)&&l>0,o.getName()+": invalid normal");m->normals.push_back({float(x/l),float(y/l),float(z/l)});}
     if(m->normalRate==shade::RateFaceVarying){std::vector<V3> f(kept.size()*3);V3* w=f.data();
      for(unsigned t:kept)for(int j=0;j<3;++j)*w++=m->normals[split.corners[size_t(t)*3+size_t(j)]];
      m->normals=std::move(f);}
    }
    // primitive_attributes: UserData of floats, colours, vectors and ints (geom/PrimitiveUserData processArbitraryData).
    for(auto* object:o.get<rdl::SceneObjectVector>("primitive_attributes")){
     need(object&&object->getSceneClass().getName()=="UserData",o.getName()+": a primitive attribute that is not UserData");
     attributes(*object,{"bool_key","bool_values","int_key","int_values","float_key","float_values_0","float_values_1","string_key","string_values","color_key","color_values_0","color_values_1","vec2f_key","vec2f_values_0","vec2f_values_1","vec3f_key","vec3f_values_0","vec3f_values_1","mat3f_key","mat3f_values_0","mat3f_values_1","mat4f_key","mat4f_values_0","mat4f_values_1","rate"});
     const int explicitRate=get<rdl::Int>(*object,"rate");
     // pickRate: the explicit rate when its count fits, else guessed from the count (auto).
     auto pick=[&](size_t n)->unsigned{const size_t fv=indices.size(),uni=counts.size(),vert=v.size();
      switch(explicitRate){case 1:return n==1?1u:0u;case 3:return n==uni?3u:0u;case 4:return n==vert?4u:0u;case 5:return n==vert?5u:0u;case 6:return n==fv?6u:0u;case 0:break;default:return 0u;}
      if(n==0)return 0u;if(n==1)return 1u;if(n==fv)return 6u;if(n==uni)return 3u;if(n==vert)return 4u;return 0u;};
     auto add=[&](const std::string& key,unsigned type,unsigned components,const std::vector<float>& values,size_t count){
      if(key.empty())return;const unsigned r=pick(count);need(r!=0,o.getName()+": primitive attribute "+key+" has an unsupported rate or size");
      for(float f:values)X2_NEED(std::isfinite(f),o.getName()+": invalid primitive attribute "+key);
      const std::vector<float> expanded=r==6?byCorner(values,components):r==3?byFace(values,components):values;
      m->prims.push_back(shade::PrimAttr{shade::primKey(key.c_str(),type),components,r,unsigned(m->primData.size())});m->primData.insert(m->primData.end(),expanded.begin(),expanded.end());};
     {const auto& d=get<rdl::FloatVector>(*object,"float_values_0");add(get<rdl::String>(*object,"float_key"),shade::PrimFloat,1,std::vector<float>(d.begin(),d.end()),d.size());}
     {const auto& d=get<rdl::IntVector>(*object,"int_values");std::vector<float> f;for(int i:d)f.push_back(float(i));add(get<rdl::String>(*object,"int_key"),shade::PrimInt,1,f,d.size());}
     {const auto& d=get<rdl::RgbVector>(*object,"color_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.r);f.push_back(c.g);f.push_back(c.b);}add(get<rdl::String>(*object,"color_key"),shade::PrimRgb,3,f,d.size());}
     {const auto& d=get<rdl::Vec2fVector>(*object,"vec2f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);}add(get<rdl::String>(*object,"vec2f_key"),shade::PrimVec2f,2,f,d.size());}
     {const auto& d=get<rdl::Vec3fVector>(*object,"vec3f_values_0");std::vector<float> f;for(auto c:d){f.push_back(c.x);f.push_back(c.y);f.push_back(c.z);}add(get<rdl::String>(*object,"vec3f_key"),shade::PrimVec3f,3,f,d.size());}
    }
    if(item.instancer){if(!shareInstance)instanceAttributes(*item.instancer,item.instance,item.count,*m);m->instanced=true;}
    if(item.instancer&&item.moves&&!m->steps.empty()){m->stepShading=true;
     if(!m->normals1.empty())for(const auto& step:m->steps)m->stepNormals.push_back(split.quads?displace::smoothNormals(step,split.quadVertices,4):displace::smoothNormals(step,split.vertices,3));}
    if(reversed){for(auto& n:m->normals)n=surface::mul(n,-1.0f);for(auto& n:m->normals1)n=surface::mul(n,-1.0f);
     for(auto& s:m->stepNormals)for(auto& n:s)n=surface::mul(n,-1.0f);}
    const auto& ul=o.get<rdl::Vec2fVector>("uv_list");m->uvRate=rate(ul.size());
    m->uvs.reserve(ul.size()*2);for(auto u:ul){X2_NEED(std::isfinite(u.x)&&std::isfinite(u.y),o.getName()+": invalid uv");m->uvs.push_back(u.x);m->uvs.push_back(u.y);}
    if(m->uvRate==shade::RateFaceVarying)m->uvs=byCorner(m->uvs,2);
    // No uv_list (its "surface_st"): the "st" or else the "uv" primitive attribute, or else the mesh's default, a
    // QuadMesh's each quad's corners (0, 0), (1, 0), (1, 1), (0, 1) (getQuadST), a TriMesh's (0, 0), (1, 0), (1, 1)
    // a triangle (getTriangleST, hitState's). Made a kept triangle's corners.
    if(ul.empty()){
     const shade::PrimAttr* src=nullptr;
     for(const char* key:{"surface_st","st","uv"}){for(const auto& p:m->prims)if(p.key==shade::primKey(key,shade::PrimVec2f)&&p.components==2){src=&p;break;}if(src)break;}
     if(src||split.quads){std::vector<float> uv;uv.reserve(kept.size()*6);
      static const float first[6]={0,0,1,0,0,1},next[6]={1,1,0,1,1,0};
      for(size_t i=0;i<kept.size();++i)for(int j=0;j<3;++j){
       if(src){const unsigned tri[3]={m->triangles[i].a,m->triangles[i].b,m->triangles[i].c};
        const size_t at=src->rate==1?0:src->rate==3?i:src->rate==6?i*3+size_t(j):size_t(tri[j]);
        uv.push_back(m->primData[src->offset+2*at]);uv.push_back(m->primData[src->offset+2*at+1]);}
       else{const float* d=split.second[kept[i]]?next:first;uv.push_back(d[2*j]);uv.push_back(d[2*j+1]);}}
      m->uvs=std::move(uv);m->uvRate=shade::RateFaceVarying;}}
   }
   if(paths&&&o==bakeGeometry&&!plan->cameraMap){
    // The bake camera's maps over this mesh's faces, all of them (whatever its parts).
    need(!reversed,o.getName()+": a bake camera on a mesh with reversed normals is not rendered yet");
    const int mode=plan->camera.mode;const float factor=get<rdl::Float>(*camera,"map_factor");
    const int apW=int(plan->camera.apertureWidth),apH=int(plan->camera.apertureHeight);
    const int pw=int(std::floor(float(apW)*factor)),ph=int(std::floor(float(apH)*factor));
    need(pw>=2&&ph>=2&&size_t(pw)*size_t(ph)<=limits.pixels,camera->getName()+": map factor budget");
    // Texture coordinates: uv_attribute's (none: the mesh's "st", its uv_list), by corner or by vertex.
    const std::string uvName=get<rdl::String>(*camera,"uv_attribute");
    std::vector<float> uvData;int uvRate=0;  // 1 constant, 2 face, 3 vertex, 4 face-varying
    auto byCount=[&](size_t n){return n==indices.size()?4:n==v.size()?3:n==counts.size()?2:n==1?1:0;};
    if(uvName.empty()||uvName=="st"||uvName=="surface_st"){
     const auto& ul=o.get<rdl::Vec2fVector>("uv_list");for(auto u:ul){uvData.push_back(u.x);uvData.push_back(u.y);}uvRate=byCount(ul.size());
    }else{
     for(auto* object:o.get<rdl::SceneObjectVector>("primitive_attributes"))if(object&&get<rdl::String>(*object,"vec2f_key")==uvName){
      const auto& d=get<rdl::Vec2fVector>(*object,"vec2f_values_0");for(auto u:d){uvData.push_back(u.x);uvData.push_back(u.y);}uvRate=byCount(d.size());break;}
    }
    // MoonRay asserts on a mesh without them.
    need(uvRate!=0,o.getName()+": the bake camera finds no texture coordinates "+(uvName.empty()?std::string("(st)"):uvName));
    auto uvAt=[&](const BakeFace& f,int k,float* st){
     const size_t at=uvRate==1?0:uvRate==2?f.face:uvRate==3?size_t(f.vertex[k]):f.corner[k];
     st[0]=uvData[at*2];st[1]=uvData[at*2+1];};
    // Normals: the mesh's (normal_list, by corner or vertex, as its shading normals; or its smooth ones), else none.
    std::vector<V3> normals;int normalRate=0;  // 3 vertex, 4 face-varying
    {const auto& nl=o.get<rdl::Vec3fVector>("normal_list");
     if(!nl.empty()){
      double M[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)M[i][j]=transform[i][j];
      double det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
      double inv[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){int a=(j+1)%3,b=(j+2)%3,cc=(i+1)%3,d=(i+2)%3;inv[i][j]=(M[a][cc]*M[b][d]-M[a][d]*M[b][cc])/det;}
      for(auto n:nl){double x=n.x*inv[0][0]+n.y*inv[0][1]+n.z*inv[0][2],y=n.x*inv[1][0]+n.y*inv[1][1]+n.z*inv[1][2],z=n.x*inv[2][0]+n.y*inv[2][1]+n.z*inv[2][2];
       double l=std::sqrt(x*x+y*y+z*z);normals.push_back({float(x/l),float(y/l),float(z/l)});}
      normalRate=byCount(nl.size());need(normalRate==3||normalRate==4,o.getName()+": the bake camera reads normals by vertex or corner");
     }else if(m->normalRate==shade::RateVertex&&m->normals.size()==m->positions.size()){normals=m->normals;normalRate=3;}}
    auto normalAt=[&](const BakeFace& f,int k,V3& n){if(!normalRate)return false;n=normals[normalRate==3?size_t(f.vertex[k]):f.corner[k]];return true;};
    // MoonRay's faces: a QuadMesh's in order (a triangle repeats its last corner), a TriMesh's triangles as split.
    std::vector<BakeFace> faces;
    if(split.quads){size_t b=0;
     for(size_t f=0;f<counts.size();++f){const int n=counts[f];BakeFace bf{};
      bf.face=f;for(int k=0;k<4;++k){const size_t c=b+size_t(k==3&&n==3?2:k);bf.corner[k]=c;bf.vertex[k]=unsigned(indices[c]);}
      faces.push_back(bf);b+=size_t(n);}}
    else for(size_t i=0;i<split.faceOf.size();++i){BakeFace bf{};bf.face=split.faceOf[i];
     for(int j=0;j<3;++j){bf.corner[j]=split.corners[i*3+size_t(j)];bf.vertex[j]=unsigned(split.vertices[i*3+size_t(j)]);}faces.push_back(bf);}
    // The normal map (normal_map, read only where the mode needs normals): a file OIIO opens, else none (MoonRay logs
    // that it failed to open it and bakes with the surface's normals).
    auto data=std::make_shared<Plan::CameraMapData>();
    bool haveNormalMap=false;const int normalSpace=get<rdl::Int>(*camera,"normal_map_space");
    {const std::string file=get<rdl::String>(*camera,"normal_map");
     if(mode!=0&&!file.empty())if(auto in=OIIO::ImageInput::open(file)){
      const OIIO::ImageSpec spec=in->spec();std::vector<float> texels(size_t(spec.width)*spec.height*std::max(1,spec.nchannels));
      if(spec.width>0&&spec.height>0&&in->read_image(0,0,0,-1,OIIO::TypeDesc::FLOAT,texels.data())){
       data->normalMap.assign(size_t(spec.width)*spec.height*3,0.0f);const int nc=spec.nchannels;
       for(size_t i=0;i<size_t(spec.width)*spec.height;++i)for(int c=0;c<3&&c<nc;++c)data->normalMap[i*3+size_t(c)]=texels[i*size_t(nc)+size_t(c)];
       data->view.normalWidth=unsigned(spec.width);data->view.normalHeight=unsigned(spec.height);haveNormalMap=true;}
      in->close();}}
    const bool needNrm=mode!=0&&(!haveNormalMap||normalSpace==1);
    buildBakeMap(*data,pw,ph,get<rdl::Int>(*camera,"udim"),m->positions,faces,split.quads,uvAt,normalAt,needNrm);
    data->view.positions=data->positions.data();data->view.normals=needNrm?data->normals.data():nullptr;
    data->view.posWidth=unsigned(pw);data->view.posHeight=unsigned(ph);
    data->view.normalMap=haveNormalMap?data->normalMap.data():nullptr;data->view.normalSpace=unsigned(normalSpace);data->view.needNormals=mode!=0?1u:0u;
    for(int i=0;i<12;++i)data->view.worldToRender[i]=plan->worldToRender[i];
    plan->cameraMap=data;plan->camera.map=&data->view;
   }
   prepLaps().add("made: attributes",double(nowNs()-attributesBegan)*1e-9);
   data=std::move(m);++plan->preparedMeshes;
  }
  }
  // A displaced mesh made now carries what shaped it (made here, it is this plan's alone).
  if(!displacementKey.empty()&&data->displacementKey!=displacementKey)const_cast<MeshData&>(*data).displacementKey=displacementKey;
  // A tessellation made now goes to the cache (on a thread of its own); a worker's when it is taken in.
  if(launched){launched->cacheKey=cacheHit?std::string():cacheKey;launched->displacementKey=displacementKey;launched->mesh=plan->meshes.size();tessellations.push_back(launched);}
  else if(!cacheKey.empty()&&!cacheHit)saveGeometry(cacheKey,data);
  // A shared instance's data is counted once, with its first instance.
  if(shareInstance&&!sharedHit)sharedMeshes.emplace(sharedKey,data);
  if(!sharedHit)plan->bytes+=data->positions.size()*sizeof(V3)+data->triangles.size()*sizeof(Tri)+data->curvePoints.size()*sizeof(float)+data->curveSpans.size()*sizeof(unsigned);
  // Over the budget already: said now, not after the rest of the scene is prepared.
  // Tessellated for the coarse first pass: marked, so that no later plan takes it over.
  if(coarseTessellation()&&data&&!reusedData&&(subdGeometry||polyGeometry))const_cast<MeshData&>(*data).coarse=true;
  need(plan->bytes<=limits.bytes,geometryBudgetText(plan->bytes,limits.bytes));plan->meshes.push_back({data,mat?materialIds.at(mat->getName()):~0u,volumeIndex,
   paths?(geometryVisibility(o)&(item.instancer?geometryVisibility(*item.instancer):0xffu)):0xffu,
   paths&&volumeIndex<0&&get<rdl::Int>(o,"side_type")==1,
   plan->aovs?plan->aovs->geomLabel(aov::attribute<rdl::String>(o,"label","")):-1});
  plan->meshes.back().families=indexedFamilies(o,frame,familyCache)|(mat?indexedFamilies(*mat,frame,familyCache):0);
  if(!item.instancer)plan->meshes.back().owner=o.getName();
  else if(shareInstance){plan->meshes.back().owner=item.instancer->getName();plan->meshes.back().ownerInstancer=true;
   plan->meshes.back().ownerIndex=item.instance;}
  if(shareInstance){auto& e=plan->meshes.back();placeInstance(e,instanceMatrix(item));
   instanceValues(*item.instancer,item.instance,item.count,e.instancePrims,e.instancePrimData);}
  if(placedAnew){auto& e=plan->meshes.back();const rdl::Mat4d now=affine(o);placeInstance(e,placedFrom.inverse()*now);
   const auto w=now.inverse();const double rows[12]={w[0][0],w[1][0],w[2][0],w[3][0],w[0][1],w[1][1],w[2][1],w[3][1],w[0][2],w[1][2],w[2][2],w[3][2]};
   for(int i=0;i<12;++i)e.worldToObject[i]=float(rows[i]);}
  if(displacementShader&&!subdGeometry&&!polyGeometry&&!reusedData)displaced.push_back({plan->meshes.size()-1,edgeVertices,displacement});
  // An emissive mesh volume lights the scene (an emissive region over its mesh's box).
  if(volumeIndex>=0&&plan->volumes[size_t(volumeIndex)].kind==volume::KindVdb&&plan->volumes[size_t(volumeIndex)].mapped)
   mappedVolumes.push_back({unsigned(volumeIndex),plan->meshes.size()-1,volumeInner,affine(o),false,&o});
  if(volumeIndex>=0&&plan->volumes[size_t(volumeIndex)].kind==volume::KindBaseMapped)
   mappedVolumes.push_back({unsigned(volumeIndex),plan->meshes.size()-1,volumeInner,affine(o),o.getSceneClass().getName()=="RdlMeshGeometry"&&!get<rdl::Bool>(o,"is_subd")});
  if(volumeIndex>=0&&plan->volumes[size_t(volumeIndex)].kind==volume::KindBase){const auto& vs=plan->volumes[size_t(volumeIndex)];
   if(!(vs.emission.x==0&&vs.emission.y==0&&vs.emission.z==0)&&(vs.visibility&0x3fcu))meshEmissionRegion(*plan,*data,unsigned(volumeIndex),vs.visibility);}
  // XPU2_SESSION_PREP_ITEMS=1: each mesh made, by name, as it is done.
  {static const bool said=[]{const char* v=std::getenv("XPU2_SESSION_PREP_ITEMS");return v&&v[0]=='1';}();
   if(said&&!reusedData)std::printf("MoonRay XPU2: made %s%s in %.3f s (%zu triangles)%c",o.getName().c_str(),piece.suffix.c_str(),double(nowNs()-pieceBegan)*1e-9,data?data->triangles.size():size_t(0),char(10));}
  prepLaps().add(reusedData?"kept meshes' items":"made meshes' items",double(nowNs()-pieceBegan)*1e-9);
  }
 }
 // The workers' tessellations taken in, in the meshes' order: their plan meshes, budget and cache.
 for(auto& job:tessellations){std::shared_ptr<const MeshData> made=job->result.get();
  if(!job->displacementKey.empty()&&made->displacementKey!=job->displacementKey)const_cast<MeshData&>(*made).displacementKey=job->displacementKey;
  plan->bytes+=made->positions.size()*sizeof(V3)+made->triangles.size()*sizeof(Tri);
  need(plan->bytes<=limits.bytes,geometryBudgetText(plan->bytes,limits.bytes));
  plan->meshes[job->mesh].data=made;if(!job->cacheKey.empty())saveGeometry(job->cacheKey,made);}
 tessellations.clear();
 prepLaps().lap("render items (meshes, tessellation, materials)");
 if(gpuTextures(domain)){uint64_t total=0;std::set<texture::File*> files;
  for(auto* set:{&plan->diffuseMaps,&plan->emissionMaps})for(auto& pair:*set){auto& file=*pair.second.file;if(!files.insert(&file).second)continue;total+=file.decodedBytes;need(total<=(4ull<<30),"Aggregate host texture budget");if(!file.pyramid)file.pyramid=std::make_shared<texture::Pyramid>(file.name.string());}
  for(auto& f:plan->textures){auto& file=*f;if(files.insert(&file).second){total+=file.decodedBytes;need(total<=(4ull<<30),"Aggregate host texture budget");}if(!file.pyramid)file.pyramid=std::make_shared<texture::Pyramid>(file.name.string(),file.alpha);plan->textureImages.push_back(&file.pyramid->image);}
 }
 prepLaps().lap("texture pyramids");
 if(plan->paths&&plan->randomWalk)plan->walkSequences=walkSequences();
 // Extra AOVs (RenderContext's material extensions): a material's extra_aovs, a ListMap of ExtraAovMaps, each with a
 // label an expression reads ('U:' and its label); its colour, times its bound map's result, at each hit, before or after
 // the scattering (post_scatter). A program of their own for each material, run once the material is shaded. MoonRay
 // warns of and leaves out a list that is not a ListMap, an entry without extra AOV data and a label no expression reads.
 if(paths&&plan->aovs&&plan->aovs->hasLpe())for(const auto& [name,index]:materialIds){
  const auto found=sorted.find(name);if(found==sorted.end())continue;const rdl::SceneObject& o=*found->second;
  const rdl::SceneObject* list=nullptr;
  for(auto i=o.getSceneClass().beginAttributes();i!=o.getSceneClass().endAttributes();++i)if((*i)->getName()=="extra_aovs")list=get<rdl::SceneObject*>(o,"extra_aovs");
  if(!list||list->getSceneClass().getName()!="ListMap")continue;
  MapProgramBuilder program{plan->materialPool,textureContext};std::vector<float> entries;unsigned count=0;
  auto word=[&](unsigned v){float f;std::memcpy(&f,&v,sizeof f);entries.push_back(f);};
  for(int i=0;i<20;++i){
   const rdl::SceneObject* e=get<rdl::SceneObject*>(*list,"map"+std::to_string(i));
   if(!e||e->getSceneClass().getName()!="ExtraAovMap")continue;
   const std::string label=get<rdl::String>(*e,"label");if(label.empty())continue;
   const int id=plan->aovs->lpeLabel("U:"+label);if(id<0)continue;
   attributes(*e,{"label","post_scatter","color"},{"color"});
   const rdl::Attribute* colour=nullptr;
   for(auto a=e->getSceneClass().beginAttributes();a!=e->getSceneClass().endAttributes();++a)if((*a)->getName()=="color")colour=*a;
   const rdl::SceneObject* bound=colour?e->getBinding(*colour):nullptr;
   const unsigned reg=bound?program.reg(*bound):shade::MapNoRegister;
   const auto c=get<rdl::Rgb>(*e,"color");need(std::isfinite(c.r)&&std::isfinite(c.g)&&std::isfinite(c.b),e->getName()+": invalid colour");
   word(unsigned(id));word(get<rdl::Bool>(*e,"post_scatter")?1u:0u);word(reg);entries.push_back(c.r);entries.push_back(c.g);entries.push_back(c.b);
   need(++count<=shade::Bsdf::MaxExtraAovs,o.getName()+": more extra AOVs than a ListMap holds");}
  if(!count)continue;
  const unsigned prog=program.finish();
  auto& rec=plan->materialRecords.at(index);rec.extras=unsigned(plan->materialPool.size());
  {float f;std::memcpy(&f,&prog,sizeof f);plan->materialPool.push_back(f);std::memcpy(&f,&count,sizeof f);plan->materialPool.push_back(f);}
  plan->materialPool.insert(plan->materialPool.end(),entries.begin(),entries.end());}
 // MoonRay's bake camera bakes a mesh the scene renders (its procedural).
 need(!bakeGeometry||plan->cameraMap,camera->getName()+": the bake camera's mesh is not in the layer");
 // The mapped volumes' extinction, and the emissive ones' regions from their maps (computeEmissionDistribution).
 if(!mappedVolumes.empty()){plan->volumeMaps=volume::MapEval{&runVolumeMaps,plan.get()};
  for(const auto& mv:mappedVolumes){
   if(mv.vdb){bakeVdbVolume(*plan,mv.volume,*mv.vdb,*mv.shader);continue;}
   {const auto bakedFrom=deepSignature(*mv.shader,frame);
    bakeVolume(*plan,mv.volume,*plan->meshes[mv.mesh].data,*mv.shader,mv.nodeXform,mv.polygons,previous.get(),&bakedFrom);}
   const auto& vs=plan->volumes[mv.volume];
   const bool emits=(vs.mapped&((1u<<volume::MapEmissionColor)|(1u<<volume::MapEmissionIntensity)))!=0||!(vs.emission.x==0&&vs.emission.y==0&&vs.emission.z==0);
   if(emits&&(vs.visibility&0x3fcu))meshEmissionRegion(*plan,*plan->meshes[mv.mesh].data,mv.volume,vs.visibility,true);}}
 prepLaps().lap("extra AOVs, volumes");
 // The displaced meshes (mesh_displacement.h), their maps' textures read.
 if(!displaced.empty()){const displace::Textures textures{plan->textureImages.data(),unsigned(plan->textureImages.size())};
  for(auto& d:displaced){auto& mesh=plan->meshes[d.mesh];const MeshData& base=*mesh.data;
   auto out=displace::displaceMesh(base,d.edgeVertices,d.shader,plan->materialPool.data(),textures,plan->worldToRender);
   for(size_t k=0;k<base.steps.size();++k){MeshData step=base;step.positions=base.steps[k];step.steps.clear();step.normals1.clear();
    auto moved=displace::displaceMesh(step,d.edgeVertices,d.shader,plan->materialPool.data(),textures,plan->worldToRender);
    need(moved->positions.size()==out->positions.size(),base.name+": a time step's displacement does not match the first");
    if(k==0)out->normals1=moved->normals;
    out->steps.push_back(std::move(moved->positions));}
   need(out->positions.size()<=limits.vertices&&out->triangles.size()<=limits.triangles&&!out->triangles.empty(),base.name+": displaced mesh budget");
   // Kept for the next plan under what shaped it (the mesh's signature and displacement key).
   out->signature=base.signature;out->displacementKey=base.displacementKey;
   plan->bytes+=out->positions.size()*sizeof(V3)+out->triangles.size()*sizeof(Tri);plan->bytes-=base.positions.size()*sizeof(V3)+base.triangles.size()*sizeof(Tri);
   need(plan->bytes<=limits.bytes,geometryBudgetText(plan->bytes,limits.bytes));
   mesh.data=std::move(out);}}
 // Textures at bounced hits need MoonRay's ray differentials through lobes (PATH_TRACER.md, phase 1b).
 if(plan->paths&&(!plan->diffuseMaps.empty()||!plan->emissionMaps.empty()))need(plan->maxDepth==0||plan->maxDiffuseDepth==0,"Textured bounces need secondary ray differentials");
 prepLaps().lap("displacement");
 // The light sets' trees (Scene::updateLightList's LightAccelerator::buildSamplingTree), for adaptive light sampling.
 buildLightTrees(*plan);
 // Light filter tables (Scene::buildLightFilterTables): for each mesh, each light's filters, those of the mesh's
 // LightFilterSet where it has one, the filters on; lists and tables shared where they are the same.
 if(plan->paths){
  FilterAdmission filters{*plan,sv,previous.get(),{}};
  {rdl::Mat4d w2r;for(int i=0;i<4;++i)for(int j=0;j<4;++j)w2r[i][j]=plan->world2render[4*i+j];const rdl::Mat4d r2w=w2r.inverse();
   plan->renderOrigin=V3{float(r2w[3][0]),float(r2w[3][1]),float(r2w[3][2])};}
  filters.mapProgram=[&](const rdl::SceneObject& m){MapProgramBuilder b{plan->materialPool,textureContext};const unsigned reg=b.reg(m);
   need(!b.polyVertices&&b.required.empty(),m.getName()+": a cookie's map that reads the surface is not rendered");const unsigned program=b.finish();
   // The textures its maps read, decoded now (the plan's images were gathered before the filters).
   for(size_t i=plan->textureImages.size();i<plan->textures.size();++i){auto& file=*plan->textures[i];
    if(!file.pyramid)file.pyramid=std::make_shared<texture::Pyramid>(file.name.string(),file.alpha);plan->textureImages.push_back(&file.pyramid->image);}
   return std::make_pair(program,reg);};
  std::vector<const rdl::SceneObject*> lightObjects(plan->lights.size(),nullptr);for(auto& pr:lightIds)lightObjects[pr.second]=pr.first;
  std::map<std::vector<unsigned>,unsigned> listIds,tableIds;
  plan->meshFilterTable.assign(plan->meshes.size(),~0u);
  for(size_t mi=0;mi<plan->meshes.size()&&mi<meshFilterSets.size();++mi){
   const rdl::LightFilterSet* set=meshFilterSets[mi];std::vector<unsigned> table(plan->lights.size(),~0u);bool any=false;
   for(size_t li=0;li<lightObjects.size();++li){const rdl::SceneObject* lo=lightObjects[li];if(!lo)continue;
    std::vector<unsigned> items;
    for(auto* fo:lo->get<rdl::SceneObjectVector>("light_filters")){
     if(!fo||!fo->isA<rdl::LightFilter>())continue;const rdl::LightFilter* lf=fo->asA<rdl::LightFilter>();
     if(set&&!set->contains(lf))continue;if(!get<rdl::Bool>(*fo,"on"))continue;
     const unsigned id=filters.admit(*fo);if(std::find(items.begin(),items.end(),id)==items.end())items.push_back(id);}
    if(items.empty())continue;
    need(items.size()<=lights::FilterMaxList,lo->getName()+": more than 16 light filters");
    std::sort(items.begin(),items.end());
    auto it=listIds.find(items);
    if(it==listIds.end()){lights::FilterList l;for(size_t k=0;k<items.size();++k){l.items[k]=items[k];if(plan->filters[items[k]].needsSamples)l.needsSamples=1;}
     l.count=unsigned(items.size());it=listIds.emplace(items,unsigned(plan->filterLists.size())).first;plan->filterLists.push_back(l);}
    table[li]=it->second;any=true;}
   if(!any)continue;
   auto it=tableIds.find(table);
   if(it==tableIds.end()){it=tableIds.emplace(table,unsigned(plan->filterTables.size())).first;plan->filterTables.insert(plan->filterTables.end(),table.begin(),table.end());}
   plan->meshFilterTable[mi]=it->second;}
 }
 need(plan->bytes<=limits.bytes,geometryBudgetText(plan->bytes,limits.bytes));
 if(paths)routeMeshes(*plan,limits.gpuBytes);
 // The coarse first pass: dense meshes simplified (coarse_geometry.h), and how many of its meshes are lighter than the full
 // plan's (the session renders a coarse pass only where some are).
 if(limits.coarse){const float angle=pixelAngle(plan->camera,plan->width,plan->height);
  const V3 eye=surface::transformPoint(plan->camera.cameraToRender,plan->camera.focalPoint);
  std::map<const MeshData*,std::shared_ptr<const MeshData>> lighter;
  for(auto& mesh:plan->meshes){if(!mesh.data)continue;auto it=lighter.find(mesh.data.get());
   if(it==lighter.end())it=lighter.emplace(mesh.data.get(),coarseMesh(*mesh.data,eye,angle)).first;
   if(it->second)mesh.data=it->second;}
  for(const auto& mesh:plan->meshes)if(mesh.data&&mesh.data->coarse)++plan->coarsened;}
 // A realtime plan's shading reads each texture from its first level of 1024 pixels or less (Pyramid::view); what the
 // plan worked out from the textures (displacement above) read them whole.
 if(plan->realtime){size_t capped=0;
  for(size_t i=0;i<plan->textureImages.size();++i){const auto& view=plan->textures[i]->pyramid->view(true);capped+=&view!=plan->textureImages[i];plan->textureImages[i]=&view;}
  if(capped)std::printf("XPU2: realtime textures at most %u pixels a side (%zu of %zu made smaller)\n",texture::Pyramid::realtimeSide(),capped,plan->textureImages.size());}
 prepLaps().lap("light trees, filters, the rest");return plan;
}
X2_SPLIT_INLINE PlanView prepare(CanonicalSession& session,const CanonicalView& frame,PlanView previous,Limits limits,const fs::path& assetRoot,Domain domain,bool allowAdaptive,bool paths){
 return session.inspectPrepared(frame,[&](const rdl::SceneContext& context,CanonicalView exact){return prepare(context,exact,previous,limits,assetRoot,domain,allowAdaptive,paths);});
}
#endif
}

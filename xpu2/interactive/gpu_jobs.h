// Private shared bounded job ABI. Whole samples, no constructed BSDF crosses PCIe.
#pragma once
#include "surface_core.h"
#include "path_stage.h"
#include "texture_core.h"
namespace xpu2::interactive::resume {struct Batch;}
namespace xpu2::interactive::mixed {
using sample::U32;using sample::U64;
constexpr U32 Capacity=256;
struct Config {path::Epoch epoch;surface::Camera camera;surface::V3 sky;U32 width,height,seed,meshes;surface::Filter filter;float filterWidth;U32 spatial=0,surfaceLimit=1,spp=1,textures=0,sampling=3;
    // The asynchronous renderer's path tracer (async_integrator.h, PATH_TRACER.md); 0 keeps the prototype's estimator.
    U32 paths=0,lightCount=0,bsdfSamples=1,lightSamples=1,maxDepth=0,maxDiffuseDepth=0,clampDepth=1;float rrThreshold=0,clampValue=0;
    // The rest of MoonRay's integrator settings for the materials' lobes (PathIntegrator::update).
    U32 maxGlossyDepth=0,maxMirrorDepth=0,maxHairDepth=0,maxPresenceDepth=0,shadowTerminatorFix=0;float roughnessClamping=0,transparencyThreshold=1;
    U32 bssrdfSamples=4,maxSubsurfacePerPath=1,enableSss=1;
    // World to MoonRay's render space (the camera's), 3x4 rows: the maps' render space.
    float worldToRender[12]={1,0,0,0,0,1,0,0,0,0,1,0};
    // Volumes (volume_core.h; PathIntegrator::update's volume settings): how many, and MoonRay's volume scene variables.
    U32 volumeCount=0,maxVolumeDepth=1,volumeIlluminationSamples=4,volumeOverlapMode=0;
    // How many lights the camera sees (their visible_in_camera), which camera rays meet.
    U32 visibleLights=0;
    // Adaptive light sampling (light_sampling_mode 1): whether the lights' trees pick the lights sampled, and the
    // quality threshold of their splitting (light_sampling_quality).
    U32 lightTree=0;float lightTreeThreshold=0.5f;
    float presenceThreshold=0.999f,presenceQuality=0.75f;  // presence_threshold, presence_quality
    float volumeTransmittanceThreshold=0.005f,volumeAttenuationFactor=0.65f,volumeContributionFactor=0.65f,volumePhaseAttenuationFactor=0.5f;
    // volume_quality and volume_shadow_quality, inverted: a heterogeneous volume's steps are its feature size times these.
    float invVolumeQuality=2.0f,invVolumeShadowQuality=1.0f;
    // The pixels rendered (the sub viewport: min x, min y, max x, max y, half open) and the texture differential
    // scale (PathIntegrator::initPrimaryRay's mTextureDiffScale).
    int viewport[4]={0,0,0x7fffffff,0x7fffffff};float textureDiffScale=1;
    // The camera's pixel sample map (RenderDriver::computeTotalNumSamples): each pixel's own sample count, or none.
    const U32* pixelSamples=nullptr;
    // Motion blur (enable_motion_blur): each sample at a time in the shutter (0 open, 1 close), bent by the camera's
    // mb_shutter_bias as Sampler::getPrimarySample bends it.
    U32 motionBlur=0;float shutterBias=0;
    // Light filters (light_filter_core.h): whether any light has some at any shading point.
    U32 lightFilters=0;
    // The kit's MoonRay source switches the integrator reads (Plan::modoSwitches): 1 MOONRAY_MODO_TOON_ALBEDO, 2
    // MOONRAY_MODO_FOG_ALPHA, 4 MOONRAY_MODO_CRYPTO_VOLUMES, 8 MOONRAY_MODO_CAMERA_LIGHT_LPE.
    U32 modoSwitches=0;
    // Bounce-ray reuse (async_integrator.h): the ray visibility bits (rdl2 VisibilityType) of the bounces whose rays
    // meet exactly the occluders a shadow ray meets, every mesh visible to both or to neither; 0: no reuse.
    U32 shadowReuse=0;};
struct Job {path::Epoch epoch{};U64 sequence=0;U32 count=0,first=0,samples=0,fault=0;U64 delay=0;U32 pixels[Capacity]{};};
struct Pixel {double sum[4]{};U32 count=0,status=0,pixel=0,pad=0;};
struct Result {path::Epoch epoch{};U64 sequence=0;U32 count=0,pad=0;Pixel pixels[Capacity]{};};
struct Geometry {const surface::V3* vertices;const surface::Triangle* triangles;U32 vertexCount,triangleCount;};
struct Params {const Config* scene;const surface::Lambert* materials;U64 traversable;Result* result;Job job;resume::Batch* continuation=nullptr;U32 stageSteps=0,reset=0;const Geometry* geometry=nullptr;const tex::Maps* maps=nullptr;};
static_assert(sizeof(Pixel)==48&&sizeof(Job)==1088&&sizeof(Params)==1152&&sizeof(Result)==12336,"Shared job ABI");
X2_HD inline bool same(const Job& j,const Config& c){return path::equal(j.epoch,c.epoch)&&j.count>0&&j.count<=Capacity&&j.samples>0&&j.samples<=4096&&c.sampling<=3&&j.first<=0xffffffffu-j.samples;}
}

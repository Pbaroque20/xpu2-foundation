// The GPU program's volume module (INDEXER.md, 5): the volumes' radiance, the subsurface's and the shadow rays'
// transmittance through volumes (async_integrator.h), as continuation callables of the path tracer: they trace, so
// they cannot be direct callables. OptiX compiles it beside the other modules, in parallel.
#include "async_gpu_device.h"

extern "C" __device__ void __continuation_callable__volumeRadiance(const mixed::Config* c, const sample::Identity* root, GpuBackend* backend,
                                                                   paths::Pending* pv, surface::V3 org, surface::V3 dir, float tnear, float tfar,
                                                                   surface::V3* alphaTransmittance, surface::V3* minTransmittance,
                                                                   VolumeCryptoShares* cryptoShares, const VolumeLpe* lpe, float* surfaceT,
                                                                   surface::V3* out) {
    *out = volumeRadiance(*c, *root, *backend, *pv, org, dir, tnear, tfar, alphaTransmittance, minTransmittance, cryptoShares, lpe, surfaceT);
}

extern "C" __device__ void __continuation_callable__subsurfaceRadiance(const mixed::Config* c, const sample::Identity* root, GpuBackend* backend,
                                                                       U32 lightSet, const paths::Pending* pv, U32 subsurfaceDepth,
                                                                       const shade::State* st, const resume::Hit* hit, surface::V3 rayDir,
                                                                       const shade::Bsdf* bsdf, bool doIndirect, paths::Pending* stack, U32* top,
                                                                       SampleStatus* status, const SubsurfaceLpe* lpe, shade::Color* ssAov,
                                                                       surface::V3* out) {
    *out = subsurfaceRadiance(*c, *root, *backend, lightSet, *pv, subsurfaceDepth, *st, *hit, rayDir, *bsdf, doIndirect, stack, *top, *status, lpe,
                              ssAov);
}

extern "C" __device__ void __continuation_callable__volumeTransmittance(const mixed::Config* c, GpuBackend* backend, paths::Stream* s,
                                                                        surface::V3 org, surface::V3 dir, float tnear, float tfar, float scale,
                                                                        int depth, surface::V3* out) {
    *out = volumeTransmittance(*c, *backend, *s, org, dir, tnear, tfar, scale, depth);
}

// The emissive volume integral, for the path tracer's surfaces lit by glowing volumes: compiled here alone (the main
// module calls this instead of holding its own copy).
extern "C" __device__ void __continuation_callable__emissiveVolumeIntegral(const mixed::Config* c, GpuBackend* backend, paths::Stream* s,
                                                                            surface::V3 org, surface::V3 dir, float tnear, float tEnd, U32 volume,
                                                                            int depth, surface::V3* out) {
    *out = emissiveVolumeIntegral(*c, *backend, *s, org, dir, tnear, tEnd, volume, depth);
}

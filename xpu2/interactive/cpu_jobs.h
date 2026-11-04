// Same admitted planar estimator as cpu_lane.h, in bounded complete-pixel jobs.
#pragma once
#include "cpu_lane.h"
#include "gpu_jobs.h"
#include "resume_core.h"
#include <chrono>
namespace xpu2::interactive::mixed {
inline Config config(const cpu::PlanView& p){auto e=p->source->routing->epoch;Config c{{e.session,e.scene,e.camera,e.settings},p->camera,p->sky,p->width,p->height,p->seed,unsigned(p->meshes.size()),p->filter,p->filterWidth,cpu::isSpatial(p->domain)?1u:0u,p->surfaceLimit,p->spp,cpu::gpuTextures(p->domain)?1u:0u,p->sampling};
 // An investigation switch: the sampling policy's bits (sample_stream.h; 1 the camera's Sobol dimensions, 2 the first
 // diffuse direction's), XPU2_SESSION_SAMPLING_POLICY.
 if(const char* policy=std::getenv("XPU2_SESSION_SAMPLING_POLICY"))c.sampling=unsigned(std::atoi(policy));
 // The path tracer's first-hit light samples take Sobol dimensions of their own (sample_stream.h, bit 4).
 if(p->paths){c.paths=1;if(p->sampling&2)c.sampling|=4|8;c.lightCount=U32(p->lights.size());for(const auto& l:p->lights)c.visibleLights+=l.visibleInCamera?1u:0u;c.lightTree=p->lightTree.empty()?0u:1u;c.presenceThreshold=p->presenceThreshold;c.presenceQuality=p->presenceQuality;c.lightTreeThreshold=p->lightSamplingQuality;c.bsdfSamples=p->bsdfSamples;c.lightSamples=p->lightSamples;c.maxDepth=p->maxDepth;c.maxDiffuseDepth=p->maxDiffuseDepth;c.clampDepth=p->clampDepth;c.rrThreshold=p->rrThreshold;c.clampValue=p->clampValue;
  c.maxGlossyDepth=p->maxGlossyDepth;c.maxMirrorDepth=p->maxMirrorDepth;c.maxHairDepth=p->maxHairDepth;c.maxPresenceDepth=p->maxPresenceDepth;
  c.shadowTerminatorFix=p->shadowTerminatorFix;c.roughnessClamping=p->roughnessClamping;c.transparencyThreshold=p->transparencyThreshold;
  c.bssrdfSamples=p->bssrdfSamples;c.maxSubsurfacePerPath=p->maxSubsurfacePerPath;c.enableSss=p->enableSss;
  for(int i=0;i<12;++i)c.worldToRender[i]=p->worldToRender[i];
  c.volumeCount=U32(p->volumes.size());c.maxVolumeDepth=p->maxVolumeDepth;c.volumeIlluminationSamples=p->volumeIlluminationSamples;
  c.volumeOverlapMode=p->volumeOverlapMode;c.volumeTransmittanceThreshold=p->volumeTransmittanceThreshold;
  c.volumeAttenuationFactor=p->volumeAttenuationFactor;c.volumeContributionFactor=p->volumeContributionFactor;
  c.volumePhaseAttenuationFactor=p->volumePhaseAttenuationFactor;
  c.invVolumeQuality=1.0f/p->volumeQuality;c.invVolumeShadowQuality=1.0f/p->volumeShadowQuality;
  for(int i=0;i<4;++i)c.viewport[i]=p->viewport[i];c.textureDiffScale=p->textureDiffScale;
  c.pixelSamples=p->pixelSamples.empty()?nullptr:p->pixelSamples.data();
  c.motionBlur=p->motion.on?1u:0u;c.shutterBias=p->shutterBias;c.lightFilters=p->filterLists.empty()?0u:1u;c.modoSwitches=p->modoSwitches;
  // Bounce-ray reuse: the bounce ray types each mesh is as visible to as to shadow rays (2); none with shadow linking,
  // whose occluders depend on the light.
  const char* reuse=std::getenv("XPU2_SESSION_SHADOW_REUSE");
  if(p->shadowLinks.empty()&&!(reuse&&reuse[0]=='0')){U32 mask=0;for(U32 b=4;b;b<<=1){bool same=true;
    for(const auto& m:p->meshes)if(((m.visibility&2u)!=0)!=((m.visibility&b)!=0)){same=false;break;}if(same)mask|=b;}c.shadowReuse=mask;}}
 return c;}
// Primary ray differentials projected onto the admitted XY plane. MoonRay's
// non-tessellated triangle without a named ST attribute uses (0,0),(1,0),(1,1).
inline surface::Lambert material(const cpu::PlanView& plan,const RTCRayHit& hit,const surface::Ray& ray){
 const auto& mesh=plan->meshes.at(hit.hit.geomID);auto value=plan->materials.at(mesh.material);
 if(!plan->diffuseMaps.count(mesh.material)&&!plan->emissionMaps.count(mesh.material))return value;
 if(cpu::gpuTextures(plan->domain)){const auto& t=mesh.data->triangles.at(hit.hit.primID);tex::Footprint f;
  cpu::need(tex::footprint(mesh.data->positions.at(t.a),mesh.data->positions.at(t.b),mesh.data->positions.at(t.c),ray,hit.hit.u,hit.hit.v,plan->spp,f,cpu::isSpatial(plan->domain)),"Shared texture derivatives");
  auto apply=[&](surface::V3& base,const auto& maps){auto it=maps.find(mesh.material);if(it==maps.end()||(base.x==0&&base.y==0&&base.z==0))return;surface::V3 color;cpu::need(tex::lookup(it->second.descriptor(plan->realtime),f,color)==tex::Ready,"CPU software texture unavailable");base={base.x*color.x,base.y*color.y,base.z*color.z};};apply(value.albedo,plan->diffuseMaps);apply(value.emission,plan->emissionMaps);return value;
 }
 const auto& tri=mesh.data->triangles.at(hit.hit.primID);
 auto a=mesh.data->positions.at(tri.a),b=mesh.data->positions.at(tri.b),c=mesh.data->positions.at(tri.c);
 auto e1=surface::add(b,surface::mul(a,-1)),e2=surface::add(c,surface::mul(a,-1));float determinant=e1.x*e2.y-e1.y*e2.x;
 cpu::need(std::isfinite(determinant)&&determinant>0,"Texture differential geometry");
 // PathIntegrator scales primary ray differentials by 1/min(sqrt(planned spp),2).
 // Use the full plan, never this progressive pass's sample count.
 float footprint=1.f/std::min(std::sqrt(float(plan->spp)),2.f);
 auto dp=[&](surface::V3 d){if(footprint<1){surface::V3 scaled;cpu::need(surface::normalize(surface::add(ray.direction,surface::mul(surface::add(d,surface::mul(ray.direction,-1)),footprint)),scaled),"Texture footprint");d=scaled;}cpu::need(d.z<0,"Texture differential ray");return surface::mul(surface::add(surface::mul(d,-ray.origin.z/d.z),surface::mul(ray.direction,ray.origin.z/ray.direction.z)),1.f);};
 auto uv=[&](surface::V3 d){auto p=dp(d);float u=(p.x*e2.y-p.y*e2.x)/determinant,v=(e1.x*p.y-e1.y*p.x)/determinant;return std::array<float,2>{u+v,v};};
 auto dx=uv(ray.dx),dy=uv(ray.dy);float u=hit.hit.u+hit.hit.v,v=hit.hit.v;
 auto evaluate=[&](surface::V3 base,const std::map<unsigned,texture::Binding>& bindings){
  auto found=bindings.find(mesh.material);if(found==bindings.end()||(base.x==0&&base.y==0&&base.z==0))return base;
  auto map=found->second.lookup(u,v,dx[0],dx[1],dy[0],dy[1]);return surface::V3{base.x*map.x,base.y*map.y,base.z*map.z};
 };
 value.albedo=evaluate(value.albedo,plan->diffuseMaps);value.emission=evaluate(value.emission,plan->emissionMaps);return value;
}
inline Result spatialJob(const cpu::PlanView&,const cpu::AccelView&,const Job&);
inline Result cpuJob(const cpu::PlanView& p,const cpu::AccelView& a,const Job& j){
 if(cpu::isSpatial(p->domain))return spatialJob(p,a,j);
 auto c=config(p);cpu::need(same(j,c),"Invalid CPU job");Result result{};result.epoch=j.epoch;result.sequence=j.sequence;result.count=j.count;
 for(unsigned lane=0;lane<j.count;++lane){auto& out=result.pixels[lane];out.pixel=j.pixels[lane];cpu::need(out.pixel<c.width*c.height,"Invalid pixel");
  for(unsigned s=0;s<j.samples;++s){sample::Identity id{out.pixel,j.first+s,c.seed,0};float x,y;
   cpu::need(surface::filter(c.filter,c.filterWidth,sample::draw(id,0,c.sampling),x)&&surface::filter(c.filter,c.filterWidth,sample::draw(id,1,c.sampling),y),"Filter failure");surface::Ray ray;
   cpu::need(surface::cameraRay(c.camera,float(out.pixel%c.width)+x,float(out.pixel/c.width)+y,ray),"Camera failure");
   RTCRayHit hit{};hit.ray.org_x=ray.origin.x;hit.ray.org_y=ray.origin.y;hit.ray.org_z=ray.origin.z;hit.ray.dir_x=ray.direction.x;hit.ray.dir_y=ray.direction.y;hit.ray.dir_z=ray.direction.z;hit.ray.tnear=ray.nearT;hit.ray.tfar=ray.farT;hit.ray.mask=~0u;hit.hit.geomID=RTC_INVALID_GEOMETRY_ID;RTCIntersectArguments args;rtcInitIntersectArguments(&args);rtcIntersect1(a->scene,&hit,&args);
   if(hit.hit.geomID!=RTC_INVALID_GEOMETRY_ID){cpu::need(hit.hit.geomID<p->meshes.size(),"Geometry id");surface::V3 normal;cpu::need(surface::normalize({hit.hit.Ng_x,hit.hit.Ng_y,hit.hit.Ng_z},normal)&&normal.z>0,"Normal");surface::Sample sample;auto mat=material(p,hit,ray);
    cpu::need(surface::lambert(mat,normal,sample::draw(id,16,c.sampling),sample::draw(id,17,c.sampling),sample),"Material");
    auto origin=surface::add(ray.origin,surface::mul(ray.direction,hit.ray.tfar));origin.z=std::numeric_limits<float>::min();RTCRay shadow{};shadow.org_x=origin.x;shadow.org_y=origin.y;shadow.org_z=origin.z;shadow.dir_x=sample.wi.x;shadow.dir_y=sample.wi.y;shadow.dir_z=sample.wi.z;shadow.tfar=INFINITY;shadow.mask=~0u;RTCOccludedArguments oa;rtcInitOccludedArguments(&oa);rtcOccluded1(a->scene,&shadow,&oa);
    if(shadow.tfar>=0&&sample.pdf>0){out.sum[0]+=double(sample.f.x/sample.pdf)*c.sky.x;out.sum[1]+=double(sample.f.y/sample.pdf)*c.sky.y;out.sum[2]+=double(sample.f.z/sample.pdf)*c.sky.z;}out.sum[0]+=mat.emission.x;out.sum[1]+=mat.emission.y;out.sum[2]+=mat.emission.z;out.sum[3]+=1;
   }++out.count;
  }
 }a->device->check();return result;
}

struct CpuStageBackend {
 const cpu::PlanView& p;const cpu::AccelView& a;
 resume::Hit trace(const surface::Ray& ray){
  RTCRayHit h{};h.ray.org_x=ray.origin.x;h.ray.org_y=ray.origin.y;h.ray.org_z=ray.origin.z;h.ray.dir_x=ray.direction.x;h.ray.dir_y=ray.direction.y;h.ray.dir_z=ray.direction.z;h.ray.tnear=ray.nearT;h.ray.tfar=ray.farT;h.ray.mask=~0u;h.hit.geomID=RTC_INVALID_GEOMETRY_ID;RTCIntersectArguments args;rtcInitIntersectArguments(&args);cpu::PreciseContext context;args.context=&context.base;rtcIntersect1(a->scene,&h,&args);
  resume::Hit out;out.geom=h.hit.geomID;if(out.geom!=~0u){out.prim=h.hit.primID;out.t=h.ray.tfar;out.u=h.hit.u;out.v=h.hit.v;if(cpu::isSpatial(p->domain)){const auto& mesh=*p->meshes.at(out.geom).data;auto tri=mesh.triangles.at(out.prim);cpu::need(resume::geometryHit(out,ray,mesh.positions.at(tri.a),mesh.positions.at(tri.b),mesh.positions.at(tri.c)),"Spatial hit geometry");}
   else cpu::need(surface::normalize({h.hit.Ng_x,h.hit.Ng_y,h.hit.Ng_z},out.normal)&&out.normal.z>0,"Resume normal");}return out;
 }
 bool material(const resume::Hit& h,const surface::Ray& ray,surface::Lambert& out){RTCRayHit hit{};hit.hit.geomID=h.geom;hit.hit.primID=h.prim;hit.hit.u=h.u;hit.hit.v=h.v;out=mixed::material(p,hit,ray);return true;}
 bool visible(surface::V3 o,surface::V3 d){RTCRay ray{};ray.org_x=o.x;ray.org_y=o.y;ray.org_z=o.z;ray.dir_x=d.x;ray.dir_y=d.y;ray.dir_z=d.z;ray.tfar=INFINITY;ray.mask=~0u;RTCOccludedArguments args;rtcInitOccludedArguments(&args);rtcOccluded1(a->scene,&ray,&args);return ray.tfar>=0;}
};
inline Result spatialJob(const cpu::PlanView& p,const cpu::AccelView& a,const Job& j){
 auto c=config(p);cpu::need(same(j,c),"Invalid spatial job");CpuStageBackend backend{p,a};Result r{};r.epoch=j.epoch;r.sequence=j.sequence;r.count=j.count;
 for(unsigned i=0;i<j.count;++i){resume::State state{};state.film.pixel=j.pixels[i];while(state.phase<resume::Complete)resume::step(state,j,c,backend);r.pixels[i]=resume::snapshot(state);}
 a->device->check();return r;
}
inline Result cpuSlice(const cpu::PlanView& p,const cpu::AccelView& a,const Job& j,resume::Batch& batch,unsigned steps,std::chrono::steady_clock::time_point until){
 auto c=config(p);cpu::need(same(j,c)&&resume::matches(batch,j)&&steps>0&&steps<=64,"Invalid CPU continuation");CpuStageBackend backend{p,a};Result r{};r.epoch=j.epoch;r.sequence=j.sequence;r.count=j.count;r.pad=++batch.revision;
 for(unsigned i=0;i<j.count;++i){auto& state=batch.paths[i];for(unsigned n=0;n<steps&&state.phase<resume::Complete;++n){if(std::chrono::steady_clock::now()>=until)break;resume::step(state,j,c,backend);}r.pixels[i]=resume::snapshot(state);}
 a->device->check();return r;
}

}

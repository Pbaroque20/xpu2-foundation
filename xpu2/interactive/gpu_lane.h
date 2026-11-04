// Private OptiX lane. One feeder owns this context for its entire lifetime.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cuda.h>
#include <optix.h>
// OptiX's function table is defined once: in the session's unit when the session is split (XPU2_SPLIT_UNITS).
#if !defined(XPU2_SPLIT_UNITS) || (!defined(XPU2_SCENE_UNIT) && !defined(XPU2_CPU_UNIT))
#include <optix_function_table_definition.h>
#endif
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "cpu_jobs.h"
#include <fstream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <deque>
namespace xpu2::interactive::gpu {
inline void ox(OptixResult e){if(e!=OPTIX_SUCCESS)throw std::runtime_error("OptiX error "+std::to_string(e));}
using StartClock=std::chrono::steady_clock;
inline double startLap(StartClock::time_point& tick){auto now=StartClock::now();double v=std::chrono::duration<double>(now-tick).count();tick=now;return v;}
struct StartupOptions {std::string cache;unsigned compileThreads=4;};
struct StartupTimes {double driver=0,context=0,module=0,groups=0,pipeline=0,buffers=0;};
struct Driver {
 double startupSeconds=0;
 HMODULE library=nullptr;CUcontext context=nullptr,previous=nullptr;
 // Allocation caps guard against runaway use; the asynchronous lane raises them for its tile cache.
 size_t deviceBytes=0,pinnedBytes=0,peakDevice=0,peakPinned=0,deviceCap=256ull<<20,pinnedCap=64ull<<20;unsigned synchronizations=0;uint64_t allocations=0,frees=0;std::thread::id owner=std::this_thread::get_id();
#define API(n) decltype(&n) n##_p=nullptr;
 API(cuInit) API(cuDeviceGet) API(cuDeviceGetName) API(cuCtxGetCurrent) API(cuCtxSetCurrent) API(cuCtxCreate_v2) API(cuCtxDestroy_v2) API(cuCtxSynchronize)
 API(cuMemAlloc_v2) API(cuMemFree_v2) API(cuMemAllocHost_v2) API(cuMemFreeHost) API(cuMemcpyHtoDAsync_v2) API(cuMemcpyDtoHAsync_v2)
 API(cuStreamCreate) API(cuStreamDestroy_v2) API(cuStreamWaitEvent) API(cuEventCreate) API(cuEventRecord) API(cuEventQuery) API(cuEventDestroy_v2)
 API(cuMemsetD8Async) API(cuMemsetD32Async) API(cuDeviceGetAttribute) API(cuEventElapsedTime) API(cuMemcpyDtoDAsync_v2)
 API(cuModuleLoadData) API(cuModuleGetFunction) API(cuModuleUnload) API(cuLaunchKernel)
#undef API
 static void check(CUresult e){if(e!=CUDA_SUCCESS)throw std::runtime_error("CUDA error "+std::to_string(e));}
 void thread()const{cpu::need(std::this_thread::get_id()==owner,"GPU context requires feeder owner");}
 Driver(){auto start=StartClock::now();try{
  wchar_t path[MAX_PATH];auto n=GetSystemDirectoryW(path,MAX_PATH);cpu::need(n&&n<MAX_PATH-12,"System DLL path");std::wstring file(path);file+=L"\\nvcuda.dll";library=LoadLibraryW(file.c_str());cpu::need(library,"CUDA driver unavailable");
#define LOAD(n) n##_p=reinterpret_cast<decltype(n##_p)>(GetProcAddress(library,#n));cpu::need(n##_p,#n);
 LOAD(cuInit) LOAD(cuDeviceGet) LOAD(cuDeviceGetName) LOAD(cuCtxGetCurrent) LOAD(cuCtxSetCurrent) LOAD(cuCtxCreate_v2) LOAD(cuCtxDestroy_v2) LOAD(cuCtxSynchronize)
 LOAD(cuMemAlloc_v2) LOAD(cuMemFree_v2) LOAD(cuMemAllocHost_v2) LOAD(cuMemFreeHost) LOAD(cuMemcpyHtoDAsync_v2) LOAD(cuMemcpyDtoHAsync_v2)
 LOAD(cuStreamCreate) LOAD(cuStreamDestroy_v2) LOAD(cuStreamWaitEvent) LOAD(cuEventCreate) LOAD(cuEventRecord) LOAD(cuEventQuery) LOAD(cuEventDestroy_v2)
 LOAD(cuMemsetD8Async) LOAD(cuMemsetD32Async) LOAD(cuDeviceGetAttribute) LOAD(cuEventElapsedTime) LOAD(cuMemcpyDtoDAsync_v2)
 LOAD(cuModuleLoadData) LOAD(cuModuleGetFunction) LOAD(cuModuleUnload) LOAD(cuLaunchKernel)
#undef LOAD
  check(cuInit_p(0));CUdevice dev;check(cuDeviceGet_p(&dev,0));check(cuCtxGetCurrent_p(&previous));check(cuCtxCreate_v2_p(&context,0,dev));startupSeconds=startLap(start);
 }catch(...){if(context)cuCtxDestroy_v2_p(context);if(library)FreeLibrary(library);throw;}}
 // Blocking is restricted to startup, shutdown or aborting a failed epoch.
 void drain(){thread();++synchronizations;check(cuCtxSynchronize_p());}
 ~Driver(){if(context){cuCtxDestroy_v2_p(context);if(previous)cuCtxSetCurrent_p(previous);}if(library)FreeLibrary(library);}
};
// What the lane is allocating, for the Terminal's memory lines (Buffer): named where a large allocation is made.
inline thread_local const char* allocating=nullptr;
struct Allocating {const char* before;explicit Allocating(const char* what):before(allocating){allocating=what;}~Allocating(){allocating=before;}
 Allocating(const Allocating&)=delete;};
struct Buffer {
 Driver& d;size_t bytes;bool pinned;CUdeviceptr pointer=0;
 Buffer(Driver& owner,size_t n,bool host=false):d(owner),bytes(n),pinned(host){d.thread();cpu::need(n>0,"Empty allocation");auto& used=host?d.pinnedBytes:d.deviceBytes;size_t cap=host?d.pinnedCap:d.deviceCap;if(used>cap||n>cap-used)cpu::need(false,std::string("GPU staging/residency budget: ")+(host?"pinned host":"device")+" memory, "+std::to_string(n>>20)+" MB asked with "+std::to_string(used>>20)+" MB of "+std::to_string(cap>>20)+" MB in use");if(host){void* p=nullptr;Driver::check(d.cuMemAllocHost_v2_p(&p,n));pointer=reinterpret_cast<CUdeviceptr>(p);}else Driver::check(d.cuMemAlloc_v2_p(&pointer,n));used+=n;
  // Each allocation of 256 MB or more said in the MoonRay Terminal, what it is for and what the lane then holds
  // (XPU2_SESSION_LOG_ALLOCATIONS=N: from N MB; 0: none).
  static const size_t logAt=[]{const char* v=std::getenv("XPU2_SESSION_LOG_ALLOCATIONS");const int mb=v?std::atoi(v):256;return mb>0?size_t(mb)<<20:~size_t(0);}();
  if(n>=logAt){std::printf("MoonRay XPU2: %s %zu MB for %s (XPU2 holds %zu MB of the card, %zu MB of pinned RAM)\n",host?"pinned RAM":"card memory",n>>20,
                           allocating?allocating:"the scene",d.deviceBytes>>20,d.pinnedBytes>>20);std::fflush(stdout);}++d.allocations;d.peakPinned=std::max(d.peakPinned,d.pinnedBytes);d.peakDevice=std::max(d.peakDevice,d.deviceBytes);}
 void* host()const{return reinterpret_cast<void*>(pointer);}
 void release(){if(pointer){if(pinned)d.cuMemFreeHost_p(host());else d.cuMemFree_v2_p(pointer);(pinned?d.pinnedBytes:d.deviceBytes)-=bytes;pointer=0;++d.frees;}}
 ~Buffer(){release();}
 Buffer(const Buffer&)=delete;
};
using B=std::shared_ptr<Buffer>;
struct Event {Driver& d;CUevent value=nullptr;explicit Event(Driver& v):d(v){Driver::check(d.cuEventCreate_p(&value,CU_EVENT_DISABLE_TIMING));}~Event(){release();}void release(){if(value){d.cuEventDestroy_v2_p(value);value=nullptr;}}bool ready(){if(!value)return false;auto e=d.cuEventQuery_p(value);if(e==CUDA_ERROR_NOT_READY)return false;Driver::check(e);return true;}};
struct Gas {std::shared_ptr<const cpu::MeshData> mesh;B vertices,indices,bounds,storage;bool precise=false;OptixTraversableHandle handle=0;};
struct Top {std::vector<std::shared_ptr<Gas>> meshes;B instances,storage,geometry;OptixTraversableHandle handle=0;};
struct TextureOptions {size_t texelBudget=16ull<<20;bool includeFine=false,refillFine=false;};
struct ResidentTexture {std::shared_ptr<texture::File> file;B pixels,pages,image;size_t bytes=0,residentPages=0,missingPages=0;};
struct Scene {
 std::shared_ptr<const int> owner;cpu::PlanView plan;std::shared_ptr<Top> top;B material,config;std::vector<surface::Lambert> colors;
 std::vector<B> staging,scratch;std::vector<std::shared_ptr<Gas>> preparing;std::shared_ptr<Event> fence;unsigned built=0,reused=0;bool materialReused=false;B textureMaps;std::vector<std::shared_ptr<ResidentTexture>> textures;size_t textureBudget=0,textureBytes=0,texturePages=0,missingTexturePages=0;unsigned texturesReused=0;
 bool includeFine=false,refillFine=false;std::shared_ptr<Scene> fineReplacement;
 bool ready(){return fence&&fence->ready();} // Keep staging alive until quiescent collection.
};
using SceneView=std::shared_ptr<Scene>;
struct Flight {
 CUstream upload=nullptr,trace=nullptr,download=nullptr;std::shared_ptr<Event> uploaded,traced,done;
 B params,output,hostParams,hostOutput,continuation,hostContinuation;SceneView scene;bool busy=false,retained=false,migrating=false;mixed::Job job{};
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {char header[OPTIX_SBT_RECORD_HEADER_SIZE];};
class Lane {
 OptixDeviceContext optix_=nullptr;OptixModule module_=nullptr;OptixPipeline pipeline_=nullptr;OptixProgramGroup groups_[4]{};
 CUstream build_=nullptr;OptixShaderBindingTable sbt_{};B records_;
 std::shared_ptr<const int> token_=std::make_shared<const int>(0);
 std::vector<std::weak_ptr<Buffer>> allocations_;std::vector<std::weak_ptr<Event>> events_;
 std::array<Flight,4> flights_;std::vector<SceneView> retained_;bool poisoned_=false;
 B buffer(size_t bytes,bool host=false){auto b=std::make_shared<Buffer>(driver,bytes,host);allocations_.push_back(b);return b;}
 std::shared_ptr<Event> event(){auto e=std::make_shared<Event>(driver);events_.push_back(e);return e;}
 B upload(const void* data,size_t bytes,Scene& scene){auto h=buffer(bytes,true),d=buffer(bytes);std::memcpy(h->host(),data,bytes);scene.staging.push_back(h);Driver::check(driver.cuMemcpyHtoDAsync_v2_p(d->pointer,h->host(),bytes,build_));return d;}
 std::pair<B,OptixTraversableHandle> build(const OptixBuildInput& input,Scene& scene){OptixAccelBuildOptions options{};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;options.operation=OPTIX_BUILD_OPERATION_BUILD;OptixAccelBufferSizes sizes{};ox(optixAccelComputeMemoryUsage(optix_,&options,&input,1,&sizes));auto temporary=buffer(sizes.tempSizeInBytes),output=buffer(sizes.outputSizeInBytes);scene.scratch.push_back(temporary);OptixTraversableHandle handle=0;ox(optixAccelBuild(optix_,build_,&options,&input,1,temporary->pointer,temporary->bytes,output->pointer,output->bytes,&handle,nullptr,0));return {output,handle};}
 // OptiX compilation tasks are explicitly thread-safe; CUDA scene/flight work
 // remains confined to its feeder. All tasks finish before the module is linked
 // or destroyed, including task failures and partial pool creation.
 void compile(const std::string& ptx,const OptixModuleCompileOptions& mo,const OptixPipelineCompileOptions& po,unsigned threads,char* log,size_t* length){
  cpu::need(threads>0&&threads<=12,"Compiler worker bound");
  if(threads==1){ox(optixModuleCreateFromPTX(optix_,&mo,&po,ptx.data(),ptx.size(),log,length,&module_));return;}
  OptixTask first=nullptr;ox(optixModuleCreateFromPTXWithTasks(optix_,&mo,&po,ptx.data(),ptx.size(),log,length,&module_,&first));
  std::mutex mutex;std::condition_variable wake;std::deque<OptixTask> tasks;if(first)tasks.push_back(first);unsigned active=0;bool done=!first;OptixResult error=OPTIX_SUCCESS;
  auto work=[&]{for(;;){OptixTask task;{std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return done||!tasks.empty();});if(done)return;task=tasks.front();tasks.pop_front();++active;}
   OptixTask additional[12]{};unsigned count=0;auto status=optixTaskExecute(task,additional,12,&count);
   {std::lock_guard<std::mutex> lock(mutex);--active;if(status!=OPTIX_SUCCESS&&error==OPTIX_SUCCESS)error=status;
    if(error==OPTIX_SUCCESS){try{for(unsigned i=0;i<count;++i)tasks.push_back(additional[i]);}catch(...){error=OPTIX_ERROR_HOST_OUT_OF_MEMORY;tasks.clear();}}else tasks.clear();
    if(tasks.empty()&&active==0)done=true;wake.notify_all();}
  }};
  std::vector<std::thread> workers;
  try{for(unsigned i=1;i<threads;++i)workers.emplace_back(work);work();}
  catch(...){ {std::lock_guard<std::mutex> lock(mutex);done=true;wake.notify_all();}for(auto& w:workers)w.join();throw;}
  for(auto& w:workers)w.join();ox(error);OptixModuleCompileState state;ox(optixModuleGetCompilationState(module_,&state));cpu::need(state==OPTIX_MODULE_COMPILE_STATE_COMPLETED,"GPU compiler did not complete");
 }
 void cleanup()noexcept{
  if(driver.context){try{driver.drain();}catch(...){}}
  for(auto& f:flights_){f.scene.reset();f.params.reset();f.output.reset();f.hostParams.reset();f.hostOutput.reset();f.continuation.reset();f.hostContinuation.reset();f.uploaded.reset();f.traced.reset();f.done.reset();if(f.upload)driver.cuStreamDestroy_v2_p(f.upload);if(f.trace)driver.cuStreamDestroy_v2_p(f.trace);if(f.download)driver.cuStreamDestroy_v2_p(f.download);f.upload=f.trace=f.download=nullptr;}
  retained_.clear();records_.reset();
  // Expired lane views may survive shutdown, but no CUDA handle may outlive Driver.
  for(auto& weak:events_)if(auto e=weak.lock())e->release();
  for(auto& weak:allocations_)if(auto b=weak.lock())b->release();
  events_.clear();allocations_.clear();if(build_)driver.cuStreamDestroy_v2_p(build_);build_=nullptr;
  if(pipeline_)optixPipelineDestroy(pipeline_);pipeline_=nullptr;for(auto& g:groups_)if(g){optixProgramGroupDestroy(g);g=nullptr;}if(module_)optixModuleDestroy(module_);module_=nullptr;if(optix_)optixDeviceContextDestroy(optix_);optix_=nullptr;
 }
public:
 Driver driver; // Declared resources above are explicitly retired before this.
 StartupTimes startup;
 explicit Lane(const std::string& ptx,bool validation=false,StartupOptions startupOptions={}){auto stamp=StartClock::now();startup.driver=driver.startupSeconds;try{
  static const OptixResult initialized=optixInit();ox(initialized);OptixDeviceContextOptions context{};context.validationMode=validation?OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL:OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;ox(optixDeviceContextCreate(driver.context,&context,&optix_));
  if(!startupOptions.cache.empty())ox(optixDeviceContextSetCacheLocation(optix_,startupOptions.cache.c_str()));
  startup.context=startLap(stamp);OptixModuleCompileOptions mo{};mo.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;mo.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;mo.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
  OptixPipelineCompileOptions po{};po.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;po.numPayloadValues=5;po.numAttributeValues=2;po.pipelineLaunchParamsVariableName="params";po.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE|OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  char log[8192]{};size_t len=sizeof(log);try{compile(ptx,mo,po,startupOptions.compileThreads,log,&len);}catch(const std::exception& e){throw std::runtime_error(std::string(e.what())+" / OptiX module: "+std::string(log,std::find(log,log+std::min(len,sizeof(log)),char(0))));}
  startup.module=startLap(stamp);for(int i=0;i<4;++i){OptixProgramGroupDesc desc{};if(i==0){desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;desc.raygen.module=module_;desc.raygen.entryFunctionName="__raygen__render";}else if(i<3){desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;desc.miss.module=module_;desc.miss.entryFunctionName=i==1?"__miss__primary":"__miss__shadow";}else{desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;desc.hitgroup.moduleCH=module_;desc.hitgroup.entryFunctionNameCH="__closesthit__surface";desc.hitgroup.moduleIS=module_;desc.hitgroup.entryFunctionNameIS="__intersection__triangle";}OptixProgramGroupOptions options{};len=sizeof(log);ox(optixProgramGroupCreate(optix_,&desc,1,&options,log,&len,&groups_[i]));}
  startup.groups=startLap(stamp);OptixPipelineLinkOptions link{};link.maxTraceDepth=1;link.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;len=sizeof(log);ox(optixPipelineCreate(optix_,&po,&link,groups_,4,log,&len,&pipeline_));OptixStackSizes stacks{};for(auto group:groups_)ox(optixUtilAccumulateStackSizes(group,&stacks));unsigned a,b,c;ox(optixUtilComputeStackSizes(&stacks,1,0,0,&a,&b,&c));ox(optixPipelineSetStackSize(pipeline_,a,b,c,2));
  startup.pipeline=startLap(stamp);Driver::check(driver.cuStreamCreate_p(&build_,CU_STREAM_NON_BLOCKING));auto h=buffer(sizeof(Record)*4,true);auto* rec=static_cast<Record*>(h->host());for(int i=0;i<4;++i)ox(optixSbtRecordPackHeader(groups_[i],&rec[i]));records_=buffer(sizeof(Record)*4);Driver::check(driver.cuMemcpyHtoDAsync_v2_p(records_->pointer,h->host(),h->bytes,build_));
  sbt_.raygenRecord=records_->pointer;sbt_.missRecordBase=records_->pointer+sizeof(Record);sbt_.missRecordStrideInBytes=sizeof(Record);sbt_.missRecordCount=2;sbt_.hitgroupRecordBase=records_->pointer+3*sizeof(Record);sbt_.hitgroupRecordStrideInBytes=sizeof(Record);sbt_.hitgroupRecordCount=1;
  driver.drain(); // Startup only; the local SBT staging allocation can now retire.
  for(auto& f:flights_){Driver::check(driver.cuStreamCreate_p(&f.upload,CU_STREAM_NON_BLOCKING));Driver::check(driver.cuStreamCreate_p(&f.trace,CU_STREAM_NON_BLOCKING));Driver::check(driver.cuStreamCreate_p(&f.download,CU_STREAM_NON_BLOCKING));f.uploaded=event();f.traced=event();f.done=event();f.params=buffer(sizeof(mixed::Params));f.output=buffer(sizeof(mixed::Result));f.hostParams=buffer(sizeof(mixed::Params),true);f.hostOutput=buffer(sizeof(mixed::Result),true);f.continuation=buffer(sizeof(resume::Batch));f.hostContinuation=buffer(sizeof(resume::Batch),true);}
 startup.buffers=startLap(stamp);
 }catch(...){cleanup();throw;}}
 ~Lane(){cleanup();}Lane(const Lane&)=delete;
 SceneView prepare(cpu::PlanView plan,SceneView old={},TextureOptions textureOptions={}){
  driver.thread();cpu::need(!poisoned_&&retained_.size()<8,"GPU scene preparation backpressure");cpu::need(plan&&(!old||(old->owner==token_&&old->config&&old->config->pointer)),"Invalid GPU preparation input");auto scene=std::make_shared<Scene>();scene->owner=token_;scene->plan=plan;scene->includeFine=textureOptions.includeFine;scene->refillFine=textureOptions.refillFine;retained_.push_back(scene);
  try{
   std::vector<std::shared_ptr<Gas>> meshes;
   for(auto& mesh:plan->meshes){std::shared_ptr<Gas> gas;if(old)for(auto& prior:old->top->meshes)if(prior->mesh==mesh.data&&prior->precise==cpu::isSpatial(plan->domain)){gas=prior;break;}
    if(gas)++scene->reused;else{gas=std::make_shared<Gas>();scene->preparing.push_back(gas);gas->mesh=mesh.data;gas->precise=cpu::isSpatial(plan->domain);gas->vertices=upload(mesh.data->positions.data(),mesh.data->positions.size()*sizeof(surface::V3),*scene);gas->indices=upload(mesh.data->triangles.data(),mesh.data->triangles.size()*sizeof(cpu::Tri),*scene);
     OptixBuildInput input{};unsigned flag=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
     if(gas->precise){std::vector<OptixAabb> bounds;bounds.reserve(mesh.data->triangles.size());for(auto t:mesh.data->triangles){auto a=mesh.data->positions[t.a],b=mesh.data->positions[t.b],c=mesh.data->positions[t.c];bounds.push_back({std::nextafter(std::min({a.x,b.x,c.x}),-INFINITY),std::nextafter(std::min({a.y,b.y,c.y}),-INFINITY),std::nextafter(std::min({a.z,b.z,c.z}),-INFINITY),std::nextafter(std::max({a.x,b.x,c.x}),INFINITY),std::nextafter(std::max({a.y,b.y,c.y}),INFINITY),std::nextafter(std::max({a.z,b.z,c.z}),INFINITY)});}gas->bounds=upload(bounds.data(),bounds.size()*sizeof(OptixAabb),*scene);input.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;auto& custom=input.customPrimitiveArray;custom.aabbBuffers=&gas->bounds->pointer;custom.numPrimitives=unsigned(bounds.size());custom.strideInBytes=sizeof(OptixAabb);custom.flags=&flag;custom.numSbtRecords=1;
     }else{input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;auto& tri=input.triangleArray;tri.vertexBuffers=&gas->vertices->pointer;tri.numVertices=unsigned(mesh.data->positions.size());tri.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;tri.vertexStrideInBytes=sizeof(surface::V3);tri.indexBuffer=gas->indices->pointer;tri.numIndexTriplets=unsigned(mesh.data->triangles.size());tri.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3;tri.indexStrideInBytes=sizeof(cpu::Tri);tri.flags=&flag;tri.numSbtRecords=1;}auto built=build(input,*scene);gas->storage=built.first;gas->handle=built.second;++scene->built;}
    meshes.push_back(gas);scene->colors.push_back(plan->materials.at(mesh.material));
   }
   bool sameTop=old&&old->top->meshes==meshes;if(sameTop)scene->top=old->top;else{
    auto top=std::make_shared<Top>();top->meshes=std::move(meshes);scene->top=top;std::vector<OptixInstance> instances(top->meshes.size());for(unsigned i=0;i<instances.size();++i){auto& inst=instances[i];inst.transform[0]=inst.transform[5]=inst.transform[10]=1;inst.instanceId=i;inst.visibilityMask=255;inst.traversableHandle=top->meshes[i]->handle;}
    std::vector<mixed::Geometry> geometry;for(const auto& g:top->meshes)geometry.push_back({reinterpret_cast<const surface::V3*>(g->vertices->pointer),reinterpret_cast<const surface::Triangle*>(g->indices->pointer),unsigned(g->mesh->positions.size()),unsigned(g->mesh->triangles.size())});top->geometry=upload(geometry.data(),geometry.size()*sizeof(mixed::Geometry),*scene);
    top->instances=upload(instances.data(),instances.size()*sizeof(OptixInstance),*scene);OptixBuildInput input{};input.type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;input.instanceArray.instances=top->instances->pointer;input.instanceArray.numInstances=unsigned(instances.size());auto built=build(input,*scene);top->storage=built.first;top->handle=built.second;
   }
   bool sameMaterials=old&&old->colors.size()==scene->colors.size();if(sameMaterials)for(size_t i=0;i<scene->colors.size();++i){auto a=old->colors[i].albedo,b=scene->colors[i].albedo;auto ea=old->colors[i].emission,eb=scene->colors[i].emission;sameMaterials&=ea.x==eb.x&&ea.y==eb.y&&ea.z==eb.z;sameMaterials&=a.x==b.x&&a.y==b.y&&a.z==b.z;}
   if(sameMaterials){scene->material=old->material;scene->materialReused=true;}else scene->material=upload(scene->colors.data(),scene->colors.size()*sizeof(surface::Lambert),*scene);
   if(cpu::gpuTextures(plan->domain)){
    cpu::need(textureOptions.texelBudget>=128&&textureOptions.texelBudget<=64ull<<20,"Texture residency budget");scene->textureBudget=textureOptions.texelBudget;
    std::vector<std::shared_ptr<texture::File>> files;for(auto* bindings:{&plan->diffuseMaps,&plan->emissionMaps})for(auto& pair:*bindings)if(std::find(files.begin(),files.end(),pair.second.file)==files.end())files.push_back(pair.second.file);
    // Reserve every coarsest page before spending on finer pages. Level zero is
    // excluded by default. Opt-in refill publishes a new immutable scene view.
    size_t remaining=textureOptions.texelBudget;cpu::need(files.size()*sizeof(surface::V3)<=remaining,"Pinned coarse mip budget");remaining-=files.size()*sizeof(surface::V3);
    for(auto file:files){cpu::need(bool(file->pyramid),"Missing host mip pyramid");const auto& host=file->pyramid->image;std::shared_ptr<ResidentTexture> texture;
     if(old&&old->includeFine==scene->includeFine&&old->textureBudget==scene->textureBudget&&old->plan->diffuseMaps.size()==plan->diffuseMaps.size()&&old->plan->emissionMaps.size()==plan->emissionMaps.size()){
      bool sameFiles=old->textures.size()==files.size();if(sameFiles)for(size_t n=0;n<files.size();++n)sameFiles&=old->textures[n]->file==files[n];
      if(sameFiles)for(auto prior:old->textures)if(prior->file==file){texture=prior;++scene->texturesReused;break;}
     }
     if(texture){if(host.count==1&&!scene->includeFine){cpu::need(texture->bytes==0,"Level-zero residency forbidden");remaining+=sizeof(surface::V3);}else{cpu::need(texture->bytes>=sizeof(surface::V3)&&texture->bytes-sizeof(surface::V3)<=remaining,"Reused texture budget");remaining-=texture->bytes-sizeof(surface::V3);}}
     else {texture=std::make_shared<ResidentTexture>();texture->file=file;std::vector<surface::V3> packed;std::vector<size_t> offsets;tex::Image descriptor{};descriptor.count=host.count;size_t pageStarts[tex::MaxLevels]{};
      for(unsigned level=0;level<host.count;++level){auto m=host.levels[level];cpu::need(m.format==0&&!m.load,"This lane reads textures whole, in floats (XPU2_SESSION_FLOAT_TEXTURES=1, XPU2_SESSION_EAGER_TEXTURES=1)");descriptor.levels[level]={m.width,m.height,nullptr};pageStarts[level]=offsets.size();offsets.resize(offsets.size()+size_t((m.width+31)/32)*((m.height+31)/32),size_t(-1));}
      for(int level=int(host.count)-1;level>=(scene->includeFine?0:1);--level){auto m=host.levels[level];unsigned nx=(m.width+31)/32,ny=(m.height+31)/32;
       for(unsigned y=0;y<ny;++y)for(unsigned x=0;x<nx;++x){size_t count=size_t(std::min(32u,m.width-x*32))*std::min(32u,m.height-y*32),bytes=count*sizeof(surface::V3);bool coarse=level==int(host.count)-1;
        if(!coarse&&bytes>remaining){++texture->missingPages;continue;}if(!coarse)remaining-=bytes;offsets[pageStarts[level]+y*nx+x]=packed.size();auto* first=m.pages[y*nx+x];packed.insert(packed.end(),first,first+count);++texture->residentPages;
       }
      }
      // Default mode leaves a 1x1 source on CPU; opt-in fine residency can admit it.
      if(host.count==1&&!scene->includeFine){remaining+=sizeof(surface::V3);texture->bytes=0;}else{texture->bytes=packed.size()*sizeof(surface::V3);texture->pixels=upload(packed.data(),texture->bytes,*scene);}
      texture->missingPages=std::count(offsets.begin(),offsets.end(),size_t(-1));
      std::vector<const surface::V3*> addresses(offsets.size(),nullptr);for(size_t n=0;n<offsets.size();++n)if(offsets[n]!=size_t(-1))addresses[n]=reinterpret_cast<const surface::V3*>(texture->pixels->pointer)+offsets[n];
      texture->pages=upload(addresses.data(),addresses.size()*sizeof(addresses[0]),*scene);for(unsigned level=0;level<host.count;++level)descriptor.levels[level].pages=reinterpret_cast<const surface::V3* const*>(texture->pages->pointer)+pageStarts[level];
      texture->image=upload(&descriptor,sizeof(descriptor),*scene);
     }
     scene->textureBytes+=texture->bytes;scene->texturePages+=texture->residentPages;scene->missingTexturePages+=texture->missingPages;scene->textures.push_back(texture);
    }
    std::vector<tex::Maps> maps;for(const auto& mesh:plan->meshes){tex::Maps pair;auto fill=[&](tex::Binding& b,const auto& bindings){auto it=bindings.find(mesh.material);if(it==bindings.end())return;b=it->second.descriptor();for(auto texture:scene->textures)if(texture->file==it->second.file){b.image=reinterpret_cast<const tex::Image*>(texture->image->pointer);return;}throw std::runtime_error("Unmapped texture residency");};fill(pair.diffuse,plan->diffuseMaps);fill(pair.emission,plan->emissionMaps);maps.push_back(pair);}
    scene->textureMaps=upload(maps.data(),maps.size()*sizeof(tex::Maps),*scene);
   }
   auto c=mixed::config(plan);scene->config=upload(&c,sizeof(c),*scene);scene->fence=event();Driver::check(driver.cuEventRecord_p(scene->fence->value,build_));return scene;
  }catch(...){poisoned_=true;driver.drain();throw;}
 }

 // One bounded refill per immutable scene version. Uploads run on the build
 // stream; old flights retain old descriptors until an exclusively owned pause.
 SceneView requestFine(SceneView old){driver.thread();cpu::need(old&&old->owner==token_&&old->refillFine&&!old->includeFine,"Fine refill not enabled");if(!old->fineReplacement)old->fineReplacement=prepare(old->plan,old,TextureOptions{old->textureBudget,true,false});return old->fineReplacement;}
 void finishFine(SceneView scene){driver.thread();cpu::need(scene&&scene->owner==token_&&scene->includeFine&&scene->ready(),"Fine upload incomplete");scene->staging.clear();scene->scratch.clear();scene->preparing.clear();}
 void rebindResume(unsigned slot,SceneView scene){driver.thread();cpu::need(slot<flights_.size()&&scene&&scene->owner==token_&&scene->ready(),"Invalid fine resume");auto& f=flights_[slot];cpu::need(f.retained&&!f.busy&&!f.migrating&&f.scene&&f.scene->plan==scene->plan,"Unsafe resume rebind");f.scene=scene;}

 // Resumable flights keep their device path pool and scene lease while paused.
 // Only launch parameters travel on resume; path state stays in device memory.
 unsigned held()const{unsigned n=0;for(auto& f:flights_)n+=f.retained;return n;}
 void resumeLaunch(unsigned slot,unsigned steps,bool reset){
  driver.thread();cpu::need(!poisoned_&&slot<flights_.size()&&steps>0&&steps<=64,"Invalid resume launch");auto& f=flights_[slot];cpu::need(f.retained&&!f.busy&&!f.migrating&&f.scene,"Unowned resume slot");f.busy=true;
  try{mixed::Params p{reinterpret_cast<const mixed::Config*>(f.scene->config->pointer),reinterpret_cast<const surface::Lambert*>(f.scene->material->pointer),f.scene->top->handle,reinterpret_cast<mixed::Result*>(f.output->pointer),f.job,reinterpret_cast<resume::Batch*>(f.continuation->pointer),steps,reset?1u:0u,reinterpret_cast<const mixed::Geometry*>(f.scene->top->geometry->pointer),f.scene->textureMaps?reinterpret_cast<const tex::Maps*>(f.scene->textureMaps->pointer):nullptr};std::memcpy(f.hostParams->host(),&p,sizeof(p));
   Driver::check(driver.cuMemcpyHtoDAsync_v2_p(f.params->pointer,f.hostParams->host(),sizeof(p),f.upload));Driver::check(driver.cuEventRecord_p(f.uploaded->value,f.upload));Driver::check(driver.cuStreamWaitEvent_p(f.trace,f.uploaded->value,0));
   ox(optixLaunch(pipeline_,f.trace,f.params->pointer,sizeof(p),&sbt_,mixed::Capacity,1,1));Driver::check(driver.cuEventRecord_p(f.traced->value,f.trace));Driver::check(driver.cuStreamWaitEvent_p(f.download,f.traced->value,0));
   Driver::check(driver.cuMemcpyDtoHAsync_v2_p(f.hostOutput->host(),f.output->pointer,sizeof(mixed::Result),f.download));Driver::check(driver.cuEventRecord_p(f.done->value,f.download));
  }catch(...){poisoned_=true;throw;}
 }
 int submitResume(SceneView scene,const mixed::Job& job,unsigned steps){
  driver.thread();cpu::need(!poisoned_&&scene&&scene->owner==token_,"Foreign resume scene");cpu::need((scene->plan->diffuseMaps.empty()&&scene->plan->emissionMaps.empty())||cpu::gpuTextures(scene->plan->domain),"GPU texture domain unsupported");cpu::need(mixed::same(job,mixed::config(scene->plan)),"Resume epoch mismatch");if(!scene->ready())return -1;
  for(unsigned i=0;i<flights_.size();++i){auto& f=flights_[i];if(f.busy||f.retained)continue;f.scene=scene;f.job=job;f.retained=true;resumeLaunch(i,steps,true);return int(i);}return -1;
 }
 bool pollResume(unsigned slot,mixed::Result& result){driver.thread();cpu::need(!poisoned_&&slot<flights_.size(),"Invalid resume poll");auto& f=flights_[slot];cpu::need(f.retained&&!f.migrating,"Resume slot lost or migrating");if(!f.busy||!f.done->ready())return false;std::memcpy(&result,f.hostOutput->host(),sizeof(result));f.busy=false;return true;}
 // Only a completed, exclusively owned GPU slice can export its continuation.
 // Copies are asynchronous into four preallocated pinned buffers. The old GPU
 // owner remains retained until the CPU has validated the whole batch.
 void beginMigration(unsigned slot){driver.thread();cpu::need(slot<flights_.size(),"Migration slot");auto& f=flights_[slot];cpu::need(f.retained&&!f.busy&&!f.migrating,"Migration ownership");f.busy=f.migrating=true;
  try{Driver::check(driver.cuMemcpyDtoHAsync_v2_p(f.hostContinuation->host(),f.continuation->pointer,sizeof(resume::Batch),f.download));Driver::check(driver.cuEventRecord_p(f.done->value,f.download));}catch(...){poisoned_=true;throw;}
 }
 bool pollMigration(unsigned slot,resume::Batch& out){driver.thread();cpu::need(slot<flights_.size(),"Migration poll slot");auto& f=flights_[slot];cpu::need(f.retained&&f.busy&&f.migrating,"Migration poll ownership");if(!f.done->ready())return false;std::memcpy(&out,f.hostContinuation->host(),sizeof(out));f.busy=f.migrating=false;return true;}
 void releaseResume(unsigned slot){driver.thread();cpu::need(slot<flights_.size()&&!flights_[slot].busy,"Resume slot still executing");auto& f=flights_[slot];cpu::need(f.retained,"Duplicate resume release");f.retained=false;f.scene.reset();}
 unsigned busy()const{unsigned n=0;for(auto& f:flights_)n+=f.busy;return n;}
 // Free only at an explicitly quiescent scene boundary, never per render trip.
 void collect(SceneView keep){driver.thread();cpu::need(!busy()&&!held(),"Cannot retire in-flight scenes");for(auto& s:retained_)cpu::need(s->ready(),"Scene still building");retained_.erase(std::remove_if(retained_.begin(),retained_.end(),[&](auto& s){return s!=keep;}),retained_.end());allocations_.erase(std::remove_if(allocations_.begin(),allocations_.end(),[](auto& w){return w.expired();}),allocations_.end());events_.erase(std::remove_if(events_.begin(),events_.end(),[](auto& w){return w.expired();}),events_.end());if(keep){keep->staging.clear();keep->scratch.clear();keep->preparing.clear();}}
 int submit(SceneView scene,const mixed::Job& job){
  driver.thread();cpu::need(!poisoned_&&scene&&scene->owner==token_,"GPU lane poisoned or foreign scene");cpu::need(scene->plan->diffuseMaps.empty()&&scene->plan->emissionMaps.empty(),"GPU textures not qualified; use CPU lane");if(!scene->ready())return -1;cpu::need(mixed::same(job,mixed::config(scene->plan)),"Job epoch mismatch");
  for(unsigned i=0;i<flights_.size();++i){auto& f=flights_[i];if(f.busy||f.retained)continue;f.busy=true;f.scene=scene;f.job=job;
   try{mixed::Params p{reinterpret_cast<const mixed::Config*>(scene->config->pointer),reinterpret_cast<const surface::Lambert*>(scene->material->pointer),scene->top->handle,reinterpret_cast<mixed::Result*>(f.output->pointer),job,nullptr,0,0,reinterpret_cast<const mixed::Geometry*>(scene->top->geometry->pointer),nullptr};std::memcpy(f.hostParams->host(),&p,sizeof(p));
    Driver::check(driver.cuMemcpyHtoDAsync_v2_p(f.params->pointer,f.hostParams->host(),sizeof(p),f.upload));Driver::check(driver.cuEventRecord_p(f.uploaded->value,f.upload));Driver::check(driver.cuStreamWaitEvent_p(f.trace,f.uploaded->value,0));
    ox(optixLaunch(pipeline_,f.trace,f.params->pointer,sizeof(p),&sbt_,mixed::Capacity,1,1));Driver::check(driver.cuEventRecord_p(f.traced->value,f.trace));Driver::check(driver.cuStreamWaitEvent_p(f.download,f.traced->value,0));
    Driver::check(driver.cuMemcpyDtoHAsync_v2_p(f.hostOutput->host(),f.output->pointer,sizeof(mixed::Result),f.download));Driver::check(driver.cuEventRecord_p(f.done->value,f.download));return int(i);
   }catch(...){poisoned_=true;throw;}
  }return -1;
 }
 bool poll(unsigned slot,mixed::Result& result){driver.thread();cpu::need(slot<flights_.size()&&!poisoned_,"Invalid GPU poll");auto& f=flights_[slot];cpu::need(!f.retained,"Continuation requires resume poll");if(!f.busy||!f.done->ready())return false;std::memcpy(&result,f.hostOutput->host(),sizeof(result));f.busy=false;f.scene.reset();return true;}
};
}

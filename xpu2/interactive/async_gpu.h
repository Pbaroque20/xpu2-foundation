// The asynchronous renderer's GPU lane, host side (spec §10, §13).
//
// One CUDA stream carries everything in order: chunk uploads into the mailbox
// ring, launches, readbacks of each launch's state, and tile-cache updates.
// Two launches stay queued so the GPU never waits for the host, and the host
// never waits for the GPU: it polls events (ARC-01).
//
// Texture residency follows the memory rule: host RAM holds every level of
// every texture; VRAM holds each texture's mip levels from 1 up, and a bounded
// cache of fine-detail (level-0) tiles loaded on demand from host RAM and
// evicted least recently used. A sample that needs a tile that is not resident
// parks in the GPU's bounded queue, its tile is requested once, and a later
// launch runs it again on the GPU after the tile has landed (MEM-04, MEM-05).
// With partial shading such a sample is finished from the texture's next level
// up instead, kept apart in the provisional film, and rendered again in full by
// the CPU lane or by a later launch once the tile has landed (async_render.h).
#pragma once
#include "async_adaptive.h"
#include "async_gpu_abi.h"
#include <chrono>
#include "gpu_lane.h"
#include "multi_device.h"
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace xpu2::interactive::async {

struct GpuOptions {
    bool profileChain = false;         // time the pass chain's stages on the device (a diagnostic)
    size_t fineBudget = 0;  // VRAM for fine-detail tiles (0: from the card's free memory and the scene's needs)
    U32 launchThreads = 1u << 16;      // fixed grid size of every render launch
    U32 maxChunksPerLaunch = 16384;
    U32 ringChunks = 1u << 17;         // mailbox ring capacity, in chunks
    // Called as the programs compile: OptiX compile tasks done so far, and those known to remain.
    std::function<void(unsigned done, unsigned waiting)> compileProgress;
    // The program's map module (async_maps.cu), called by the material module's callables, and its light and lobe
    // modules (async_lights.cu, async_lobes.cu) and volume module (async_volumes.cu), called by the path tracer: a
    // program built in modules has them all.
    std::string mapsPtx, lightsPtx, lobesPtx, volumesPtx;
    // The volume module again with the volumes' map program in its march (XPU2_VOLUME_MAPS_MODULE): bound in place of
    // the plain one only for a scene with volumes whose maps the card runs.
    std::string volumeMapsPtx;
    // The DwaBase modules (async_dwa.cu, async_dwalobes.cu), called by the material module's callables.
    std::string dwaPtx, dwaLobesPtx;
    // The extras module (async_extras.cu): the camera's ray and the AOVs' accumulation, called by the path tracer.
    std::string extrasPtx;
    int device = 0;  // the graphics card, CUDA's numbering (multi_device.h)
    int lane = 0;    // its place among the lanes, the first 0 (multi_device.h's simulation switches)
    int node = -1;   // the NUMA node nearest the card, -1 where there is one or it is not known (gpu::cardNode)
};


// What the host learns when a launch has finished.
struct Completion {
    U64 chunks = 0, items = 0, parkedRun = 0, refinesRun = 0;
    U64 samples = 0;  // the samples of its chunks, as the host counted them at launch
    U32 errors = 0;
    double deviceSeconds = 0;  // the launch's own duration, timed on the GPU
};

// A CUDA event that records time, for measuring launches on the device itself:
// the host's view of a launch includes however late it noticed the completion.
struct TimingEvent {
    gpu::Driver& d;
    CUevent value = nullptr;
    explicit TimingEvent(gpu::Driver& driver) : d(driver) { gpu::Driver::check(d.cuEventCreate_p(&value, CU_EVENT_DEFAULT)); }
    ~TimingEvent() { if (value) d.cuEventDestroy_v2_p(value); }
    TimingEvent(const TimingEvent&) = delete;
};

class GpuLane {
public:
    gpu::Driver driver;
    // The GPU program's modules: the path tracer's, then its callables' (material, maps, lights, lobes, volumes, DwaBase
    // resolve, DwaBase lobes, extras), and the mapped volume module (VolumeMapsModule, bound in place of the volume
    // module where a scene needs it).
    static constexpr int ModuleCount = 10;
    static constexpr int VolumeModule = 5, VolumeMapsModule = 9;

    // ptx renders every frame (async_gpu.cu: one program for uniform and adaptive frames); materialPtx, its material
    // module (async_material.cu: the material callables; none: the program shades materials itself);
    // chainPtx holds the adaptive pass chain's CUDA kernels (async_adaptive.cu;
    // none: adaptive passes are made on the host).
    explicit GpuLane(const std::string& ptx, const std::string& materialPtx = {}, const std::string& chainPtx = {},
                     GpuOptions options = {})
        : driver(options.device), options_(options) {
        driver.deviceCap = 20ull << 30;
        {
            // The allocation cap: the card's own memory where it has more than that.
            size_t freeBytes = 0, totalBytes = 0;
            if (memoryInfo(freeBytes, totalBytes) && totalBytes > driver.deviceCap) driver.deviceCap = totalBytes;
        }
        driver.pinnedCap = 2ull << 30;
        {
            // Pinned (page-locked) host memory, the frame's readbacks and pictures: an eighth of the machine's, between
            // 2 and 8 GB (a fixed 2 GB refused his Dose preview: 1600 x 2000 with 26 outputs). Memory Windows cannot page
            // out: much more of it leaves the machine nothing to page to (a quarter, 16 GB, was the cap when a heavy
            // scene froze his machine).
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof status;
            if (GlobalMemoryStatusEx(&status))
                driver.pinnedCap = std::clamp<size_t>(size_t(status.ullTotalPhys / 8), 2ull << 30, 8ull << 30);
        }
        try {
            ox(optixInit());
            OptixDeviceContextOptions context{};
            ox(optixDeviceContextCreate(driver.context, &context, &optix_));
            OptixModule modules[ModuleCount]{};
            const std::string* sources[ModuleCount] = {&ptx, &materialPtx, &options_.mapsPtx, &options_.lightsPtx, &options_.lobesPtx,
                                                       &options_.volumesPtx, &options_.dwaPtx, &options_.dwaLobesPtx, &options_.extrasPtx,
                                                       &options_.volumeMapsPtx};
            compileModules(sources, modules);
            programs_[0].create(*this, modules);
            program_ = &programs_[0];
            single_ = true;
            if (!chainPtx.empty()) {
                gpu::Driver::check(driver.cuModuleLoadData_p(&chain_.module, chainPtx.c_str()));
                gpu::Driver::check(driver.cuModuleGetFunction_p(&chain_.errors, chain_.module, "chainErrors"));
                gpu::Driver::check(driver.cuModuleGetFunction_p(&chain_.trees, chain_.module, "chainTrees"));
                gpu::Driver::check(driver.cuModuleGetFunction_p(&chain_.masks, chain_.module, "chainMasks"));
                gpu::Driver::check(driver.cuModuleGetFunction_p(&chain_.compact, chain_.module, "chainCompact"));
            }
            gpu::Driver::check(driver.cuStreamCreate_p(&stream_, CU_STREAM_NON_BLOCKING));

            for (auto& p : params_) {
                p.host = pinned(sizeof(LaunchParams));
                p.device = device(sizeof(LaunchParams));
                p.done = std::make_shared<gpu::Event>(driver);
            }
            for (auto& r : readbacks_) r.state = pinned(sizeof(DeviceState));
            ring_ = device(size_t(options_.ringChunks) * sizeof(Chunk));
            state_ = device(sizeof(DeviceState));
            for (auto& s : staging_) s = pinned(size_t(options_.maxChunksPerLaunch) * sizeof(Chunk));
            // Partial shading: each launch thread's mark, the provisional samples' ring (pinned: the host reads it
            // after a launch) and each launch slot's provisional samples to render again (pinned: the device reads them).
            // The launch width by the card: a power of two under the threads its multiprocessors hold at once (2^16 on
            // a 3090), never more than options_.launchThreads, so a card as large or larger launches as before and a
            // smaller one is as full. prepare narrows it further where the card has no room for its scratch.
            {
                CUdevice dev;
                gpu::Driver::check(driver.cuDeviceGet_p(&dev, driver.ordinal));
                int sms = 0, perSm = 0;
                if (driver.cuDeviceGetAttribute_p(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUDA_SUCCESS &&
                    driver.cuDeviceGetAttribute_p(&perSm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev) == CUDA_SUCCESS &&
                    sms > 0 && perSm > 0) {
                    U32 width = std::min(MinLaunchThreads, options_.launchThreads);
                    while (width < options_.launchThreads && U64(width) * 2 < U64(sms) * U64(perSm)) width *= 2;
                    options_.launchThreads = width;
                }
                // XPU2_SESSION_LAUNCH_THREADS caps it (a power of two from MinLaunchThreads), to measure a narrower launch.
                if (const char* cap = std::getenv("XPU2_SESSION_LAUNCH_THREADS"))
                    while (options_.launchThreads > MinLaunchThreads && options_.launchThreads > U32(std::max(0, std::atoi(cap))))
                        options_.launchThreads /= 2;
                cardThreads_ = options_.launchThreads;
            }
            lowFlags_ = device(size_t(options_.launchThreads) * sizeof(U32));
            refines_ = pinned(size_t(refineCapacity_) * sizeof(Refine));
            for (auto& r : refineIn_) r = pinned(size_t(RefineInCapacity) * sizeof(Refine));
            driver.drain();  // startup only
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~GpuLane() {
        for (auto& f : followers_) {  // each in its own context
            f->use();
            f.reset();
        }
        if (driver.context) use();
        cleanup();
    }
    // The multi-device extension (multi_device.h). The lanes of the other graphics cards follow this one: each takes
    // the scene, its edits and each frame's tables as this one does (forward), and its film is added into this one's
    // host copy as that lands; the renderer claims and launches on each itself (async_render.h). Every lane is made
    // and used on the one feeder thread, each call in its own context (use). With no follower nothing here runs.
    void follow(std::unique_ptr<GpuLane> lane) {
        lane->countsWithErrors_ = true;
        followers_.push_back(std::shared_ptr<GpuLane>(std::move(lane)));
        countsWithErrors_ = true;
        use();
    }
    const std::vector<std::shared_ptr<GpuLane>>& followers() const { return followers_; }
    // This lane's context current on the calling thread.
    void use() const { gpu::Driver::check(driver.cuCtxSetCurrent_p(driver.context)); }
    int card() const { return driver.ordinal; }
    int lane() const { return options_.lane; }
    int node() const { return options_.node; }
    GpuLane(const GpuLane&) = delete;

    // Bytes of scratch per launch thread: a Bsdf, the pending rays and the DwaBase parameter slots
    // the scene's material trees need (GpuBackend::bsdf, pending, dwaSlots).
    // A glitter scratch (shade_glitter.h) follows when a material has glitter (glitterOffset).
    static size_t scratchStride(unsigned dwaSlots, bool glitter = false) {
        return glitterOffset(dwaSlots) + (glitter ? (sizeof(shade::GlitterScratch) + 15) & ~size_t(15) : 0);
    }
    static size_t glitterOffset(unsigned dwaSlots) {
        return ((sizeof(shade::Bsdf) + 15) & ~size_t(15)) + ((sizeof(paths::Pending) * paths::MaxPending + 15) & ~size_t(15)) +
               dwaSlots * ((sizeof(shade::DwaParams) + 15) & ~size_t(15));
    }
    // Uploads a scene: acceleration structures, geometry, materials, the
    // resident mip levels and an empty fine-detail cache. Startup work.
    // What the GPU programs do not render yet (the session then renders the scene on the CPU alone), or empty.
    static std::string cpuOnly(const cpu::PlanView& plan) {
        if (!plan->lightMotion.empty() && !motionOnGpu()) return "lights in motion";
        if (!plan->filterLists.empty()) return "light filters";
        if (!plan->shadowLinks.empty()) return "shadow linking";
        if (!plan->lobeLightSetBits.empty()) return "lobe light sets";
        // Emissive volumes light every surface (their emissive regions are light sources, volume_emission.h): a GPU
        // sample that does not cross one would still miss its light, so a glowing volume the GPU does not render (one
        // whose maps read image textures: volume_core.h's mapsOnHost) leaves none to the GPU. The others' regions go
        // to the card (uploadRegions), mapped ones' maps run there too (GpuVolumeMaps).
        if (!plan->emissiveRegions.empty()) {
            if (!volume::GpuEmission) return "emissive volumes";
            const volume::Grids host{plan->volumeGridViews.data(), U32(plan->volumeGridViews.size()), nullptr, 0.0f};
            for (const auto& r : plan->emissiveRegions)
                if (r.volume >= plan->volumes.size() || !volume::gpuRenders(plan->volumes[r.volume], host)) return volume::GpuVolumeMaps ? "emissive volumes whose maps read image textures" : "emissive volumes with mapped emission";
        }
        // Volumes the GPU does not render are stand-ins (standIn): their samples are deferred to the CPU lane.
        return {};
    }
    // A launch thread's records of lights in motion at the sample's time (GpuBackend::movedLight): however many move, the
    // card holds this many a thread.
    static constexpr size_t MovedLightSlots = 16;
    // The narrowest launch prepare makes for a card short of memory.
    static constexpr U32 MinLaunchThreads = 8192;
    // XPU2_GPU_MOTION_OFF: points, geometry and lights in motion left to the CPU lane, as before the GPU rendered them, to compare.
    static bool motionOnGpu() {
        static const bool on = std::getenv("XPU2_GPU_MOTION_OFF") == nullptr;
        return on;
    }
    // A mesh the GPU programs do not render, or one the card's budget left to the CPU lane (cpu::routeMeshes): its samples
    // go to the CPU lane. Points shared by an instance's transform are one: the point program reads them in world space.
    static bool standIn(const cpu::Mesh& mesh) {
        const auto& m = *mesh.data;
        const bool moving = !m.steps.empty() || !m.curveSteps.empty();
        return mesh.cpuOnly || (m.points && mesh.instance) || ((moving || m.points) && !motionOnGpu());
    }
    // Whether the GPU hands samples to the CPU lane: stand-in meshes, or volumes its programs do not render (mapped
    // colour, emission or anisotropy; grids not on the card: volumeCrossings defers them), which need the CPU lane and the deferred samples' ring.
    // Each follower in its context, then this lane's again. Between frames (the scene, its edits, a frame's begin) a
    // follower that fails is let go and the others go on (forwardOrDrop); within a frame its failure is the frame's
    // (forward), which goes to the fallback as any GPU failure does.
    template <class F>
    void forwardOrDrop(F&& call) {
        for (size_t i = 0; i < followers_.size();) {
            try {
                followers_[i]->use();
                {
                    multi::OnNode near(followers_[i]->node());
                    call(*followers_[i]);
                }
                ++i;
            } catch (const std::exception& e) {
                std::printf("MoonRay XPU2: graphics card %d left out: %s\n", followers_[i]->card(), e.what());
                std::fflush(stdout);
                followers_[i]->use();
                followers_.erase(followers_.begin() + std::ptrdiff_t(i));
            }
        }
        use();
    }
    template <class F>
    void forward(F&& call) {
        try {
            for (const auto& f : followers_) {
                f->use();
                call(*f);
            }
        } catch (...) {
            use();
            throw;
        }
        use();
    }
    // Whether the errors may be asked for now: a launch slot free on every card (requestErrors takes one on each).
    bool canAsk() const {
        if (followers_.empty()) return true;
        if (!paramsFree()) return false;
        for (const auto& f : followers_)
            if (!f->paramsFree()) return false;
        return true;
    }
    bool standIns() const { return scene_.standInTraversable != 0 || scene_.defersVolumes; }
    // The samples the GPU deferred since the last call, for the CPU lane.
    void takeDeferred(std::vector<Parked>& out) {
        out.insert(out.end(), deferredOut_.begin(), deferredOut_.end());
        deferredOut_.clear();
    }
    // The frame's configuration (mixed::config: the camera, render space, the integrator's settings) on the device, with
    // the bokeh image, pixel sample map and bake camera maps it points to, held in a list of their own: made again alone
    // for a camera-only edit (updateCamera).
    std::vector<gpu::B> configKeep_;
    void uploadConfig(const cpu::PlanView& plan) {
        configKeep_.clear();
        auto config = mixed::config(plan);
        {
            // The camera's bokeh image and bake camera maps (camera_core.h) with device pointers.
            auto keep = [&](const void* host, size_t bytes) {
                B b = upload(host, bytes);
                configKeep_.push_back(b);
                return b->pointer;
            };
            if (plan->bokehImage && config.camera.bokehImage) {
                const lights::ImageData& data = *plan->bokehImage;
                lights::Image view = data.view;
                view.pixels = reinterpret_cast<const float*>(keep(data.pixels.data(), data.pixels.size() * sizeof(float)));
                view.cdf = reinterpret_cast<const float*>(keep(data.cdf.data(), data.cdf.size() * sizeof(float)));
                view.guide = reinterpret_cast<const U32*>(keep(data.guide.data(), data.guide.size() * sizeof(U32)));
                view.dists = reinterpret_cast<const lights::Dist1D*>(keep(data.dists.data(), data.dists.size() * sizeof(lights::Dist1D)));
                view.levels = reinterpret_cast<const lights::ImageLevel*>(keep(data.levels.data(), data.levels.size() * sizeof(lights::ImageLevel)));
                config.camera.bokehImage = reinterpret_cast<const void*>(keep(&view, sizeof(view)));
            }
            if (!plan->pixelSamples.empty())
                config.pixelSamples = reinterpret_cast<const U32*>(keep(plan->pixelSamples.data(), plan->pixelSamples.size() * sizeof(U32)));
            if (plan->cameraMap && config.camera.map) {
                const auto& data = *plan->cameraMap;
                surface::CameraMap view = data.view;
                auto floats = [&](const std::vector<float>& v) {
                    return v.empty() ? nullptr : reinterpret_cast<const float*>(keep(v.data(), v.size() * sizeof(float)));
                };
                view.positions = floats(data.positions);
                view.normals = floats(data.normals);
                view.normalMap = floats(data.normalMap);
                config.camera.map = reinterpret_cast<const void*>(keep(&view, sizeof(view)));
            }
        }
        scene_.config = upload(&config, sizeof(config));
    }

public:
    // A camera-only edit (the session's fast path, realtime): the plan is the one prepared, with only its camera, render
    // space and render origin made again (cpu::updateCamera). Its configuration goes to the device; the rest stays.
    void updateCamera(const cpu::PlanView& plan) {
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { f.updateCamera(plan); });
        plan_ = plan;
        // The volumes' records again: a mapped one's settings at the render origin, where the camera now is.
        if (scene_.volumes && !plan->volumes.empty()) {
            scene_.volumes = upload(plan->volumes.data(), plan->volumes.size() * sizeof(volume::Shader));
            uploadRegions(*plan);
        }
        uploadConfig(plan);
    }
    // A placement-only edit (the session's fast path, realtime): the plan is the one prepared with some meshes placed
    // anew (cpu::placeMoved), its lights and camera as they now are (cpu::updateLights, cpu::updateCamera). Each mesh's
    // structure, buffers and textures stay; the moved meshes' instances and records, the top trees over the instances
    // and the lights' records go again. False where it cannot (a moved stand-in or curve, a light whose tables are not
    // on the card), and the lane is as it was: prepare the plan then.
    bool updatePlacements(const cpu::PlanView& plan) {
        if (!followers_.empty()) {  // a follower that cannot is prepared again with the rest (the caller's prepare)
            bool all = true;
            forwardOrDrop([&](GpuLane& f) { all = f.updatePlacements(plan) && all; });
            if (!all) return false;
        }
        if (!plan_ || !scene_.traversable || plan->meshes.size() != plan_->meshes.size() || instances_.size() != plan->meshes.size() ||
            meshAttributesHost_.size() != plan->meshes.size())
            return false;
        std::vector<unsigned> moved;
        for (unsigned i = 0; i < plan->meshes.size(); ++i) {
            const auto &x = plan_->meshes[i], &y = plan->meshes[i];
            if (x.data != y.data || x.visibility != y.visibility || x.volume != y.volume || x.singleSided != y.singleSided ||
                x.material != y.material || x.cpuOnly != y.cpuOnly)
                return false;
            if (x.instance == y.instance && std::memcmp(x.objectToWorld, y.objectToWorld, sizeof x.objectToWorld) == 0) continue;
            if (standIn(y) || y.data->curveBasis || standIn(x) || !y.data->steps.empty()) return false;
            if (meshCache_.find(y.data.get()) == meshCache_.end()) return false;
            moved.push_back(i);
        }
        std::vector<lights::Light> deviceLights = plan->lights;
        for (auto& l : deviceLights) {
            if (l.mesh) {
                const auto found = lightTables_.find(l.mesh);
                if (found == lightTables_.end()) return false;
                l.mesh = static_cast<const lights::MeshLightData*>(found->second);
            }
            if (l.image) {
                const auto found = lightTables_.find(l.image);
                if (found == lightTables_.end()) return false;
                l.image = static_cast<const lights::Image*>(found->second);
            }
        }
        plan_ = plan;
        for (unsigned i : moved) {
            const auto& e = plan->meshes[i];
            OptixInstance& inst = instances_[i];
            std::memset(inst.transform, 0, sizeof inst.transform);
            inst.transform[0] = inst.transform[5] = inst.transform[10] = 1;
            shade::MeshAttributes& a = meshAttributesHost_[i];
            a = meshCache_.at(e.data.get()).attributes;
            if (!e.instance) continue;
            for (int r = 0; r < 3; ++r) {
                inst.transform[4 * r + 0] = e.objectToWorld[0 + r];
                inst.transform[4 * r + 1] = e.objectToWorld[3 + r];
                inst.transform[4 * r + 2] = e.objectToWorld[6 + r];
                inst.transform[4 * r + 3] = e.objectToWorld[9 + r];
            }
            a.instance = e.instance;
            for (int k = 0; k < 12; ++k) a.objectToWorld[k] = e.objectToWorld[k];
            for (int k = 0; k < 9; ++k) a.normalToWorld[k] = e.normalToWorld[k];
            for (int k = 0; k < 12; ++k) a.worldToObject[k] = e.worldToObject[k];
            if (!e.instancePrims.empty()) {
                auto prims = upload(e.instancePrims.data(), e.instancePrims.size() * sizeof(shade::PrimAttr));
                auto data = upload(e.instancePrimData.data(), std::max<size_t>(e.instancePrimData.size(), 1) * sizeof(float));
                a.instancePrims = reinterpret_cast<const shade::PrimAttr*>(prims->pointer);
                a.instancePrimData = reinterpret_cast<const float*>(data->pointer);
                a.instancePrimCount = U32(e.instancePrims.size());
                placedKeep_[i] = {prims, data};
            }
        }
        if (!moved.empty()) {
            buildTops(*plan);
            scene_.meshAttributes = upload(meshAttributesHost_.data(), meshAttributesHost_.size() * sizeof(shade::MeshAttributes));
        }
        uploadLights(*plan, deviceLights);
        if (scene_.volumes && !plan->volumes.empty()) {
            // A moved volume's record (its grids' transforms, its settings at the render origin), and its baked grid
            // where it was made again.
            scene_.volumes = upload(plan->volumes.data(), plan->volumes.size() * sizeof(volume::Shader));
            bool regrid = plan->volumeGridViews.size() != gridsSent_.size();
            for (size_t k = 0; !regrid && k < gridsSent_.size(); ++k) regrid = gridHeld(plan->volumeGridViews[k]) != gridsSent_[k];
            if (regrid) uploadGrids(*plan);
            uploadRegions(*plan);  // a moved glowing volume's regions where it is
        }
        uploadConfig(plan);
        placementsMoved_ = moved.size();
        return true;
    }
    size_t placementsMoved_ = 0;
    // The GPU's rate in samples a second, as its launches have found it across frames (realtime's launch sizes).
    double rateHint() const { return rateHint_; }
    // The lane's pinned host and device memory now and at its peak, in MB (gpu::Driver's counts).
    std::string memoryText() const {
        if (!followers_.empty()) {
            std::string all = memoryTextOwn();
            for (const auto& f : followers_) all += "; card " + std::to_string(f->card()) + " " + f->memoryTextOwn();
            return all;
        }
        return memoryTextOwn();
    }
    std::string memoryTextOwn() const {
        char text[260];
        std::snprintf(text, sizeof text, "pinned %zu MB (at most %zu, of %zu), device %zu MB (at most %zu); structures %zu MB compacted from %zu",
                      driver.pinnedBytes >> 20, driver.peakPinned >> 20, driver.pinnedCap >> 20, driver.deviceBytes >> 20, driver.peakDevice >> 20,
                      compactedBytes_ >> 20, builtBytes_ >> 20);
        return text;
    }
    void noteRate(double rate) {
        if (rate > 0) rateHint_ = rateHint_ > 0 ? 0.8 * rateHint_ + 0.2 * rate : rate;
    }
    double rateHint_ = 0;

    // Realtime's denoised pictures (the session's RealtimeDenoiser): OptiX's denoiser, its HDR model with albedo and
    // normal guides, on a stream of its own. It is called from the session's denoising thread, where the lane's context
    // is made current, and holds buffers of its own (the lane's are its feeder's): host pictures in, three floats a
    // pixel, and the denoised picture out. Made by the lane, and let go before it.
    class Denoiser {
    public:
        // Its guides (XPU2_SESSION_RT_GUIDES): "clean", the albedo denoised on its own first, as the kit's "Denoiser
        // cleans its guides first" has Open Image Denoise do (a material whose albedo differs from sample to sample,
        // flakes or iridescence, gives a guide as noisy as the picture at realtime's few samples, and the denoiser
        // keeps what its guide shows); "raw", the albedo as rendered; "normal", the normals alone; "none". Unset, as
        // that MoonRay Source Override is set (MOONRAY_MODO_DENOISE_CLEAN_GUIDES, on by default): clean, or raw.
        // upscale: OptiX's model that denoises a picture and makes it twice its size each way (UPSCALE2X), for the pictures
        // realtime shows from a quarter of their pixels (a pixel in each two by two) while the view or the scene moves;
        // off (XPU2_SESSION_RT_UPSCALE=0, the kit's Lower Resolution While Moving off), the HDR model at that size,
        // enlarged after as it is (RealtimeDenoiser::upscale).
        explicit Denoiser(GpuLane& lane, bool upscale = false) : lane_(lane) {
            const char* up = std::getenv("XPU2_SESSION_RT_UPSCALE");
            upscale_ = upscale && !(up && up[0] == '0');
            const char* guides = std::getenv("XPU2_SESSION_RT_GUIDES");
            const char* cleaned = std::getenv("MOONRAY_MODO_DENOISE_CLEAN_GUIDES");
            const std::string g = guides ? guides : cleaned && cleaned[0] == '0' ? "raw" : "clean";
            mode_ = g == "raw" ? Raw : g == "normal" ? NormalOnly : g == "none" ? NoGuides : Clean;
        }
        ~Denoiser() { release(); }
        Denoiser(const Denoiser&) = delete;
        // Whether it makes its pictures twice their size each way: out then holds (2w x 2h) pixels.
        bool upscales() const { return upscale_; }
        bool run(const float* color, const float* albedo, const float* normal, float* out, U32 w, U32 h, std::string& error) {
            auto& d = lane_.driver;
            try {
                gpu::Driver::check(d.cuCtxSetCurrent_p(d.context));
                if (!stream_) {
                    // At the GPU's highest priority, so that its kernels go ahead of the path tracer's queued blocks (off:
                    // XPU2_SESSION_RT_PRIORITY=0).
                    using Create = CUresult(CUDAAPI*)(CUstream*, unsigned int, int);
                    using Range = CUresult(CUDAAPI*)(int*, int*);
                    const auto create = reinterpret_cast<Create>(GetProcAddress(d.library, "cuStreamCreateWithPriority"));
                    const auto range = reinterpret_cast<Range>(GetProcAddress(d.library, "cuCtxGetStreamPriorityRange"));
                    const char* priority = std::getenv("XPU2_SESSION_RT_PRIORITY");
                    int least = 0, greatest = 0;
                    if (create && range && !(priority && priority[0] == '0') && range(&least, &greatest) == CUDA_SUCCESS)
                        gpu::Driver::check(create(&stream_, CU_STREAM_NON_BLOCKING, greatest));
                    else
                        gpu::Driver::check(d.cuStreamCreate_p(&stream_, CU_STREAM_NON_BLOCKING));
                }
                if (!done_) gpu::Driver::check(d.cuEventCreate_p(&done_, CU_EVENT_DISABLE_TIMING));
                const bool withAlbedo = mode_ == Raw || mode_ == Clean, withNormal = mode_ != NoGuides;
                if (!beauty_.handle) beauty_.create(lane_.optix_, withAlbedo, withNormal, upscale_);
                if (mode_ == Clean && !albedo_.handle) albedo_.create(lane_.optix_, false, false, false);
                const size_t bytes = size_t(w) * h * 3 * sizeof(float);
                // The picture out: twice the size each way where it upscales.
                const U32 ow = upscale_ ? 2 * w : w, oh = upscale_ ? 2 * h : h;
                const size_t outBytes = size_t(ow) * oh * 3 * sizeof(float);
                if (w != width_ || h != height_) {
                    buffers();
                    beauty_.setup(d, stream_, w, h, ow, oh);
                    if (albedo_.handle) albedo_.setup(d, stream_, w, h, w, h);
                    for (auto& i : intensity_) gpu::Driver::check(d.cuMemAlloc_v2_p(&i, 16));
                    for (size_t k = 0; k < 5; ++k) gpu::Driver::check(d.cuMemAlloc_v2_p(&images_[k], k == 3 ? outBytes : bytes));
                    gpu::Driver::check(d.cuMemAllocHost_v2_p(&host_, bytes * 3 + outBytes + 64));
                    width_ = w;
                    height_ = h;
                }
                float* host = static_cast<float*>(host_);
                const size_t n = size_t(w) * h * 3;
                std::copy(color, color + n, host);
                std::copy(albedo, albedo + n, host + n);
                std::copy(normal, normal + n, host + 2 * n);
                for (int i = 0; i < 3; ++i)
                    gpu::Driver::check(d.cuMemcpyHtoDAsync_v2_p(images_[i], host + i * n, bytes, stream_));
                auto image = [&](CUdeviceptr data, U32 iw = 0, U32 ih = 0) {
                    if (!iw) iw = w, ih = h;
                    OptixImage2D v{};
                    v.data = data;
                    v.width = iw;
                    v.height = ih;
                    v.rowStrideInBytes = iw * 3 * sizeof(float);
                    v.pixelStrideInBytes = 3 * sizeof(float);
                    v.format = OPTIX_PIXEL_FORMAT_FLOAT3;
                    return v;
                };
                OptixDenoiserGuideLayer guide{};
                if (withAlbedo) guide.albedo = image(images_[mode_ == Clean ? 4 : 1]);
                if (withNormal) guide.normal = image(images_[2]);
                if (mode_ == Clean) albedo_.invoke(stream_, image(images_[1]), image(images_[4]), OptixDenoiserGuideLayer{}, intensity_[1]);
                beauty_.invoke(stream_, image(images_[0]), image(images_[3], ow, oh), guide, intensity_[0]);
                const size_t outN = size_t(ow) * oh * 3;
                gpu::Driver::check(d.cuMemcpyDtoHAsync_v2_p(host + 3 * n, images_[3], outBytes, stream_));
                gpu::Driver::check(d.cuMemcpyDtoHAsync_v2_p(host + 3 * n + outN, intensity_[0], sizeof(float), stream_));
                gpu::Driver::check(d.cuEventRecord_p(done_, stream_));
                // Waited for by polling, as the lane's own work is: the context is not synchronized.
                for (;;) {
                    const CUresult e = d.cuEventQuery_p(done_);
                    if (e == CUDA_SUCCESS) break;
                    if (e != CUDA_ERROR_NOT_READY) gpu::Driver::check(e);
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                std::copy(host + 3 * n, host + 3 * n + outN, out);
                if (std::getenv("XPU2_SESSION_RT_DEBUG") && !upscale_) {
                    // A diagnostic: the inputs' values that are not finite, the intensity OptiX found, how much it changed.
                    size_t bad[3] = {}, i = 0;
                    double change = 0;
                    for (const float* in : {color, albedo, normal}) {
                        for (size_t k = 0; k < n; ++k) bad[i] += !std::isfinite(in[k]);
                        ++i;
                    }
                    for (size_t k = 0; k < n; ++k) change += std::fabs(double(out[k]) - double(color[k]));
                    std::printf("XPU2: realtime denoise: not finite %zu %zu %zu, intensity %g, mean change %g\n", bad[0], bad[1], bad[2],
                                double(host[4 * n]), change / double(n));
                    std::fflush(stdout);
                }
                return true;
            } catch (const std::exception& e) {
                error = std::string("OptiX's denoiser: ") + e.what();
                return false;
            }
        }

    private:
        enum Mode { Clean, Raw, NormalOnly, NoGuides };
        // One OptiX denoiser (HDR model) with its state and scratch.
        struct Model {
            OptixDenoiser handle = nullptr;
            CUdeviceptr state = 0, scratch = 0;
            size_t stateBytes = 0, scratchBytes = 0;
            bool upscale = false;  // OptiX's UPSCALE2X (an AOV model): its picture out twice the size each way
            void create(OptixDeviceContext context, bool albedo, bool normal, bool twice) {
                OptixDenoiserOptions options{};
                options.guideAlbedo = albedo ? 1u : 0u;
                options.guideNormal = normal ? 1u : 0u;
                upscale = twice;
                gpu::ox(optixDenoiserCreate(context, twice ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_HDR, &options, &handle));
            }
            // w, h: the picture in; ow, oh: out (OptiX's memory sized by what it denoises into, set up by what comes in).
            void setup(gpu::Driver& d, CUstream stream, U32 w, U32 h, U32 ow, U32 oh) {
                OptixDenoiserSizes sizes{};
                gpu::ox(optixDenoiserComputeMemoryResources(handle, ow, oh, &sizes));
                stateBytes = sizes.stateSizeInBytes;
                scratchBytes = std::max({sizes.withoutOverlapScratchSizeInBytes, sizes.computeIntensitySizeInBytes, sizes.computeAverageColorSizeInBytes});
                gpu::Driver::check(d.cuMemAlloc_v2_p(&state, std::max<size_t>(stateBytes, 16)));
                gpu::Driver::check(d.cuMemAlloc_v2_p(&scratch, std::max<size_t>(scratchBytes, 16)));
                gpu::ox(optixDenoiserSetup(handle, stream, w, h, state, stateBytes, scratch, scratchBytes));
            }
            void invoke(CUstream stream, const OptixImage2D& input, const OptixImage2D& output, const OptixDenoiserGuideLayer& guide,
                        CUdeviceptr intensity) {
                OptixDenoiserLayer layer{};
                layer.input = input;
                layer.output = output;
                OptixDenoiserParams params{};
                if (upscale) {
                    // An AOV model goes by the picture's average colour (three floats) where the HDR one goes by its
                    // intensity.
                    gpu::ox(optixDenoiserComputeAverageColor(handle, stream, &layer.input, intensity, scratch, scratchBytes));
                    params.hdrAverageColor = intensity;
                } else {
                    gpu::ox(optixDenoiserComputeIntensity(handle, stream, &layer.input, intensity, scratch, scratchBytes));
                    params.hdrIntensity = intensity;
                }
                gpu::ox(optixDenoiserInvoke(handle, stream, &params, state, stateBytes, &guide, &layer, 1, 0, 0, scratch, scratchBytes));
            }
            void free(gpu::Driver& d) {
                for (CUdeviceptr* p : {&state, &scratch})
                    if (*p) {
                        d.cuMemFree_v2_p(*p);
                        *p = 0;
                    }
            }
            void destroy(gpu::Driver& d) {
                free(d);
                if (handle) optixDenoiserDestroy(handle);
                handle = nullptr;
            }
        };
        void buffers() {
            auto& d = lane_.driver;
            beauty_.free(d);
            albedo_.free(d);
            for (CUdeviceptr* p : {&intensity_[0], &intensity_[1], &images_[0], &images_[1], &images_[2], &images_[3], &images_[4]})
                if (*p) {
                    d.cuMemFree_v2_p(*p);
                    *p = 0;
                }
            if (host_) d.cuMemFreeHost_p(host_);
            host_ = nullptr;
            width_ = height_ = 0;
        }
        void release() {
            auto& d = lane_.driver;
            if (d.context) d.cuCtxSetCurrent_p(d.context);
            if (stream_ && done_) {
                d.cuEventRecord_p(done_, stream_);
                while (d.cuEventQuery_p(done_) == CUDA_ERROR_NOT_READY) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            buffers();
            beauty_.destroy(d);
            albedo_.destroy(d);
            if (done_) d.cuEventDestroy_v2_p(done_);
            done_ = nullptr;
            if (stream_) d.cuStreamDestroy_v2_p(stream_);
            stream_ = nullptr;
        }
        GpuLane& lane_;
        Mode mode_ = Clean;
        CUstream stream_ = nullptr;
        CUevent done_ = nullptr;
        Model beauty_, albedo_;
        bool upscale_ = false;
        CUdeviceptr intensity_[2] = {}, images_[5] = {};  // colour, albedo, normal, output, the albedo cleaned
        void* host_ = nullptr;
        U32 width_ = 0, height_ = 0;
    };

private:
public:
    void prepare(const cpu::PlanView& plan) {
        {
            // The scene's pinned buffers on the node nearest the card (multi_device.h).
            multi::OnNode near(options_.node);
            prepareOwn(plan);
        }
        if (followers_.empty()) return;
        forwardOrDrop([&](GpuLane& f) {
            if (multi::simulated("XPU2_SIMULATE_CARD_FAIL") == unsigned(f.options_.lane) && f.options_.lane)
                throw std::runtime_error("failed (simulated, XPU2_SIMULATE_CARD_FAIL)");
            f.prepare(plan);
        });
        // The card's memory as every lane left it: lanes sharing a card (XPU2_SIMULATE_CARDS) did not take it since.
        size_t freeBytes = 0, totalBytes = 0;
        forwardOrDrop([&](GpuLane& f) {
            if (f.freeAfterPrepare_ && f.memoryInfo(freeBytes, totalBytes)) f.freeAfterPrepare_ = freeBytes;
        });
        if (freeAfterPrepare_ && memoryInfo(freeBytes, totalBytes)) freeAfterPrepare_ = freeBytes;
    }
    void prepareOwn(const cpu::PlanView& plan) {
        cpu::need(plan && cpu::isSpatial(plan->domain), "Asynchronous GPU lane requires a spatial plan");
        cpu::need(plan->paths, "The GPU programs carry the path tracer only");
        {const std::string why = cpuOnly(plan); if (!why.empty()) throw cpu::Unsupported("the GPU leaves " + why + " to the CPU");}
        plan_ = plan;
        scene_ = {};
        placedKeep_.clear();
        tops_.clear();
        // Where the prepare's time goes (prepareLaps_, the session prints them): meshes, the rest of the records,
        // textures, films.
        prepareLaps_.clear();
        auto prepareBegan = std::chrono::steady_clock::now();
        auto lap = [&](const char* what) {
            const auto now = std::chrono::steady_clock::now();
            prepareLaps_.push_back({what, std::chrono::duration<double>(now - prepareBegan).count()});
            prepareBegan = now;
        };
        std::vector<OptixInstance> instances(plan->meshes.size());
        std::vector<mixed::Geometry> geometry;
        std::vector<surface::Lambert> materials;
        std::vector<shade::MaterialRecord> meshMaterials;
        std::vector<shade::MeshAttributes> meshAttributes;
        std::vector<OptixInstance> standIns;
        std::vector<MeshMotion> meshMotion(plan->meshes.size());
        std::map<const cpu::MeshData*, MeshEntry> nextCache;
        meshesReused_ = 0;
        for (unsigned i = 0; i < plan->meshes.size(); ++i) {
            const auto& m = *plan->meshes[i].data;
            if (standIn(plan->meshes[i])) {
                // Its box over every time step (radii too), twelve triangles: out of the surfaces' tree (visibility
                // 0, the index kept) and in the stand-ins', where a ray meeting it defers its sample.
                float lo[3] = {3.4e38f, 3.4e38f, 3.4e38f}, hi[3] = {-3.4e38f, -3.4e38f, -3.4e38f};
                auto grow = [&](float x, float y, float z, float r) {
                    const float p[3] = {x, y, z};
                    for (int a = 0; a < 3; ++a) {
                        lo[a] = std::min(lo[a], p[a] - r);
                        hi[a] = std::max(hi[a], p[a] + r);
                    }
                };
                // A shared instance's data is in its reference's space: its box is made of the placed vertices.
                const cpu::Mesh& entry = plan->meshes[i];
                auto points = [&](const std::vector<surface::V3>& ps) {
                    for (size_t k = 0; k < ps.size(); ++k) {
                        surface::V3 q = ps[k];
                        if (entry.instance) {
                            const float* x = entry.objectToWorld;
                            q = {ps[k].x * x[0] + ps[k].y * x[3] + ps[k].z * x[6] + x[9], ps[k].x * x[1] + ps[k].y * x[4] + ps[k].z * x[7] + x[10],
                                 ps[k].x * x[2] + ps[k].y * x[5] + ps[k].z * x[8] + x[11]};
                        }
                        grow(q.x, q.y, q.z, k < m.pointRadius.size() ? m.pointRadius[k] : 0.0f);
                    }
                };
                points(m.positions);
                for (const auto& s : m.steps) points(s);
                auto curvePoints = [&](const float* c, size_t n) {
                    for (size_t k = 0; 4 * k + 3 < n; ++k) grow(c[4 * k], c[4 * k + 1], c[4 * k + 2], std::fabs(c[4 * k + 3]));
                };
                curvePoints(m.curvePoints.data(), m.curvePoints.size());
                for (const auto& s : m.curveSteps) curvePoints(s.data(), s.size());
                if (lo[0] > hi[0]) lo[0] = lo[1] = lo[2] = hi[0] = hi[1] = hi[2] = 0.0f;
                static const cpu::Tri faces[12] = {{0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}, {0, 1, 5}, {0, 5, 4},
                                                   {2, 6, 7}, {2, 7, 3}, {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}};
                std::vector<surface::V3> box;
                std::vector<cpu::Tri> boxFaces;
                auto addBox = [&](const float* l, const float* h) {
                    const unsigned first = unsigned(box.size());
                    for (int k = 0; k < 8; ++k) box.push_back({k & 1 ? h[0] : l[0], k & 2 ? h[1] : l[1], k & 4 ? h[2] : l[2]});
                    for (const auto& f : faces) boxFaces.push_back({first + f.a, first + f.b, first + f.c});
                };
                if (entry.cpuOnly && !m.points && m.steps.empty() && !m.curveBasis && !m.triangles.empty()) {
                    // A still mesh the card's budget left to the CPU lane (cpu::routeMeshes): a box round each run of
                    // ClusterTriangles of its triangles, in their order (a tessellation's come face by face, so a run
                    // is a patch of the surface), in place of one box round the whole: a ray deferred only where it
                    // passes near the mesh. Every triangle is inside its box, so no hit is lost.
                    constexpr size_t ClusterTriangles = 128;
                    auto placedAt = [&](U32 v) {
                        const surface::V3 p = m.positions[v];
                        if (!entry.instance) return p;
                        const float* x = entry.objectToWorld;
                        return surface::V3{p.x * x[0] + p.y * x[3] + p.z * x[6] + x[9], p.x * x[1] + p.y * x[4] + p.z * x[7] + x[10],
                                           p.x * x[2] + p.y * x[5] + p.z * x[8] + x[11]};
                    };
                    box.reserve((m.triangles.size() / ClusterTriangles + 1) * 8);
                    boxFaces.reserve((m.triangles.size() / ClusterTriangles + 1) * 12);
                    for (size_t first = 0; first < m.triangles.size(); first += ClusterTriangles) {
                        float l[3] = {3.4e38f, 3.4e38f, 3.4e38f}, h[3] = {-3.4e38f, -3.4e38f, -3.4e38f};
                        for (size_t k = first; k < std::min(m.triangles.size(), first + ClusterTriangles); ++k) {
                            const cpu::Tri& tr = m.triangles[k];
                            for (const U32 v : {tr.a, tr.b, tr.c}) {
                                const surface::V3 q = placedAt(v);
                                const float c[3] = {q.x, q.y, q.z};
                                for (int a = 0; a < 3; ++a) {
                                    l[a] = std::min(l[a], c[a]);
                                    h[a] = std::max(h[a], c[a]);
                                }
                            }
                        }
                        addBox(l, h);
                    }
                } else {
                    addBox(lo, hi);
                }
                auto vertices = upload(box.data(), box.size() * sizeof(surface::V3));
                auto indices = upload(boxFaces.data(), boxFaces.size() * sizeof(cpu::Tri));
                OptixBuildInput input{};
                unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                auto& tri = input.triangleArray;
                tri.vertexBuffers = &vertices->pointer;
                tri.numVertices = unsigned(box.size());
                tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                tri.vertexStrideInBytes = sizeof(surface::V3);
                tri.indexBuffer = indices->pointer;
                tri.numIndexTriplets = unsigned(boxFaces.size());
                tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                tri.indexStrideInBytes = sizeof(cpu::Tri);
                tri.flags = &flag;
                tri.numSbtRecords = 1;
                auto gas = build(input);
                auto& inst = instances[i];
                inst.transform[0] = inst.transform[5] = inst.transform[10] = 1;
                inst.instanceId = i;
                inst.visibilityMask = 0u;
                inst.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
                inst.traversableHandle = gas.second;
                OptixInstance stand = inst;
                stand.visibilityMask = 0xffu;
                standIns.push_back(stand);
                geometry.push_back({reinterpret_cast<const surface::V3*>(vertices->pointer), reinterpret_cast<const surface::Triangle*>(indices->pointer),
                                    unsigned(box.size()), unsigned(boxFaces.size())});
                materials.push_back(surface::Lambert{});
                meshMaterials.push_back(shade::MaterialRecord{});
                meshAttributes.push_back(shade::MeshAttributes{});
                scene_.keep.insert(scene_.keep.end(), {vertices, indices, gas.first});
                continue;
            }
            if (m.curveBasis) {
                // Curves: a custom primitive a span, its box as OptixGPUCurve::getPrimitiveAabbs makes it; the curve
                // hit group's program (async_gpu.cu) runs MoonRay XPU's curve programs (curve_core.h).
                const curves::Set host = cpu::curveSet(m);
                // In motion, its boxes and control points at each time step (OptiX's keys, evenly over the shutter).
                std::vector<U64L> boxKeys, pointKeys;
                for (U32 step = 0; step < host.stepCount; ++step) {
                    curves::Set at = host;
                    at.points = step ? m.curveSteps[step - 1].data() : m.curvePoints.data();
                    at.stepCount = 1;
                    std::vector<OptixAabb> boxes(host.spanCount);
                    for (U32 k = 0; k < host.spanCount; ++k) {
                        float lower[3], upper[3];
                        curves::spanBounds(at, k, lower, upper);
                        boxes[k] = {lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]};
                    }
                    auto stepBoxes = upload(boxes.data(), boxes.size() * sizeof(OptixAabb));
                    boxKeys.push_back(stepBoxes->pointer);
                    scene_.keep.push_back(stepBoxes);
                    if (step) {
                        const auto& s = m.curveSteps[step - 1];
                        auto stepPoints = upload(s.data(), s.size() * sizeof(float));
                        pointKeys.push_back(stepPoints->pointer);
                        scene_.keep.push_back(stepPoints);
                    }
                }
                auto points = upload(m.curvePoints.data(), m.curvePoints.size() * sizeof(float));
                auto spans = upload(m.curveSpans.data(), m.curveSpans.size() * sizeof(unsigned));
                auto counts = upload(m.curveSpanCounts.data(), m.curveSpanCounts.size() * sizeof(unsigned));
                OptixBuildInput input{};
                unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                auto& custom = input.customPrimitiveArray;
                custom.aabbBuffers = boxKeys.data();
                custom.numPrimitives = host.spanCount;
                custom.strideInBytes = sizeof(OptixAabb);
                custom.flags = &flag;
                custom.numSbtRecords = 1;
                auto gas = build(input, host.stepCount);
                auto& inst = instances[i];
                inst.transform[0] = inst.transform[5] = inst.transform[10] = 1;
                inst.instanceId = i;
                inst.sbtOffset = 1;  // the curve hit group
                inst.visibilityMask = plan->meshes[i].visibility & 0xffu;  // its visibility (rdl2 VisibilityType)
                inst.traversableHandle = gas.second;
                geometry.push_back({nullptr, nullptr, 0, 0});
                materials.push_back(plan->materials.at(plan->meshes[i].material));
                meshMaterials.push_back(plan->materialRecords.at(plan->meshes[i].material));
                shade::MeshAttributes attributes;
                for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = m.worldToObject[k];
                if (!m.prims.empty()) {
                    auto prims = upload(m.prims.data(), m.prims.size() * sizeof(shade::PrimAttr));
                    auto data = upload(m.primData.data(), std::max<size_t>(m.primData.size(), 1) * sizeof(float));
                    attributes.prims = reinterpret_cast<const shade::PrimAttr*>(prims->pointer);
                    attributes.primData = reinterpret_cast<const float*>(data->pointer);
                    attributes.primCount = U32(m.prims.size());
                    scene_.keep.push_back(prims);
                    scene_.keep.push_back(data);
                }
                attributes.curve = host;
                attributes.curve.points = reinterpret_cast<const float*>(points->pointer);
                attributes.curve.spans = reinterpret_cast<const U32*>(spans->pointer);
                attributes.curve.curveSpans = reinterpret_cast<const U32*>(counts->pointer);
                if (host.stepCount > 1) {
                    auto steps = upload(pointKeys.data(), pointKeys.size() * sizeof(U64L));
                    attributes.curve.steps = reinterpret_cast<const float* const*>(steps->pointer);
                    scene_.keep.push_back(steps);
                    meshMotion[i].steps = host.stepCount;
                    meshMotion[i].kind = MotionCurves;
                } else {
                    attributes.curve.steps = nullptr;
                }
                meshAttributes.push_back(attributes);
                scene_.keep.insert(scene_.keep.end(), {points, spans, counts, gas.first});
                continue;
            }
            if (m.points) {
                // Points: a custom primitive each, its box at each time step; the same hit group as curves, whose program
                // runs MoonRay's sphere test (points_core.h).
                auto centres = upload(m.positions.data(), m.positions.size() * sizeof(surface::V3));
                std::vector<U64L> centreKeys{centres->pointer}, boxKeys;
                scene_.keep.push_back(centres);
                for (const auto& s : m.steps) {
                    auto step = upload(s.data(), s.size() * sizeof(surface::V3));
                    centreKeys.push_back(step->pointer);
                    scene_.keep.push_back(step);
                }
                for (size_t step = 0; step < centreKeys.size(); ++step) {
                    const std::vector<surface::V3>& at = step ? m.steps[step - 1] : m.positions;
                    std::vector<OptixAabb> boxes(m.points);
                    for (unsigned k = 0; k < m.points; ++k) {
                        const float r = m.pointRadius[k];
                        boxes[k] = {at[k].x - r, at[k].y - r, at[k].z - r, at[k].x + r, at[k].y + r, at[k].z + r};
                    }
                    auto stepBoxes = upload(boxes.data(), boxes.size() * sizeof(OptixAabb));
                    boxKeys.push_back(stepBoxes->pointer);
                    scene_.keep.push_back(stepBoxes);
                }
                auto radii = upload(m.pointRadius.data(), m.pointRadius.size() * sizeof(float));
                OptixBuildInput input{};
                unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                auto& custom = input.customPrimitiveArray;
                custom.aabbBuffers = boxKeys.data();
                custom.numPrimitives = m.points;
                custom.strideInBytes = sizeof(OptixAabb);
                custom.flags = &flag;
                custom.numSbtRecords = 1;
                auto gas = build(input, unsigned(boxKeys.size()));
                auto& inst = instances[i];
                inst.transform[0] = inst.transform[5] = inst.transform[10] = 1;
                inst.instanceId = i;
                inst.sbtOffset = 1;  // the curve hit group
                inst.visibilityMask = plan->meshes[i].visibility & 0xffu;
                inst.traversableHandle = gas.second;
                geometry.push_back({reinterpret_cast<const surface::V3*>(centres->pointer), nullptr, unsigned(m.positions.size()), 0});
                materials.push_back(plan->materials.at(plan->meshes[i].material));
                meshMaterials.push_back(plan->materialRecords.at(plan->meshes[i].material));
                shade::MeshAttributes attributes;
                for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = m.worldToObject[k];
                if (!m.prims.empty()) {
                    auto prims = upload(m.prims.data(), m.prims.size() * sizeof(shade::PrimAttr));
                    auto data = upload(m.primData.data(), std::max<size_t>(m.primData.size(), 1) * sizeof(float));
                    attributes.prims = reinterpret_cast<const shade::PrimAttr*>(prims->pointer);
                    attributes.primData = reinterpret_cast<const float*>(data->pointer);
                    attributes.primCount = U32(m.prims.size());
                    scene_.keep.insert(scene_.keep.end(), {prims, data});
                }
                if (!m.steps.empty()) meshMotion[i] = uploadMotion(m, attributes, centreKeys, scene_.keep);
                meshMotion[i].pointRadius = reinterpret_cast<const float*>(radii->pointer);
                meshAttributes.push_back(attributes);
                scene_.keep.insert(scene_.keep.end(), {radii, gas.first});
                continue;
            }
            const bool surface = plan->meshes[i].material != ~0u;
            auto& inst = instances[i];
            inst.transform[0] = inst.transform[5] = inst.transform[10] = 1;
            inst.instanceId = i;
            // Visibility: a surface's (rdl2 VisibilityType, a ray's mask must share it) in the surfaces' tree; a mesh with
            // no material bounds a volume alone, so none there. Back faces are culled for a single-sided surface alone.
            inst.visibilityMask = surface ? (plan->meshes[i].visibility & 0xffu) : 0u;
            inst.flags = plan->meshes[i].singleSided ? OPTIX_INSTANCE_FLAG_NONE : OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
            materials.push_back(surface ? plan->materials.at(plan->meshes[i].material) : surface::Lambert{});
            meshMaterials.push_back(surface ? plan->materialRecords.at(plan->meshes[i].material) : shade::MaterialRecord{});
            // The same mesh as the last plan's, or as an earlier mesh of this one (an instance's): its buffers,
            // structure and records again.
            if (auto here = nextCache.find(&m); here != nextCache.end()) {
                inst.traversableHandle = here->second.handle;
                geometry.push_back(here->second.geometry);
                meshAttributes.push_back(here->second.attributes);
                meshMotion[i] = here->second.motion;
                continue;
            }
            if (auto found = meshCache_.find(&m); found != meshCache_.end()) {
                MeshEntry entry = found->second;
                inst.traversableHandle = entry.handle;
                geometry.push_back(entry.geometry);
                meshAttributes.push_back(entry.attributes);
                meshMotion[i] = entry.motion;
                scene_.keep.insert(scene_.keep.end(), entry.keep.begin(), entry.keep.end());
                nextCache.emplace(&m, std::move(entry));
                ++meshesReused_;
                continue;
            }
            MeshEntry entry;
            entry.data = plan->meshes[i].data;
            auto vertices = upload(m.positions.data(), m.positions.size() * sizeof(surface::V3));
            auto indices = upload(m.triangles.data(), m.triangles.size() * sizeof(cpu::Tri));
            // In motion, its vertices at each time step (OptiX's keys, evenly over the shutter as Embree's).
            std::vector<U64L> vertexKeys{vertices->pointer};
            for (const auto& s : m.steps) {
                auto step = upload(s.data(), s.size() * sizeof(surface::V3));
                vertexKeys.push_back(step->pointer);
                entry.keep.push_back(step);
            }
            OptixBuildInput input{};
            unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            auto& tri = input.triangleArray;
            tri.vertexBuffers = vertexKeys.data();
            tri.numVertices = unsigned(m.positions.size());
            tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            tri.vertexStrideInBytes = sizeof(surface::V3);
            tri.indexBuffer = indices->pointer;
            tri.numIndexTriplets = unsigned(m.triangles.size());
            tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            tri.indexStrideInBytes = sizeof(cpu::Tri);
            tri.flags = &flag;
            tri.numSbtRecords = 1;
            auto gas = build(input, unsigned(vertexKeys.size()));
            inst.traversableHandle = gas.second;
            entry.handle = gas.second;
            entry.geometry = {reinterpret_cast<const surface::V3*>(vertices->pointer), reinterpret_cast<const surface::Triangle*>(indices->pointer),
                              unsigned(m.positions.size()), unsigned(m.triangles.size())};
            geometry.push_back(entry.geometry);
            shade::MeshAttributes attributes;
            for (int k = 0; k < 12; ++k) attributes.worldToObject[k] = m.worldToObject[k];
            if (!m.prims.empty()) {
                auto prims = upload(m.prims.data(), m.prims.size() * sizeof(shade::PrimAttr));
                auto data = upload(m.primData.data(), std::max<size_t>(m.primData.size(), 1) * sizeof(float));
                attributes.prims = reinterpret_cast<const shade::PrimAttr*>(prims->pointer);
                attributes.primData = reinterpret_cast<const float*>(data->pointer);
                attributes.primCount = U32(m.prims.size());
                entry.keep.push_back(prims);
                entry.keep.push_back(data);
            }
            if (!m.normals.empty()) {
                auto normals = upload(m.normals.data(), m.normals.size() * sizeof(surface::V3));
                attributes.normals = reinterpret_cast<const surface::V3*>(normals->pointer);
                attributes.normalRate = m.normalRate;
                entry.keep.push_back(normals);
            }
            if (!m.uvs.empty()) {
                auto uvs = upload(m.uvs.data(), m.uvs.size() * sizeof(float));
                attributes.uvs = reinterpret_cast<const float*>(uvs->pointer);
                attributes.uvRate = m.uvRate;
                entry.keep.push_back(uvs);
            }
            if (!m.dpds.empty() && !m.dpdt.empty()) {  // a subdivision surface's limit derivatives
                auto dpds = upload(m.dpds.data(), m.dpds.size() * sizeof(surface::V3));
                auto dpdt = upload(m.dpdt.data(), m.dpdt.size() * sizeof(surface::V3));
                attributes.dpds = reinterpret_cast<const surface::V3*>(dpds->pointer);
                attributes.dpdt = reinterpret_cast<const surface::V3*>(dpdt->pointer);
                entry.keep.push_back(dpds);
                entry.keep.push_back(dpdt);
            }
            meshAttributes.push_back(attributes);
            entry.attributes = attributes;
            if (!m.steps.empty()) {
                entry.motion = uploadMotion(m, attributes, vertexKeys, entry.keep);
                meshMotion[i] = entry.motion;
            }
            entry.keep.insert(entry.keep.end(), {vertices, indices, gas.first});
            scene_.keep.insert(scene_.keep.end(), entry.keep.begin(), entry.keep.end());
            nextCache.emplace(&m, std::move(entry));
        }
        // Shared instances (cpu::Mesh::instance): their data's structure under the instance's transform (OptiX's three rows
        // of four, for column vectors: the plan's rows vx, vy, vz and p are its columns), their records an instance's.
        for (unsigned i = 0; i < plan->meshes.size(); ++i) {
            const auto& e = plan->meshes[i];
            if (!e.instance || standIn(e)) continue;  // a stand-in's box is in world space already
            for (int r = 0; r < 3; ++r) {
                instances[i].transform[4 * r + 0] = e.objectToWorld[0 + r];
                instances[i].transform[4 * r + 1] = e.objectToWorld[3 + r];
                instances[i].transform[4 * r + 2] = e.objectToWorld[6 + r];
                instances[i].transform[4 * r + 3] = e.objectToWorld[9 + r];
            }
            shade::MeshAttributes& a = meshAttributes[i];
            a.instance = e.instance;
            for (int k = 0; k < 12; ++k) a.objectToWorld[k] = e.objectToWorld[k];
            for (int k = 0; k < 9; ++k) a.normalToWorld[k] = e.normalToWorld[k];
            for (int k = 0; k < 12; ++k) a.worldToObject[k] = e.worldToObject[k];
            if (!e.instancePrims.empty()) {
                auto prims = upload(e.instancePrims.data(), e.instancePrims.size() * sizeof(shade::PrimAttr));
                auto data = upload(e.instancePrimData.data(), std::max<size_t>(e.instancePrimData.size(), 1) * sizeof(float));
                a.instancePrims = reinterpret_cast<const shade::PrimAttr*>(prims->pointer);
                a.instancePrimData = reinterpret_cast<const float*>(data->pointer);
                a.instancePrimCount = U32(e.instancePrims.size());
                scene_.keep.insert(scene_.keep.end(), {prims, data});
            }
        }
        // Meshes the new plan no longer has leave the cache (their buffers with them once the scene lets go).
        meshCache_ = std::move(nextCache);
        instances_ = instances;
        meshAttributesHost_ = meshAttributes;
        lap("meshes");
        buildTops(*plan);
        lap("top trees");
        // Volumes the GPU's programs do not render (volume::gpuRenders): their samples are deferred to the CPU lane too.
        {
            const volume::Grids hostGrids{plan->volumeGridViews.data(), U32(plan->volumeGridViews.size()), nullptr, 0.0f};
            for (const auto& v : plan->volumes)
                if (!volume::gpuRenders(v, hostGrids)) scene_.defersVolumes = true;
        }
        if (scene_.defersVolumes && !deferred_) {  // the ring the deferred samples go to (pinned: the host reads it after a launch)
            deferredCapacity_ = 1u << 22;
            deferred_ = pinned(size_t(deferredCapacity_) * sizeof(Parked));
        }
        // The stand-ins' tree, and the ring their deferred samples go to (pinned: the host reads it after a launch).
        if (!standIns.empty()) {
            auto standBuffer = upload(standIns.data(), standIns.size() * sizeof(OptixInstance));
            OptixBuildInput sinput{};
            sinput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            sinput.instanceArray.instances = standBuffer->pointer;
            sinput.instanceArray.numInstances = unsigned(standIns.size());
            auto stop = build(sinput);
            scene_.standInTraversable = stop.second;
            scene_.keep.insert(scene_.keep.end(), {standBuffer, stop.first});
            if (!deferred_) {
                deferredCapacity_ = 1u << 22;
                deferred_ = pinned(size_t(deferredCapacity_) * sizeof(Parked));
            }
        }
        scene_.geometry = upload(geometry.data(), geometry.size() * sizeof(mixed::Geometry));
        scene_.materials = upload(materials.data(), materials.size() * sizeof(surface::Lambert));
        scene_.meshMotion = nullptr;
        for (const MeshMotion& motion : meshMotion)
            if (motion.steps || motion.pointRadius) {
                scene_.meshMotion = upload(meshMotion.data(), meshMotion.size() * sizeof(MeshMotion));
                break;
            }
        if (plan->paths) {
            // The lights with their textures' tables on the device: each Image's arrays, then the Image itself with
            // device pointers, which the light then points to.
            std::vector<lights::Light> deviceLights = plan->lights;
            lightTables_.clear();
            for (auto& l : deviceLights) {
                auto keep = [&](const void* host, size_t bytes) {
                    B b = upload(host, bytes);
                    scene_.keep.push_back(b);
                    return b->pointer;
                };
                // A mesh light's faces and sampling tree, likewise.
                if (l.mesh) {
                    const lights::MeshLightTables* tables = nullptr;
                    for (const auto& e : plan->meshLightTables)
                        if (&e.tables->view == l.mesh) tables = e.tables.get();
                    cpu::need(tables != nullptr, "A mesh light's faces are not the plan's");
                    lights::MeshLightData view = tables->view;
                    view.nodes = reinterpret_cast<const lights::MeshNode*>(keep(tables->nodes.data(), tables->nodes.size() * sizeof(lights::MeshNode)));
                    view.faces = reinterpret_cast<const lights::MeshFace*>(keep(tables->faces.data(), tables->faces.size() * sizeof(lights::MeshFace)));
                    view.vertices = reinterpret_cast<const surface::V3*>(keep(tables->vertices.data(), tables->vertices.size() * sizeof(surface::V3)));
                    const void* host = l.mesh;
                    l.mesh = reinterpret_cast<const lights::MeshLightData*>(keep(&view, sizeof(view)));
                    lightTables_[host] = l.mesh;
                }
                if (!l.image) continue;
                const lights::ImageData* data = nullptr;
                for (const auto& d : plan->lightImages)
                    if (&d->view == l.image) data = d.get();
                cpu::need(data != nullptr, "A light's texture is not the plan's");
                lights::Image view = data->view;
                view.pixels = reinterpret_cast<const float*>(keep(data->pixels.data(), data->pixels.size() * sizeof(float)));
                view.cdf = reinterpret_cast<const float*>(keep(data->cdf.data(), data->cdf.size() * sizeof(float)));
                view.guide = reinterpret_cast<const U32*>(keep(data->guide.data(), data->guide.size() * sizeof(U32)));
                view.dists = reinterpret_cast<const lights::Dist1D*>(keep(data->dists.data(), data->dists.size() * sizeof(lights::Dist1D)));
                view.levels = reinterpret_cast<const lights::ImageLevel*>(keep(data->levels.data(), data->levels.size() * sizeof(lights::ImageLevel)));
                const void* host = l.image;
                l.image = reinterpret_cast<const lights::Image*>(keep(&view, sizeof(view)));
                lightTables_[host] = l.image;
            }
            uploadLights(*plan, deviceLights);
            if (!plan->volumes.empty()) {
                scene_.volumes = upload(plan->volumes.data(), plan->volumes.size() * sizeof(volume::Shader));
                uploadGrids(*plan);
                uploadRegions(*plan);
                std::vector<int> meshVolumes;
                for (const auto& mesh : plan->meshes) meshVolumes.push_back(mesh.volume);
                scene_.meshVolumes = upload(meshVolumes.data(), meshVolumes.size() * sizeof(int));
            }
            if (plan->walkSequences)
                scene_.walkSequences = upload(plan->walkSequences->data(), plan->walkSequences->size() * sizeof(float));
            scene_.meshMaterials = upload(meshMaterials.data(), meshMaterials.size() * sizeof(shade::MaterialRecord));
            scene_.meshAttributes = upload(meshAttributes.data(), meshAttributes.size() * sizeof(shade::MeshAttributes));
            const std::vector<float> pool = plan->materialPool.empty() ? std::vector<float>{0.0f} : plan->materialPool;
            scene_.materialPool = upload(pool.data(), pool.size() * sizeof(float));
            // The render outputs' AOVs (aov_core.h): the schema's tables on the device, each mesh's geometry label, and
            // a sample's values and closest depths in its thread's scratch.
            aov::Tables t;
            aovChannels_ = aovClosest_ = aovClosestChannels_ = 0;
            aovEntries_.clear();
            aovScratch_ = 0;
            if (plan->aovs) {
                const aov::Plan& a = *plan->aovs;
                t = a.tables();
                t.primKeys = plan->aovPrimKeys.data();
                t.primCount = U32(plan->aovPrimKeys.size());
                auto put = [&](auto& pointer, size_t count) {
                    using T = std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>>;
                    if (!pointer || !count) {
                        pointer = nullptr;
                        return;
                    }
                    B b = upload(pointer, count * sizeof(T));
                    scene_.keep.push_back(b);
                    pointer = reinterpret_cast<const T*>(b->pointer);
                };
                put(t.entries, t.count);
                put(t.lpe, size_t(t.lpeStates) * t.lpeStride);
                put(t.lpeValid, size_t(t.lpeStates) * t.lpeWords);
                put(t.materials, t.materialCount);
                put(t.labelLists, a.labelLists.size());
                put(t.primKeys, t.primCount);
                std::vector<int> labels;
                for (const auto& mesh : plan->meshes) labels.push_back(mesh.geomLabel);
                if (!labels.empty()) scene_.geomLabels = upload(labels.data(), labels.size() * sizeof(int));
                aovChannels_ = t.channels;
                aovClosest_ = t.closestCount;
                aovClosestChannels_ = 0;
                aovEntries_.assign(a.entries.begin(), a.entries.end());
                for (const aov::Entry& e : aovEntries_)
                    if (e.filter == aov::FilterClosest) aovClosestChannels_ += e.channels;
                if (aovChannels_) aovScratch_ = (size_t(2 * aovChannels_ + aovClosest_) * sizeof(float) + 15) & ~size_t(15);
            }
            scene_.aovTables = upload(&t, sizeof(t));
            // Cryptomatte's events a thread: a camera hit and two a presence layer.
            cryptoEvents_ = 0;
            cryptoScratch_ = 0;
            if (plan->aovs && plan->aovs->cryptomatte) {
                cryptoEvents_ = std::min<U32>(aov::CryptoMaxEvents, 2 * plan->maxPresenceDepth + 3);
                cryptoScratch_ = (size_t(cryptoEvents_) * sizeof(aov::CryptoEvent) + 15) & ~size_t(15);
            }
            movedLightSlots_ = U32(std::min(plan->lightMotion.size(), MovedLightSlots));
            movedLightsScratch_ = (movedLightSlots_ * sizeof(lights::Light) + (movedLightSlots_ + 1) * sizeof(U32) + 15) & ~size_t(15);
            if (!movedLightSlots_) movedLightsScratch_ = 0;
            // Two runs of map registers (a surface's, and its bump's shifted states), a third where the card runs a
            // mapped volume's maps (volume_core.h's GpuVolumeMaps: deviceVolumeMaps).
            bool volumeMaps = false;
            for (const auto& v : plan->volumes) volumeMaps = volumeMaps || (volume::GpuVolumeMaps && volume::pointMaps(v) && !v.mapsOnHost);
            volumeMaps_ = volumeMaps;
            registersScratch_ = (size_t((volumeMaps ? 3 : 2) * plan->mapRegisters) * sizeof(shade::Color) + 15) & ~size_t(15);
            const size_t threadScratch = scratchStride(plan->dwaSlots, plan->glitter) + aovScratch_ + cryptoScratch_ + movedLightsScratch_ + registersScratch_;
            // The launch the card has room for: the scratch at most a quarter of what is free, halved until it is.
            options_.launchThreads = cardThreads_;
            {
                size_t freeNow = 0, totalNow = 0;
                if (memoryInfo(freeNow, totalNow))
                    while (options_.launchThreads > MinLaunchThreads && size_t(options_.launchThreads) * threadScratch > freeNow / 4)
                        options_.launchThreads /= 2;
            }
            if (std::getenv("XPU2_ASYNC_STACKS"))
                std::fprintf(stderr, "scratch: threads %u (card %u) of %zu bytes: %zu MB\n", options_.launchThreads, cardThreads_, threadScratch,
                             (size_t(options_.launchThreads) * threadScratch) >> 20);
            gpu::Allocating what("each launch thread's scratch (materials, outputs, Cryptomatte)");
            scene_.scratch = device(size_t(options_.launchThreads) * threadScratch);
        }
        lap("lights, volumes, materials, outputs");
        uploadConfig(plan);
        if (cpu::gpuTextures(plan->domain)) prepareTextures(*plan);
        lap("textures");
        const size_t pixels = size_t(plan->width) * plan->height;
        gpu::Allocating films("the picture's films and outputs");
        scene_.sums = device(pixels * 4 * sizeof(double));
        scene_.odd = device(pixels * 4 * sizeof(double));
        scene_.counts = device(pixels * sizeof(U32));
        // The provisional film (partial shading): the provisional samples' sums and counts a pixel.
        scene_.provSums = device(pixels * 4 * sizeof(double));
        scene_.provCounts = device(pixels * sizeof(U32));
        // The AOV film (floats, as MoonRay's), the closest filter's words and the depths read from them, and the heat
        // map's seconds.
        if (aovChannels_) {
            scene_.aovs = device(pixels * aovChannels_ * sizeof(float));
            scene_.hostAovs = pinned(pixels * aovChannels_ * sizeof(float));
            if (aovClosestChannels_) {
                scene_.aovClosest = device(pixels * aovClosestChannels_ * sizeof(U64L));
                scene_.hostAovClosest = pinned(pixels * aovClosestChannels_ * sizeof(U64L));
                scene_.hostAovDepths = pinned(pixels * aovClosest_ * sizeof(float));
            }
        }
        if (plan->aovs && plan->aovs->heatMap) {
            scene_.heat = device(pixels * sizeof(double));
            scene_.hostHeat = pinned(pixels * sizeof(double));
            CUdevice dev;
            gpu::Driver::check(driver.cuDeviceGet_p(&dev, driver.ordinal));
            int kilohertz = 0;
            gpu::Driver::check(driver.cuDeviceGetAttribute_p(&kilohertz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
            secondsPerTick_ = kilohertz > 0 ? 1.0 / (double(kilohertz) * 1000.0) : 0.0;
        }
        scene_.aovInitDone = std::make_shared<gpu::Event>(driver);
        // Cryptomatte's fragment lists: a head a pixel and category, a lock a pixel, and the node pool (four fragments a
        // pixel and category; extras when an output writes positions, normals, beauty and the like).
        if (cryptoEvents_) {
            const aov::Plan& a = *plan->aovs;
            cryptoCategories_ = a.cryptoCategories;
            cryptoMultiPresence_ = a.cryptoMultiPresence ? 1u : 0u;
            cryptoExtras_ = 0;
            for (const aov::Output* o : a.all)
                if (o->result == aov::ResultCryptomatte)
                    for (int k = 0; k < 7; ++k) cryptoExtras_ |= o->cryptoExtra[k] ? 1u : 0u;
            cryptoNodeWords_ = cryptoExtras_ ? 26u : 5u;
            cryptoNodeCapacity_ = U32(std::min<size_t>(pixels * cryptoCategories_ * 4 + (size_t(1) << 20), size_t(0x7fffffff) / cryptoNodeWords_));
            scene_.cryptoHeads = device(pixels * cryptoCategories_ * sizeof(U32));
            gpu::Allocating what("Cryptomatte's fragments");
            scene_.cryptoNodes = device(size_t(cryptoNodeCapacity_) * cryptoNodeWords_ * sizeof(U32));
            scene_.cryptoNodeCount = device(sizeof(U32));
        }
        scene_.hostSums = pinned(pixels * 4 * sizeof(double));
        scene_.hostOdd = pinned(pixels * 4 * sizeof(double));
        scene_.hostCounts = pinned(pixels * sizeof(U32));
        scene_.hostProvSums = pinned(pixels * 4 * sizeof(double));
        scene_.hostProvCounts = pinned(pixels * sizeof(U32));
        scene_.filmDone = std::make_shared<gpu::Event>(driver);
        scene_.previewDone = std::make_shared<gpu::Event>(driver);
        const U32 tiles = ((plan->width + TileSize - 1) / TileSize) * ((plan->height + TileSize - 1) / TileSize);
        for (U32 s = 0; s < PassSlots; ++s) {
            scene_.passMask[s] = device(size_t(tiles) * sizeof(U64L));
            scene_.passBase[s] = device(pixels * sizeof(U32));
            scene_.passStaging[s] = pinned(size_t(tiles) * sizeof(U64L));
        }
        if (plan->adaptive) {
            scene_.errors = device(pixels * sizeof(float));
            scene_.hostErrors = pinned(pixels * sizeof(float));
            scene_.errorsDone = std::make_shared<gpu::Event>(driver);
        }
        passDone_.assign(MaxPasses, 0u);
        driver.drain();  // startup only: everything above has landed
        uploads_.clear();
        lap("films, and everything landed");
        {
            size_t totalBytes = 0;
            if (!memoryInfo(freeAfterPrepare_, totalBytes)) freeAfterPrepare_ = 0;
        }
    }
    std::vector<std::pair<const char*, double>> prepareLaps_;

    // The film's rows a frame writes (a share of a frame, Rhizome: its region's rows): cleared as it begins and downloaded
    // as it ends, the rest left as they are (the next frames' shares clear their own). Unset: every row.
    void filmRows(U32 first, U32 end) {
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { f.filmRows(first, end); });
        rowFirst_ = first, rowEnd_ = end;
    }
    // A band of the film: its first pixel and its pixels.
    size_t bandStart() const { return size_t(std::min<U32>(rowFirst_, plan_->height)) * plan_->width; }
    size_t bandPixels() const {
        const U32 a = std::min<U32>(rowFirst_, plan_->height), e = std::min<U32>(rowEnd_, plan_->height);
        return e > a ? size_t(e - a) * plan_->width : 0;
    }
    void copyBand(const std::shared_ptr<gpu::Buffer>& h, const std::shared_ptr<gpu::Buffer>& d, size_t bytesPerPixel) {
        const size_t at = bandStart() * bytesPerPixel;
        gpu::Driver::check(driver.cuMemcpyDtoHAsync_v2_p(static_cast<char*>(h->host()) + at, d->pointer + at, bandPixels() * bytesPerPixel, stream_));
    }
    void clearBand(const std::shared_ptr<gpu::Buffer>& d, size_t wordsPerPixel) { memset32(d->pointer + bandStart() * wordsPerPixel * 4, 0, bandPixels() * wordsPerPixel); }
    U32 rowFirst_ = 0, rowEnd_ = ~0u;

    // Starts a frame: an empty film and empty queues. partial: texture misses finish from a coarser level as
    // provisional samples (never in the pass chain, which runs on the device alone).
    void begin(U32 samplesPerChunk, bool adaptive = false, bool partial = false) {
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { f.begin(samplesPerChunk, adaptive, partial); });
        cpu::need(plan_ && inFlight_.empty(), "GPU frame overlap");
        cpu::need(!adaptive || single_ || programs_[1].pipeline, "No adaptive GPU program");
        program_ = &programs_[adaptive && !single_ ? 1 : 0];
        // A scene with volumes whose maps the card runs: the program with the mapped volume module, made the first time.
        if (volumeMaps_ && program_ == &programs_[0] && programs_[0].modules[VolumeMapsModule]) {
            if (!mapped_.pipeline) {
                OptixModule modules[ModuleCount];
                for (int i = 0; i < ModuleCount; ++i) modules[i] = programs_[0].modules[i];
                modules[VolumeModule] = programs_[0].modules[VolumeMapsModule];
                mapped_.create(*this, modules);
                mapped_.ownsModules = false;  // programs_[0]'s
            }
            program_ = &mapped_;
        }
        itemsPerChunk_ = samplesPerChunk * TilePixels;
        adaptive_ = adaptive;
        sizeParked(partial);
        program_->fitStack(!plan_->volumes.empty(), plan_->subsurface);
        clearBand(scene_.sums, 8);
        clearBand(scene_.odd, 8);
        std::fill(passDone_.begin(), passDone_.end(), 0u);
        clearBand(scene_.counts, 1);
        clearBand(scene_.provSums, 8);
        clearBand(scene_.provCounts, 1);
        partial_ = partial;
        std::fill(profile_, profile_ + ProfileStages, 0ull);
        refineHead_ = 0;
        refinesOut_.clear();
        // With the device's state, whose count of deferred samples starts again: its second frame's were never taken for
        // the CPU lane, and a scene with motion blur or a volume only the CPU renders waited on them for good.
        deferredHead_ = 0;
        deferredOut_.clear();
        memset32(state_->pointer, 0, sizeof(DeviceState) / 4);
        if (aovChannels_ || scene_.heat || cryptoEvents_) {  // mode 4: the AOV film readied
            LaunchParams p = baseParams();
            p.mode = 4;
            auto& slot = takeParams(p);
            ox(optixLaunch(program_->pipeline, stream_, slot.device->pointer, sizeof(LaunchParams), &program_->sbt, p.pixels, 1, 1));
            gpu::Driver::check(driver.cuEventRecord_p(scene_.aovInitDone->value, stream_));
            slot.done = scene_.aovInitDone;
        }
        published_ = 0;
        parkedTail_ = parkedIssued_ = 0;
        // The device's request ring restarts with its state; requested tiles not
        // yet loaded stay pending, and resident ones stay resident across frames.
        fine_.requestHead = 0;
        launches_ = 0;
        completed_ = {};
    }

    // Whether another launch may be queued now (at most two in flight).
    bool ready() const { return inFlight_.size() < 2 && paramsFree(); }
    unsigned inFlight() const { return unsigned(inFlight_.size()); }
    U64 parkedOutstanding() const { return parkedTail_ - parkedIssued_; }
    U64 maxChunksPerLaunch() const { return options_.maxChunksPerLaunch; }

    // Publishes chunks through the mailbox ring and queues a launch that runs
    // them, after any provisional samples to render again in full (refines,
    // refineCount of them) and parked samples whose tiles have landed. Returns
    // false when there is nothing to run.
    bool launch(const Chunk* chunks, U32 count, const Refine* refines = nullptr, U32 refineCount = 0, U64 samples = 0) {
        cpu::need(ready() && count <= options_.maxChunksPerLaunch, "GPU launch backpressure");
        cpu::need(refineCount <= RefineInCapacity, "Too many provisional samples for one launch");
        // Tiles asked for so far land first, so every sample parked up to the last
        // readback may run again; one whose tile is still waiting parks again.
        commitTiles();
        const U64 parkedBegin = parkedIssued_, parkedEnd = parkedTail_;
        if (!count && parkedBegin == parkedEnd && !refineCount) return false;
        // Each sample this launch runs makes at most one provisional sample, on top of those the launches in flight may.
        const U64 mayRefine = partial_ ? U64(count) * itemsPerChunk_ + (parkedEnd - parkedBegin) + refineCount : 0;
        cpu::need(refineBound() + mayRefine <= refineCapacity_, "Provisional-sample ring would overflow");
        // The bounded queue: whatever this launch runs could park again, on top of
        // every launch still in flight (the host sizes launches, MEM-05).
        const U64 items = U64(count) * itemsPerChunk_ + (parkedEnd - parkedBegin);
        cpu::need(!mayPark_ || parkedBound() - parkedBegin + items <= parkedCapacity_, "Parked-sample queue would overflow");
        // The ring may not overwrite chunks a queued launch still reads.
        cpu::need(published_ + count - oldestChunk() <= options_.ringChunks, "Mailbox ring full");
        const unsigned s = U32(launches_ % Slots);
        auto& staging = staging_[launches_ % staging_.size()];
        std::memcpy(staging->host(), chunks, size_t(count) * sizeof(Chunk));
        for (U32 done = 0; done < count;) {  // the ring wraps
            const U64 at = (published_ + done) % options_.ringChunks;
            const U32 n = U32(std::min<U64>(count - done, options_.ringChunks - at));
            gpu::Driver::check(driver.cuMemcpyHtoDAsync_v2_p(ring_->pointer + at * sizeof(Chunk),
                                                             static_cast<char*>(staging->host()) + size_t(done) * sizeof(Chunk),
                                                             size_t(n) * sizeof(Chunk), stream_));
            done += n;
        }
        memset32(state_->pointer + offsetof(DeviceState, claimed) + s * sizeof(U32), 0, 1);
        memset32(state_->pointer + offsetof(DeviceState, unparked) + s * sizeof(U32), 0, 1);
        memset32(state_->pointer + offsetof(DeviceState, refineClaimed) + s * sizeof(U32), 0, 1);
        // The slot's provisional samples: read by the device from pinned memory, which the launch four later reuses
        // (at most two are in flight).
        if (refineCount) std::memcpy(refineIn_[s]->host(), refines, size_t(refineCount) * sizeof(Refine));

        LaunchParams p = baseParams();
        p.slot = s;
        p.refineIn = reinterpret_cast<const Refine*>(refineIn_[s]->pointer);
        p.refineInCount = refineCount;
        p.itemBegin = published_ * itemsPerChunk_;
        p.itemEnd = (published_ + count) * itemsPerChunk_;
        p.parkedBegin = parkedBegin;
        p.parkedEnd = parkedEnd;
        p.mode = 0;
        auto& slot = takeParams(p);
        auto& r = readbacks_[launches_ % readbacks_.size()];
        if (!r.begun) {
            r.begun = std::make_shared<TimingEvent>(driver);
            r.ended = std::make_shared<TimingEvent>(driver);
        }
        gpu::Driver::check(driver.cuEventRecord_p(r.begun->value, stream_));
        ox(optixLaunch(program_->pipeline, stream_, slot.device->pointer, sizeof(LaunchParams), &program_->sbt,
                       options_.launchThreads, 1, 1));
        gpu::Driver::check(driver.cuEventRecord_p(r.ended->value, stream_));

        readBack(r);
        slot.done = r.done;
        Flight flight{launches_, published_, count, parkedEnd - parkedBegin, items, gpu::StartClock::now(), &r};
        flight.refines = refineCount;
        flight.mayRefine = mayRefine;
        flight.samples = samples;
        inFlight_.push_back(flight);
        published_ += count;
        parkedIssued_ = parkedEnd;
        ++launches_;
        return true;
    }

    // Collects the oldest launch if it has finished; never waits.
    bool poll(Completion& out) {
        if (inFlight_.empty() || !inFlight_.front().readback->done->ready()) return false;
        const auto f = inFlight_.front();
        inFlight_.pop_front();
        const auto& state = *static_cast<const DeviceState*>(f.readback->state->host());
        out.chunks = f.chunks;
        out.items = U64(f.chunks) * itemsPerChunk_;
        out.parkedRun = f.parked;
        out.refinesRun = f.refines;
        out.samples = f.samples;
        out.errors = state.errors;
        std::copy(state.profile, state.profile + ProfileStages, profile_);  // the frame's so far
        if (state.exceptionIndex && !exceptionLogged_) {
            exceptionLogged_ = true;
            std::fprintf(stderr, "XPU2: OptiX exception %d at launch index %u\n", int(state.exceptionCode), state.exceptionIndex - 1);
        }
        // The provisional samples this launch made (the ring is pinned: they are here once the launch is done).
        if (state.refineTail > refineHead_) {
            if (state.refineTail - refineHead_ > refineCapacity_) out.errors += 1;  // overflowed: the frame falls back
            const Refine* ring = static_cast<const Refine*>(refines_->host());
            for (U64 k = refineHead_; k < state.refineTail; ++k) refinesOut_.push_back(ring[k % refineCapacity_]);
            refineHead_ = state.refineTail;
        }
        // The samples this launch deferred to the CPU lane (the ring is pinned: they are here once the launch is done).
        if (deferred_ && state.deferredTail > deferredHead_) {
            if (state.deferredTail - deferredHead_ > deferredCapacity_) out.errors += 1;  // overflowed: the frame falls back
            const Parked* ring = static_cast<const Parked*>(deferred_->host());
            for (U64 k = deferredHead_; k < state.deferredTail; ++k) deferredOut_.push_back(ring[k % deferredCapacity_]);
            deferredHead_ = state.deferredTail;
        }
        float ms = 0;
        gpu::Driver::check(driver.cuEventElapsedTime_p(&ms, f.readback->begun->value, f.readback->ended->value));
        out.deviceSeconds = ms / 1000.0;
        parkedTail_ = state.parkedTail;
        // A frame sized not to park (sizeParked) that did: its small queue may have wrapped, so the frame is not trusted.
        cpu::need(mayPark_ || parkedTail_ - parkedIssued_ <= parkedCapacity_, "A sample parked in a frame sized not to park");
        if (adaptive_) std::copy(state.passDone, state.passDone + MaxPasses, passDone_.begin());
        if (f.chain) {
            chainState_ = state;
            ++chainsCollected_;
            // Seconds in each stage of the chain, summed over its steps.
            for (size_t i = 1; i < chain_.marks.size(); ++i) {
                float stage = 0;
                gpu::Driver::check(driver.cuEventElapsedTime_p(&stage, chain_.marks[i - 1]->value, chain_.marks[i]->value));
                chainStages_[(i - 1) % 4] += stage / 1000.0;
                if ((i - 1) % 4 == 3) chainRenders_.push_back(stage / 1000.0);
            }
        }
        // Samples parked by this launch may run once their tiles are committed,
        // which the next launch does before it starts (commitTiles).
        if (fine_.tiles) {
            const auto* requests = static_cast<const U32*>(f.readback->requests->host());
            for (U64 i = fine_.requestHead; i < state.requestTail; ++i)
                fine_.pending.push_back(requests[i % fine_.requestCapacity]);
            fine_.requestHead = state.requestTail;
            const auto* touched = static_cast<const unsigned char*>(f.readback->touched->host());
            for (U32 t = 1; t <= fine_.tiles; ++t)
                if (touched[t]) fine_.lastUse[t] = f.launch + 1;
        }
        completed_ = out;
        return true;
    }

    // Queues the device film's download; filmReady() says when it has landed.
    void downloadFilm() {
        if (!followers_.empty()) forward([&](GpuLane& f) { f.downloadFilm(); });
        filmAdded_ = false;
        cpu::need(inFlight_.empty(), "Film download before the last launch was collected");
        const size_t pixels = size_t(plan_->width) * plan_->height;
        copyBand(scene_.hostSums, scene_.sums, 4 * sizeof(double));
        copyBand(scene_.hostCounts, scene_.counts, sizeof(U32));
        copyBand(scene_.hostProvSums, scene_.provSums, 4 * sizeof(double));
        copyBand(scene_.hostProvCounts, scene_.provCounts, sizeof(U32));
        if (adaptive_) copyBand(scene_.hostOdd, scene_.odd, 4 * sizeof(double));
        if (aovChannels_) copyBand(scene_.hostAovs, scene_.aovs, aovChannels_ * sizeof(float));
        downloadClosest();
        if (scene_.heat) copyBand(scene_.hostHeat, scene_.heat, sizeof(double));
        if (chainsLaunched_) copyOut(chain_.hostPlanned, chain_.planned, pixels * sizeof(U32));
        gpu::Driver::check(driver.cuEventRecord_p(scene_.filmDone->value, stream_));
    }
    bool filmReady() {
        if (!scene_.filmDone->ready()) return false;
        if (followers_.empty() || filmAdded_) return true;
        if (!followersReady([](GpuLane& f) { return f.filmReady(); })) return false;
        addFollowers(true);
        filmAdded_ = true;
        return true;
    }
    // A copy of the device film as it will stand once the launches queued so far have finished, for a progressive
    // picture; previewReady() says when it has landed. Read through sums() and counts(); the frame's own download
    // (downloadFilm) writes the same copy.
    void downloadPreview(bool aovs = false) {
        if (!followers_.empty()) forward([&](GpuLane& f) { f.downloadPreview(aovs); });
        previewAdded_ = false;
        const size_t pixels = size_t(plan_->width) * plan_->height;
        copyOut(scene_.hostSums, scene_.sums, pixels * 4 * sizeof(double));
        copyOut(scene_.hostCounts, scene_.counts, pixels * sizeof(U32));
        copyOut(scene_.hostProvSums, scene_.provSums, pixels * 4 * sizeof(double));
        copyOut(scene_.hostProvCounts, scene_.provCounts, pixels * sizeof(U32));
        if (aovs) {  // a preview of an AOV: its film too
            if (aovChannels_) copyOut(scene_.hostAovs, scene_.aovs, pixels * aovChannels_ * sizeof(float));
            downloadClosest();
            if (scene_.heat) copyOut(scene_.hostHeat, scene_.heat, pixels * sizeof(double));
        }
        gpu::Driver::check(driver.cuEventRecord_p(scene_.previewDone->value, stream_));
    }
    bool previewReady() {
        if (!scene_.previewDone->ready()) return false;
        if (followers_.empty() || previewAdded_) return true;
        if (!followersReady([](GpuLane& f) { return f.previewReady(); })) return false;
        addFollowers(false);
        previewAdded_ = true;
        return true;
    }
    template <class F>
    bool followersReady(F&& ready) {
        bool all = true;
        for (const auto& f : followers_) {
            f->use();
            all = ready(*f) && all;
        }
        use();
        return all;
    }
    // The followers' downloaded films added into this lane's host copy: sums, counts, the odd sums of a whole film
    // (adaptive) and the render times. Their AOVs are merged by the renderer, which holds the outputs' filters
    // (async_render.h, addCardAovs).
    void addFollowers(bool whole) {
        // The film's band (filmRows): the rows the frame wrote, the others as they were.
        const size_t from = bandStart(), pixels = from + bandPixels();
        double* sums = static_cast<double*>(scene_.hostSums->host());
        U32* counts = static_cast<U32*>(scene_.hostCounts->host());
        for (const auto& f : followers_) {
            const double* s = f->sums();
            const U32* c = f->counts();
            for (size_t i = from * 4; i < pixels * 4; ++i) sums[i] += s[i];
            for (size_t i = from; i < pixels; ++i) counts[i] += c[i];
            if (whole && adaptive_ && scene_.hostOdd && f->scene_.hostOdd) {
                double* odd = static_cast<double*>(scene_.hostOdd->host());
                const double* o = f->odd();
                for (size_t i = from * 4; i < pixels * 4; ++i) odd[i] += o[i];
            }
            if (whole && scene_.hostHeat && f->scene_.hostHeat) {
                double* heat = static_cast<double*>(scene_.hostHeat->host());
                const double* h = f->heat();
                for (size_t i = from; i < pixels; ++i) heat[i] += h[i];
            }
        }
    }
    // The closest filter's words downloaded with the AOV film; read apart into its values and depths when the film is
    // next read (landClosest): a word's high half the depth's ordered bits (mergeAovs), its low half the value's.
    void downloadClosest() {
        if (!aovClosestChannels_) return;
        copyBand(scene_.hostAovClosest, scene_.aovClosest, aovClosestChannels_ * sizeof(U64L));
        closestLanded_ = false;
    }
    void landClosest() const {
        if (closestLanded_ || !aovClosestChannels_) return;
        closestLanded_ = true;
        const size_t pixels = bandStart() + bandPixels();
        const U64L* words = static_cast<const U64L*>(scene_.hostAovClosest->host());
        float* values = static_cast<float*>(scene_.hostAovs->host());
        float* depths = static_cast<float*>(scene_.hostAovDepths->host());
        auto f = [](U32 w) {
            float v;
            std::memcpy(&v, &w, sizeof v);
            return v;
        };
        for (size_t p = bandStart(); p < pixels; ++p) {
            const U64L* k = words + p * aovClosestChannels_;
            U32 at = 0;
            for (const aov::Entry& e : aovEntries_) {
                if (e.filter != aov::FilterClosest) continue;
                float depth = aov::infinity();
                for (U32 c = 0; c < e.channels; ++c) {
                    const U64L w = k[at + c];
                    if (w == ~0ull) continue;  // no sample set it: the film's own 0 and +inf
                    const U32 o = U32(w >> 32);
                    depth = f(o & 0x80000000u ? o & 0x7fffffffu : ~o);
                    values[p * aovChannels_ + e.offset + c] = f(U32(w));
                }
                depths[p * aovClosest_ + e.closest] = depth;
                at += e.channels;
            }
        }
    }
    const double* sums() const { return static_cast<const double*>(scene_.hostSums->host()); }
    const double* odd() const { return static_cast<const double*>(scene_.hostOdd->host()); }
    const U32* counts() const { return static_cast<const U32*>(scene_.hostCounts->host()); }
    // The downloaded provisional film (partial shading): its sums and counts a pixel, before the CPU lane's renders
    // in full of its samples are taken out (HostFilm::settle).
    const double* provSums() const { return static_cast<const double*>(scene_.hostProvSums->host()); }
    const U32* provCounts() const { return static_cast<const U32*>(scene_.hostProvCounts->host()); }
    // The provisional samples made since the last call, for the host's queue (async_render.h).
    void takeRefines(std::vector<Refine>& out) {
        out.insert(out.end(), refinesOut_.begin(), refinesOut_.end());
        refinesOut_.clear();
    }
    // Whether a fine-detail tile is resident (0: not known, never), so a provisional sample needing it may go back
    // to the GPU; landTiles() commits those asked for so far first.
    bool tileResident(U32 tile) const { return tile && tile < fine_.tileSlot.size() && fine_.tileSlot[tile] != NoSlot; }  // flat ones too
    void landTiles() { commitTiles(); }
    bool partial() const { return partial_; }
    // The GPU profile (XPU2_PROFILE): the frame's clock ticks in each stage (ProfileStage), as of the last launch collected.
    const U64L* profile() const { return profile_; }
    // The chunks a launch with refineCount provisional samples may take without overflowing their ring.
    U32 chunkRoom(U32 refineCount) const {
        if (!partial_) return options_.maxChunksPerLaunch;
        const U64 used = refineBound() + refineCount + (parkedTail_ - parkedIssued_);
        return used >= refineCapacity_ ? 0u : U32(std::min<U64>(options_.maxChunksPerLaunch, (refineCapacity_ - used) / itemsPerChunk_));
    }
    static constexpr U32 RefineInCapacity = 1u << 17;  // provisional samples one launch renders again
    U32 launchThreads() const { return options_.launchThreads; }
    // The downloaded AOV film (null: none): the accumulators a pixel, the closest filter's depths, the seconds.
    const float* aovs() const {
        landClosest();
        return scene_.hostAovs ? static_cast<const float*>(scene_.hostAovs->host()) : nullptr;
    }
    const float* aovDepths() const {
        landClosest();
        return scene_.hostAovDepths ? static_cast<const float*>(scene_.hostAovDepths->host()) : nullptr;
    }
    const double* heat() const { return scene_.hostHeat ? static_cast<const double*>(scene_.hostHeat->host()) : nullptr; }
    // The device's Cryptomatte fragments into the host's buffer (after the film has landed; waits for the copies).
    void mergeCrypto(aov::CryptoBuffer& into) {
        if (!followers_.empty()) forward([&](GpuLane& f) { f.mergeCrypto(into); });
        if (!cryptoEvents_ || into.empty()) return;
        const size_t pixels = size_t(plan_->width) * plan_->height;
        B countHost = pinned(sizeof(U32)), headsHost = pinned(pixels * cryptoCategories_ * sizeof(U32));
        copyOut(countHost, scene_.cryptoNodeCount, sizeof(U32));
        copyOut(headsHost, scene_.cryptoHeads, pixels * cryptoCategories_ * sizeof(U32));
        driver.drain();  // the frame's end
        const U32 count = std::min(*static_cast<const U32*>(countHost->host()), cryptoNodeCapacity_);
        const U32* heads = static_cast<const U32*>(headsHost->host());
        B nodesHost;
        if (count) {
            nodesHost = pinned(size_t(count) * cryptoNodeWords_ * sizeof(U32));
            copyOut(nodesHost, scene_.cryptoNodes, size_t(count) * cryptoNodeWords_ * sizeof(U32));
            driver.drain();
        }
        const U32* nodes = count ? static_cast<const U32*>(nodesHost->host()) : nullptr;
        auto f = [](U32 w) {
            float v;
            std::memcpy(&v, &w, sizeof v);
            return v;
        };
        for (size_t p = 0; p < pixels; ++p)
            for (U32 c = 0; c < cryptoCategories_ && c < into.categories(); ++c) {
                // The list holds the newest first: merged oldest first, as the host's would have kept them.
                std::vector<U32> order;
                for (U32 n = heads[p * cryptoCategories_ + c]; n != ~0u && n < count; n = nodes[size_t(n) * cryptoNodeWords_]) order.push_back(n);
                for (auto it = order.rbegin(); it != order.rend(); ++it) {
                    const U32* m = nodes + size_t(*it) * cryptoNodeWords_;
                    aov::CryptoBuffer::Fragment g{f(m[1]), f(m[2]), {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0}, m[4], m[3]};
                    if (cryptoExtras_) {
                        const U32* x = m + 5;
                        g.position = V3{f(x[0]), f(x[1]), f(x[2])};
                        g.p0 = V3{f(x[3]), f(x[4]), f(x[5])};
                        g.normal = V3{f(x[6]), f(x[7]), f(x[8])};
                        for (int k = 0; k < 4; ++k) g.beauty[k] = f(x[9 + k]);
                        g.refP = V3{f(x[13]), f(x[14]), f(x[15])};
                        g.refN = V3{f(x[16]), f(x[17]), f(x[18])};
                        g.uv[0] = f(x[19]);
                        g.uv[1] = f(x[20]);
                    }
                    into.merge(p, c, g);
                }
            }
    }

    // Adaptive frames. Every pixel's error, worked out on the device behind the
    // launches already queued (they finish first, on the same stream); only the
    // errors cross to the host, for its error trees.
    void requestErrors() {
        LaunchParams p = baseParams();
        p.mode = 2;
        auto& slot = takeParams(p);
        ox(optixLaunch(program_->pipeline, stream_, slot.device->pointer, sizeof(LaunchParams), &program_->sbt, p.pixels, 1, 1));
        copyOut(scene_.hostErrors, scene_.errors, size_t(p.pixels) * sizeof(float));
        if (countsWithErrors_) {  // to pool them: the film's sums, odd sums and counts, apart from the pictures' copy
            const size_t pixels = p.pixels;
            if (!errorCounts_ || errorCounts_->bytes != pixels * sizeof(U32)) {
                errorSums_ = pinned(pixels * 4 * sizeof(double));
                errorOdd_ = pinned(pixels * 4 * sizeof(double));
                errorCounts_ = pinned(pixels * sizeof(U32));
            }
            copyOut(errorSums_, scene_.sums, pixels * 4 * sizeof(double));
            copyOut(errorOdd_, scene_.odd, pixels * 4 * sizeof(double));
            copyOut(errorCounts_, scene_.counts, pixels * sizeof(U32));
        }
        gpu::Driver::check(driver.cuEventRecord_p(scene_.errorsDone->value, stream_));
        slot.done = scene_.errorsDone;
        if (!followers_.empty()) {
            forward([&](GpuLane& f) { f.requestErrors(); });
            errorsPooled_ = false;
        }
    }
    bool errorsReady() {
        if (!scene_.errorsDone->ready()) return false;
        if (followers_.empty() || errorsPooled_) return true;
        if (!followersReady([](GpuLane& f) { return f.errorsReady(); })) return false;
        poolErrors();
        errorsPooled_ = true;
        return true;
    }
    // The cards' errors as one card's: each pixel's sums, odd sums and counts added over the cards, and its error worked
    // out from them as the device works out its own (pixelError), so it is the error one card with all the samples
    // would have found. (Pooling each card's error by its count was noisier: each card's error rests on its share of a
    // pixel's samples alone, and the samples went to the wrong places; three simulated cards came out past the noise.)
    void poolErrors() {
        const size_t pixels = size_t(plan_->width) * plan_->height;
        pooled_.resize(pixels);
        const double* sums = static_cast<const double*>(errorSums_->host());
        const double* odd = static_cast<const double*>(errorOdd_->host());
        const U32* counts = errorCounts();
        for (size_t i = 0; i < pixels; ++i) {
            double total[4], halves[4];
            U32 n = counts[i];
            for (int c = 0; c < 4; ++c) {
                total[c] = sums[i * 4 + c];
                halves[c] = odd[i * 4 + c];
            }
            for (const auto& f : followers_) {
                const double* s = static_cast<const double*>(f->errorSums_->host());
                const double* o = static_cast<const double*>(f->errorOdd_->host());
                for (int c = 0; c < 4; ++c) {
                    total[c] += s[i * 4 + c];
                    halves[c] += o[i * 4 + c];
                }
                n += f->errorCounts()[i];
            }
            pooled_[i] = pixelError(total, halves, n);
        }
    }
    const U32* errorCounts() const { return static_cast<const U32*>(errorCounts_->host()); }
    const float* errorsOwn() const { return static_cast<const float*>(scene_.hostErrors->host()); }
    const float* errors() const { return followers_.empty() ? errorsOwn() : pooled_.data(); }
    // A pass's tables, ahead of any launch that reads them: its tile masks, and
    // each pixel's sample-index base, made on the device. A pass of the uniform
    // stage gives every pixel the same base (NoSlot: none); a later pass starts
    // from the device's counts, which hold every earlier pass once they have all
    // completed. A slot's staging is reused only once its earlier pass has
    // completed, so its upload has.
    void uploadPass(U32 slot, const std::vector<std::uint64_t>& masks, U32 uniformBase) {
        if (!followers_.empty()) forward([&](GpuLane& f) { f.uploadPass(slot, masks, uniformBase); });
        void* staging = scene_.passStaging[slot]->host();
        const size_t maskBytes = masks.size() * sizeof(U64L), pixels = size_t(plan_->width) * plan_->height;
        std::memcpy(staging, masks.data(), maskBytes);
        copyIn(scene_.passMask[slot], staging, maskBytes);
        if (uniformBase != NoSlot) memset32(scene_.passBase[slot]->pointer, uniformBase, pixels);
        else gpu::Driver::check(driver.cuMemcpyDtoDAsync_v2_p(scene_.passBase[slot]->pointer, scene_.counts->pointer,
                                                               pixels * sizeof(U32), stream_));
    }
    // Launches queued so far, render or otherwise.
    U64 launchCount() const {  // the followers' too (multi_device.h)
        U64 n = launches_;
        for (const auto& f : followers_) n += f->launches_;
        return n;
    }
    // Samples parked on this card and its followers.
    U64 parkedAll() const {
        U64 n = parkedOutstanding();
        for (const auto& f : followers_) n += f->parkedOutstanding();
        return n;
    }

    // The adaptive pass chain (async_adaptive.cu): a frame's passes queued at
    // once, each made on the device from the trees of the passes before it.
    struct ChainStep {
        U32 pass, size;  // the pass's index in the schedule and its samples a pixel
        bool uniform;    // every pixel, no trees: the passes below the minimum sample count
    };
    bool hasChain() const { return chain_.module != nullptr && followers_.empty(); }  // one card's alone
    // Sets up a frame's chain: the region trees' bounds and room, the tiles in the
    // work grid's cell order, and empty planned counts. After begin().
    void beginChain(const adaptive::Regions& regions, const std::vector<std::uint64_t>& inImage,
                    const std::vector<U32>& tileOrder) {
        cpu::need(hasChain() && (single_ || programs_[1].pipeline), "No GPU pass chain");
        const size_t pixels = size_t(plan_->width) * plan_->height, tiles = inImage.size(), n = regions.count();
        std::vector<adaptive::IBox> bounds(n);
        std::vector<U32> nodeOffset(n), nodeCapacity(n);
        std::vector<U64L> satOffset(n);
        U64 nodes = 0, sat = 0;
        for (size_t r = 0; r < n; ++r) {
            const auto b = regions.tree(r).root();
            bounds[r] = b;
            nodeOffset[r] = U32(nodes);
            nodeCapacity[r] = U32(std::max(1, adaptive::maxNodes(b)));
            satOffset[r] = sat;
            nodes += nodeCapacity[r];
            sat += U64(b.x1 - b.x0 + 1) * U64(b.y1 - b.y0 + 1);
        }
        auto fit = [&](B& b, size_t bytes) {
            if (!b || b->bytes < bytes) b = device(std::max<size_t>(bytes, 16));
        };
        fit(chain_.regions, n * sizeof(adaptive::IBox));
        fit(chain_.nodes, nodes * sizeof(adaptive::Node));
        fit(chain_.nodeOffset, n * sizeof(U32));
        fit(chain_.nodeCapacity, n * sizeof(U32));
        fit(chain_.sat, sat * sizeof(double));
        fit(chain_.satOffset, n * sizeof(U64L));
        fit(chain_.regionDone, n * sizeof(U32));
        fit(chain_.tileMask, tiles * sizeof(U64L));
        fit(chain_.inImage, tiles * sizeof(U64L));
        fit(chain_.tileOrder, tiles * sizeof(U32));
        fit(chain_.activeTiles, tiles * sizeof(U32));
        fit(chain_.activeMasks, tiles * sizeof(U64L));
        fit(chain_.planned, pixels * sizeof(U32));
        fit(chain_.base, pixels * sizeof(U32));
        fit(chain_.params, MaxPasses * sizeof(LaunchParams));
        if (!chain_.hostPlanned || chain_.hostPlanned->bytes < pixels * sizeof(U32)) chain_.hostPlanned = pinned(pixels * sizeof(U32));
        if (!chain_.params_staging) chain_.params_staging = pinned(MaxPasses * sizeof(LaunchParams));
        // The frame's small tables, staged together; the last frame's chain has
        // completed, so its staging is free.
        const size_t layout = n * (sizeof(adaptive::IBox) + 2 * sizeof(U32) + sizeof(U64L)) + tiles * (sizeof(U64L) + sizeof(U32));
        if (!chain_.staging || chain_.staging->bytes < layout) chain_.staging = pinned(layout);
        char* at = static_cast<char*>(chain_.staging->host());
        auto stage = [&](const B& to, const void* from, size_t bytes) {
            std::memcpy(at, from, bytes);
            copyIn(to, at, bytes);
            at += bytes;
        };
        stage(chain_.regions, bounds.data(), n * sizeof(adaptive::IBox));
        stage(chain_.nodeOffset, nodeOffset.data(), n * sizeof(U32));
        stage(chain_.nodeCapacity, nodeCapacity.data(), n * sizeof(U32));
        stage(chain_.satOffset, satOffset.data(), n * sizeof(U64L));
        stage(chain_.inImage, inImage.data(), tiles * sizeof(U64L));
        stage(chain_.tileOrder, tileOrder.data(), tiles * sizeof(U32));
        memset32(chain_.planned->pointer, 0, pixels);
        memset32(chain_.regionDone->pointer, 0, n);
        memset32(state_->pointer + offsetof(DeviceState, chainPass), NoPass, 1);

        ChainArgs& a = chain_.args;
        a = {};
        a.state = reinterpret_cast<DeviceState*>(state_->pointer);
        a.sums = reinterpret_cast<const double*>(scene_.sums->pointer);
        a.odd = reinterpret_cast<const double*>(scene_.odd->pointer);
        a.counts = reinterpret_cast<const U32*>(scene_.counts->pointer);
        a.errors = reinterpret_cast<float*>(scene_.errors->pointer);
        a.regions = reinterpret_cast<const adaptive::IBox*>(chain_.regions->pointer);
        a.nodes = reinterpret_cast<adaptive::Node*>(chain_.nodes->pointer);
        a.nodeOffset = reinterpret_cast<const U32*>(chain_.nodeOffset->pointer);
        a.nodeCapacity = reinterpret_cast<const U32*>(chain_.nodeCapacity->pointer);
        a.sat = reinterpret_cast<double*>(chain_.sat->pointer);
        a.satOffset = reinterpret_cast<const U64L*>(chain_.satOffset->pointer);
        a.regionDone = reinterpret_cast<U32*>(chain_.regionDone->pointer);
        a.tileMask = reinterpret_cast<U64L*>(chain_.tileMask->pointer);
        a.inImage = reinterpret_cast<const U64L*>(chain_.inImage->pointer);
        a.tileOrder = reinterpret_cast<const U32*>(chain_.tileOrder->pointer);
        a.activeTiles = reinterpret_cast<U32*>(chain_.activeTiles->pointer);
        a.activeMasks = reinterpret_cast<U64L*>(chain_.activeMasks->pointer);
        a.planned = reinterpret_cast<U32*>(chain_.planned->pointer);
        a.base = reinterpret_cast<U32*>(chain_.base->pointer);
        a.width = plan_->width;
        a.height = plan_->height;
        a.tiles = U32(tiles);
        a.tilesX = (plan_->width + TileSize - 1) / TileSize;
        a.regionCount = U32(n);
        a.target = regions.target();
        a.layout = regions.layout();
        chain_.maxStep = 0;
        for (double& s : chainStages_) s = 0;
        chainRenders_.clear();
        chainsLaunched_ = 0;
        chainsCollected_ = 0;
        chainState_ = {};
    }
    // Queues steps of the chain, then one readback: the whole lot is one flight.
    // A step's render that parks a sample stalls the chain; the host lands the
    // tiles, reruns the parked samples by ordinary launches, then resumes the chain
    // after that step.
    bool launchChain(const std::vector<ChainStep>& steps) {
        cpu::need(ready() && !steps.empty() && steps.size() <= MaxPasses, "GPU chain backpressure");
        commitTiles();  // tiles asked for so far land first
        U64 most = 0;  // the most samples one step could park; a stall stops the chain after it
        for (const auto& s : steps) most = std::max<U64>(most, U64(s.size) * TilePixels * chain_.args.tiles);
        cpu::need(!mayPark_ || parkedBound() - parkedIssued_ + most <= parkedCapacity_, "Parked-sample queue would overflow");
        memset32(state_->pointer + offsetof(DeviceState, chainStall), 0, 1);
        auto* staged = static_cast<LaunchParams*>(chain_.params_staging->host());
        for (size_t k = 0; k < steps.size(); ++k) {
            LaunchParams p = baseParams();
            p.mode = 3;
            p.partial = 0;
            p.chainPass = steps[k].pass;
            p.chainSize = steps[k].size;
            p.activeTiles = reinterpret_cast<const U32*>(chain_.activeTiles->pointer);
            p.activeMasks = reinterpret_cast<const U64L*>(chain_.activeMasks->pointer);
            p.chainBase = reinterpret_cast<const U32*>(chain_.base->pointer);
            staged[k] = p;
        }
        copyIn(chain_.params, staged, steps.size() * sizeof(LaunchParams));
        auto& r = readbacks_[launches_ % readbacks_.size()];
        if (!r.begun) {
            r.begun = std::make_shared<TimingEvent>(driver);
            r.ended = std::make_shared<TimingEvent>(driver);
        }
        const U32 pixels = plan_->width * plan_->height;
        gpu::Driver::check(driver.cuEventRecord_p(r.begun->value, stream_));
        // With profiling, an event after each stage: errors, trees, masks with
        // compaction, render.
        auto mark = [&] {
            if (!options_.profileChain) return;
            chain_.marks.push_back(std::make_shared<TimingEvent>(driver));
            gpu::Driver::check(driver.cuEventRecord_p(chain_.marks.back()->value, stream_));
        };
        chain_.marks.clear();
        mark();
        for (size_t k = 0; k < steps.size(); ++k) {
            const auto& s = steps[k];
            if (!s.uniform) kernel(chain_.errors, (pixels + 255) / 256, 256);
            mark();
            if (!s.uniform) kernel(chain_.trees, chain_.args.regionCount, 512);
            mark();
            U32 uniform = s.uniform ? 1u : 0u, size = s.size, pass = s.pass;
            kernel(chain_.masks, (chain_.args.tiles + 255) / 256, 256, uniform, size);
            kernel(chain_.compact, 1, 1024, pass, size);
            mark();
            ox(optixLaunch(program_->pipeline, stream_, chain_.params->pointer + k * sizeof(LaunchParams),
                           sizeof(LaunchParams), &program_->sbt, options_.launchThreads, 1, 1));
            mark();
        }
        gpu::Driver::check(driver.cuEventRecord_p(r.ended->value, stream_));
        readBack(r);
        inFlight_.push_back({launches_, published_, 0, 0, most, gpu::StartClock::now(), &r, true});
        ++launches_;
        ++chainsLaunched_;
        return true;
    }
    // With GpuOptions::profileChain: the frame's seconds in the chain's errors,
    // trees, masks (with compaction) and renders.
    const double* chainStages() const { return chainStages_; }
    const std::vector<double>& chainRenders() const { return chainRenders_; }  // each step's render
    // Chains collected so far this frame, and the device's state after the last.
    U64 chainsCollected() const { return chainsCollected_; }
    const DeviceState& chainState() const { return chainState_; }
    // Each pixel's planned samples after the frame's chain, once the film has landed.
    const U32* chainPlanned() const { return static_cast<const U32*>(chain_.hostPlanned->host()); }
    // Samples each pass has completed on the GPU, as of the last collected launch.
    const std::vector<U32>& passDone() const {
        if (followers_.empty()) return passDone_;
        passSum_ = passDone_;
        for (const auto& f : followers_)
            for (size_t i = 0; i < passSum_.size() && i < f->passDone_.size(); ++i) passSum_[i] += f->passDone_[i];
        return passSum_;
    }

    size_t fineTiles() const { return fine_.tiles; }
    // The card's free and total memory (cuMemGetInfo); false where the driver does not say.
    bool memoryInfo(size_t& freeBytes, size_t& totalBytes) {
        using Info = CUresult(CUDAAPI*)(size_t*, size_t*);
        // XPU2_SESSION_CARD_FREE_MB: a card with this much free, of this much, for tests of a full card.
        if (const char* v = std::getenv("XPU2_SESSION_CARD_FREE_MB")) {
            freeBytes = totalBytes = size_t(std::max(1.0, std::atof(v))) << 20;
            return true;
        }
        const auto info = reinterpret_cast<Info>(GetProcAddress(driver.library, "cuMemGetInfo_v2"));
        return info && info(&freeBytes, &totalBytes) == CUDA_SUCCESS && totalBytes > 0;
    }
    // What fineBudget keeps back on the card.
    static size_t cardReserve(size_t totalBytes) { return std::max<size_t>(totalBytes / 5, 3ull << 29); }
    // Whether the card has come to hold much less than the fine-detail cache was sized for (something else took it, Modo's
    // viewport or another application): prepare again makes the cache to fit, meshes and all else kept. Read as a frame starts.
    bool fineCacheTooBig() {
        bool any = false;
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { any = f.fineCacheTooBig() || any; });
        return fineCacheTooBigOwn() || any;
    }
    bool fineCacheTooBigOwn() {
        size_t freeBytes = 0, totalBytes = 0;
        if (std::getenv("XPU2_SESSION_FINE_BUDGET_MB") || options_.fineBudget || !fine_.slots || !memoryInfo(freeBytes, totalBytes)) return false;
        const size_t held = fine_.slabBytes, reserve = cardReserve(totalBytes);
        const size_t room = freeBytes + held > reserve ? freeBytes + held - reserve : 0;
        // Taken since this scene was prepared: the scene's own films, made after the cache was sized, are not that.
        return freeAfterPrepare_ > freeBytes && freeAfterPrepare_ - freeBytes > held / 10 && std::max<size_t>(256ull << 20, room) < held / 10 * 9;
    }
    size_t freeAfterPrepare_ = 0;
    // All that is held of the scene let go, the lane left ready for prepare.
    void dropScene() {
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { f.dropScene(); });
        driver.drain();
        plan_.reset();
        scene_ = {};
        fine_ = {};
        meshCache_.clear();
        placedKeep_.clear();
        tops_.clear();
        freeAfterPrepare_ = 0;
    }
    // Whether the card is short of even what is kept back.
    bool cardTight() {
        bool any = false;
        if (!followers_.empty()) forwardOrDrop([&](GpuLane& f) { any = f.cardTight() || any; });
        size_t freeBytes = 0, totalBytes = 0;
        return any || (memoryInfo(freeBytes, totalBytes) && freeBytes < cardReserve(totalBytes) / 2);
    }
    // The fine-detail cache's VRAM: what the scene's level-0 tiles need, up to what the card has free less a reserve
    // for the rest of the frame and for Modo's own viewport (a fifth of the card, a gigabyte and a half at least), and
    // never below 256 MB. Measured at each scene's preparation, so it follows what else holds the card meanwhile.
    // XPU2_SESSION_FINE_BUDGET_MB, or GpuOptions::fineBudget, fixes it.
    size_t fineBudget(size_t need) {
        if (const char* v = std::getenv("XPU2_SESSION_FINE_BUDGET_MB")) {
            fineBudgetWhy_ = "set by XPU2_SESSION_FINE_BUDGET_MB";
            return size_t(std::max(1.0, std::atof(v))) << 20;
        }
        if (options_.fineBudget) {
            fineBudgetWhy_ = "fixed";
            return options_.fineBudget;
        }
        size_t freeBytes = 0, totalBytes = 0;
        if (!memoryInfo(freeBytes, totalBytes)) {
            fineBudgetWhy_ = "the card's memory unknown";
            return 512ull << 20;
        }
        const size_t reserve = cardReserve(totalBytes);
        const size_t spare = freeBytes > reserve ? freeBytes - reserve : 0;
        char why[160];
        std::snprintf(why, sizeof why, "%.1f GB of the card's %.1f GB free, %.1f GB kept back", double(freeBytes) / (1 << 30),
                      double(totalBytes) / (1 << 30), double(reserve) / (1 << 30));
        fineBudgetWhy_ = why;
        return std::max<size_t>(256ull << 20, std::min(need, spare));
    }
    size_t fineBudget_ = 0;
    std::string fineBudgetWhy_;
    size_t fineLoaded() const { return fine_.loaded; }
    size_t fineEvicted() const { return fine_.evicted; }
    // Where the textures stand on the card, for the session's log: the coarser levels, the fine-detail cache, its slots by
    // their size (each tile in a slot of its format's size), and the flat tiles held as one value outside them.
    std::string textureCardText() const {
        std::string sizes;
        size_t held = 0;
        for (const auto& c : fine_.classes) {
            const size_t in = c.made - std::min<size_t>(c.made, c.freeSlots.size());
            held += in;
            char one[100];
            std::snprintf(one, sizeof one, "%s%zu in %u slots of %zu KB made, of %u", sizes.empty() ? "" : ", ", in, c.made, c.slotBytes >> 10, c.slots);
            sizes += one;
        }
        char text[420];
        std::snprintf(text, sizeof text, "coarser levels %.0f MB, fine-detail cache %.0f MB (%zu tiles in it: %s; %zu sent, %zu let go; %zu flat tiles as one value each, %zu tiles past half floats' range in floats)",
                      double(scene_.mipBytes) / (1 << 20), double(fine_.slabBytes) / (1 << 20), held, sizes.c_str(), fine_.loaded, fine_.evicted, fine_.flat, fine_.wide.size());
        return text;
    }
    size_t residentMipBytes() const { return scene_.mipBytes; }

private:
    using B = gpu::B;
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    static OptixPipelineCompileOptions pipelineOptions() {
        OptixPipelineCompileOptions po{};
        po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        po.usesMotionBlur = 1;  // moving meshes and curves: structures with a key a time step, traced at the sample's time
        po.numPayloadValues = 2;
        po.numAttributeValues = 2;
        po.pipelineLaunchParamsVariableName = "params";
        po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        // A diagnostic: OptiX's own checks of the stacks and the trace depth, reported by the exception program.
        if (exceptions()) po.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        return po;
    }
    static bool exceptions() { return std::getenv("XPU2_OPTIX_EXCEPTIONS") != nullptr; }
    // Whether the machine has room for another compile task: free RAM above a reserve, an eighth of the machine's and
    // at least 4 GB (XPU2_COMPILE_RESERVE_GB: another). A task of the largest modules takes GBs of its own; several at
    // once took a 64 GB machine past 25 GB.
    static bool compileMemoryRoom() {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof status;
        if (!GlobalMemoryStatusEx(&status)) return true;
        double reserve = std::max(4.0, double(status.ullTotalPhys) / double(1ull << 30) / 8.0);
        if (const char* r = std::getenv("XPU2_COMPILE_RESERVE_GB")) reserve = std::max(1.0, std::atof(r));
        return double(status.ullAvailPhys) / double(1ull << 30) > reserve;
    }
    // The program modules (the adaptive one if any), compiled together on a pool of threads through OptiX's
    // compile tasks: one thread compiling the shading library took minutes. OptiX's disk cache still serves
    // a module it has compiled before: it hands out tasks all the same, which then finish at once.
    // The program's modules, compiled together: their tasks run on one pool of threads, in parallel.
    void compileModules(const std::string* const (&sources)[ModuleCount], OptixModule (&modules)[ModuleCount]) {
        OptixModuleCompileOptions mo{};
        mo.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        // XPU2_OPTIX_LEVEL (0 to 3): a test build's lower optimization, compiled in a fraction of the time.
        if (const char* level = std::getenv("XPU2_OPTIX_LEVEL")) {
            static const OptixCompileOptimizationLevel levels[4] = {OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
                                                                   OPTIX_COMPILE_OPTIMIZATION_LEVEL_2, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3};
            const int l = std::atoi(level);
            if (l >= 0 && l <= 3) mo.optLevel = levels[l];
        }
        mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        // Each module's own options: XPU2_OPTIX_LEVELS ("module:level,...", e.g. "0:1,5:2") sets some modules' levels, to
        // measure what a lighter level saves in compile memory and costs in speed.
        OptixModuleCompileOptions moduleOptions[ModuleCount];
        for (int i = 0; i < ModuleCount; ++i) moduleOptions[i] = mo;
        if (const char* levels = std::getenv("XPU2_OPTIX_LEVELS")) {
            static const OptixCompileOptimizationLevel named[4] = {OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
                                                                  OPTIX_COMPILE_OPTIMIZATION_LEVEL_2, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3};
            for (const char* at = levels; *at;) {
                int module = -1, level = -1, used = 0;
                if (std::sscanf(at, "%d:%d%n", &module, &level, &used) != 2) break;
                if (module >= 0 && module < ModuleCount && level >= 0 && level <= 3) moduleOptions[module].optLevel = named[level];
                at += used;
                if (*at == ',') ++at;
            }
        }
        const OptixPipelineCompileOptions po = pipelineOptions();
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<OptixTask> tasks;
        // The tasks write their feedback here until the modules complete or fail.
        static thread_local char logs[ModuleCount][8192];
        size_t lengths[ModuleCount];
        for (int i = 0; i < ModuleCount; ++i) lengths[i] = sizeof(logs[i]);
        for (int i = 0; i < ModuleCount; ++i) {
            if (sources[i]->empty()) continue;
            OptixTask first = nullptr;
            if (optixModuleCreateFromPTXWithTasks(optix_, &moduleOptions[i], &po, sources[i]->data(), sources[i]->size(), logs[i], &lengths[i],
                                                  &modules[i], &first) != OPTIX_SUCCESS)
                throw std::runtime_error("OptiX module: " + std::string(logs[i], std::min(lengths[i], sizeof(logs[i]))));
            if (first) tasks.push_back(first);
        }
        unsigned finished = 0;
        unsigned active = 0;
        bool done = tasks.empty(), heldNoted = false;
        OptixResult error = OPTIX_SUCCESS;
        // When each module completed (seconds from the start; negative: not yet).
        const auto began = std::chrono::steady_clock::now();
        double completedAt[ModuleCount];
        for (double& at : completedAt) at = -1;
        auto noteCompleted = [&] {
            for (int i = 0; i < ModuleCount; ++i) {
                if (!modules[i] || completedAt[i] >= 0) continue;
                OptixModuleCompileState state = OPTIX_MODULE_COMPILE_STATE_NOT_STARTED;
                if (optixModuleGetCompilationState(modules[i], &state) == OPTIX_SUCCESS && state == OPTIX_MODULE_COMPILE_STATE_COMPLETED) {
                    completedAt[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                    std::fprintf(stderr, "OptiX module %d compiled at %.1f s\n", i, completedAt[i]);
                    std::fflush(stderr);
                }
            }
        };
        auto work = [&] {
            for (;;) {
                OptixTask task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    // A task more only while the machine has room for it (compileMemoryRoom): one task at least runs,
                    // so the compile always moves; the others wait for memory as for work (each task can take GBs).
                    for (;;) {
                        wake.wait_for(lock, std::chrono::milliseconds(500), [&] { return done || !tasks.empty(); });
                        if (done) return;
                        if (tasks.empty()) continue;
                        if (active == 0 || compileMemoryRoom()) break;
                        if (!heldNoted) {
                            heldNoted = true;
                            std::printf("MoonRay XPU2 shaders: memory is short: compile tasks run fewer at a time\n");
                            std::fflush(stdout);
                        }
                    }
                    task = tasks.front();
                    tasks.pop_front();
                    ++active;
                }
                OptixTask additional[16]{};
                unsigned count = 0;
                const OptixResult status = optixTaskExecute(task, additional, 16, &count);
                std::lock_guard<std::mutex> lock(mutex);
                --active;
                if (status != OPTIX_SUCCESS && error == OPTIX_SUCCESS) error = status;
                if (error == OPTIX_SUCCESS)
                    for (unsigned i = 0; i < count; ++i) tasks.push_back(additional[i]);
                else
                    tasks.clear();
                ++finished;
                noteCompleted();
                if (options_.compileProgress) options_.compileProgress(finished, unsigned(tasks.size()) + active);
                if (tasks.empty() && active == 0) done = true;
                wake.notify_all();
            }
        };
        unsigned threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        // XPU2_COMPILE_THREADS: that many tasks at once (1: one at a time, each task's memory its own, to measure).
        if (const char* n = std::getenv("XPU2_COMPILE_THREADS")) threads = std::max(1, std::min(64, std::atoi(n)));
        std::vector<std::thread> workers;
        try {
            for (unsigned i = 1; i < threads; ++i) workers.emplace_back(work);
            work();
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                done = true;
                wake.notify_all();
            }
            for (auto& w : workers) w.join();
            throw;
        }
        for (auto& w : workers) w.join();
        noteCompleted();
        for (int i = 0; i < ModuleCount; ++i)
            if (modules[i] && completedAt[i] >= 0) std::fprintf(stderr, "OptiX module %d: %.1f s\n", i, completedAt[i]);
        // A module the tasks did not complete is compiled again on one thread, for OptiX's log.
        for (int i = 0; i < ModuleCount; ++i) {
            if (!modules[i]) continue;
            OptixModuleCompileState state = OPTIX_MODULE_COMPILE_STATE_FAILED;
            if (optixModuleGetCompilationState(modules[i], &state) == OPTIX_SUCCESS && state == OPTIX_MODULE_COMPILE_STATE_COMPLETED &&
                error == OPTIX_SUCCESS)
                continue;
            std::fprintf(stderr, "OptiX compile tasks: module %d state %x, result %d; compiling on one thread\n%.*s\n", i,
                         unsigned(state), int(error), int(std::min(lengths[i], sizeof(logs[i]))), logs[i]);
            optixModuleDestroy(modules[i]);
            modules[i] = nullptr;
            char log[8192];
            size_t length = sizeof(log);
            if (optixModuleCreateFromPTX(optix_, &moduleOptions[i], &po, sources[i]->data(), sources[i]->size(), log, &length, &modules[i]) !=
                OPTIX_SUCCESS)
                throw std::runtime_error("OptiX module: " + std::string(log, std::min(length, sizeof(log))));
        }
    }
    // One OptiX program: its modules, programs, pipeline and binding table.
    struct Program {
        // The groups: raygen, two misses, four hit groups (triangles, curves; the volume bounds' collecting two), then
        // the callables (async_gpu_device.h's MaterialCall order) and the exception program where it is on.
        static constexpr unsigned CallablesFirst = 7;
        OptixModule modules[ModuleCount]{};
        OptixProgramGroup groups[32]{};  // five, and the callables (async_gpu_device.h's MaterialCall)
        OptixPipeline pipeline = nullptr;
        B records;
        // The stack sizes (create) and the continuation stack's parts: the raygen's and a trace's, the volume module's
        // volume callables', its subsurface callable's; stackC as set (fitStack).
        unsigned stackA = 0, stackB = 0, stackC = 0, stackBase = 0, ccVolume = 0, ccSubsurface = 0;
        bool fitted = false;
        bool ownsModules = true;  // the mapped program's are the first program's
        OptixShaderBindingTable sbt{};

        // Takes the compiled modules (compileModules) and makes their programs and pipeline.
        void create(GpuLane& lane, OptixModule (&compiled)[ModuleCount]) {
            for (int i = 0; i < ModuleCount; ++i) modules[i] = compiled[i];
            const OptixModule module = modules[0];
            const OptixPipelineCompileOptions po = pipelineOptions();
            char log[8192];
            size_t length = sizeof(log);
            OptixProgramGroupDesc desc[32] = {};
            desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            desc[0].raygen = {module, "__raygen__async"};
            desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc[1].miss = {module, "__miss__radiance"};
            desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc[2].miss = {module, "__miss__shadow"};
            desc[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[3].hitgroup.moduleCH = module;
            desc[3].hitgroup.entryFunctionNameCH = "__closesthit__triangle";
            // Curves: their spans' own intersection program, the same closest hit.
            desc[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[4].hitgroup.moduleCH = module;
            desc[4].hitgroup.entryFunctionNameCH = "__closesthit__triangle";
            desc[4].hitgroup.moduleIS = module;
            desc[4].hitgroup.entryFunctionNameIS = "__intersection__curve";
            // The volume bounds' traversal (GpuBackend::volumeCrossings): every candidate kept by its any-hit program
            // (hit groups 2 and 3 of the table; the curves' never meet the bounds, but each type has its record).
            desc[5].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[5].hitgroup.moduleAH = module;
            desc[5].hitgroup.entryFunctionNameAH = "__anyhit__volumeCrossing";
            desc[6].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[6].hitgroup.moduleAH = module;
            desc[6].hitgroup.entryFunctionNameAH = "__anyhit__volumeCrossing";
            desc[6].hitgroup.moduleIS = module;
            desc[6].hitgroup.entryFunctionNameIS = "__intersection__curve";
            // The callables, in async_gpu_device.h's MaterialCall order: all of them in a program built in modules.
            static const struct { int module; const char* entry; bool continuation; } callables[] = {
                {1, "__direct_callable__shade", false},       {1, "__direct_callable__presence", false},
                {2, "__direct_callable__maps", false},        {3, "__direct_callable__lightSample", false},
                {3, "__direct_callable__lightIntersect", false}, {3, "__direct_callable__lightEval", false},
                {4, "__direct_callable__lobeEval", false},    {4, "__direct_callable__lobeSample", false},
                {4, "__direct_callable__lobeAlbedo", false},  {5, "__continuation_callable__volumeRadiance", true},
                {5, "__continuation_callable__subsurfaceRadiance", true}, {5, "__continuation_callable__volumeTransmittance", true},
                {6, "__direct_callable__dwaResolve", false}, {7, "__direct_callable__dwaLobes", false},
                {8, "__direct_callable__camera", false},     {8, "__direct_callable__aovLight", false},
                {8, "__direct_callable__aovVisibility", false}, {8, "__direct_callable__materialAovs", false},
                {5, "__continuation_callable__emissiveVolumeIntegral", true}};
            const bool split = modules[1] != nullptr;
            for (int i = 2; i < ModuleCount; ++i)
                if (i != VolumeMapsModule) cpu::need(!split || modules[i], "The GPU program's modules are incomplete: module " + std::to_string(i) + " is missing");
            const unsigned callableCount = split ? unsigned(sizeof(callables) / sizeof(callables[0])) : 0u;
            const unsigned count = CallablesFirst + callableCount + (exceptions() ? 1u : 0u);
            static_assert(CallablesFirst + sizeof(callables) / sizeof(callables[0]) <= sizeof(desc) / sizeof(desc[0]), "too many program groups");
            for (unsigned k = 0; k < callableCount; ++k) {
                desc[CallablesFirst + k].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                if (callables[k].continuation) {
                    desc[CallablesFirst + k].callables.moduleCC = modules[callables[k].module];
                    desc[CallablesFirst + k].callables.entryFunctionNameCC = callables[k].entry;
                } else {
                    desc[CallablesFirst + k].callables.moduleDC = modules[callables[k].module];
                    desc[CallablesFirst + k].callables.entryFunctionNameDC = callables[k].entry;
                }
            }
            if (exceptions()) {  // the diagnostic's exception program, after the callables
                desc[count - 1].kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
                desc[count - 1].exception = {module, "__exception__report"};
            }
            OptixProgramGroupOptions options{};
            length = sizeof(log);
            ox(optixProgramGroupCreate(lane.optix_, desc, count, &options, log, &length, groups));
            OptixPipelineLinkOptions link{};
            link.maxTraceDepth = 1;
            link.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            length = sizeof(log);
            ox(optixPipelineCreate(lane.optix_, &po, &link, groups, count, log, &length, &pipeline));
            OptixStackSizes stacks{};
            for (unsigned i = 0; i < count; ++i) ox(optixUtilAccumulateStackSizes(groups[i], &stacks));
            if (std::getenv("XPU2_ASYNC_STACKS"))  // a diagnostic: the raygen's stack shows state spilled across traces
                std::fprintf(stderr, "stacks: cssRG %u cssMS %u cssCH %u dssDC %u\n", stacks.cssRG, stacks.cssMS, stacks.cssCH, stacks.dssDC);
            unsigned a, b, c;
            ox(optixUtilComputeStackSizes(&stacks, 1, 0, 0, &a, &b, &c));
            if (split) {
                // The driver reserves a stack for every hardware thread, some 126 KB of the card a byte, and the utility's
                // worst case asks for the largest callable twice and a continuation callable below a closest hit. By the
                // real calls: the raygen calls the callables and the volume module's continuation callables, one level,
                // which call the direct ones; of the direct ones only shade and presence call others; closest hit, miss and
                // intersection call none. OptiX reports 24 bytes for the map callable, whose PTX frames add to some 1.1 KB
                // (mapsAtLeast keeps more than twice that), so what it reports is not taken alone.
                const auto own = [&](unsigned k) {
                    OptixStackSizes one{};
                    ox(optixUtilAccumulateStackSizes(groups[CallablesFirst + k], &one));
                    return one.dssDC;
                };
                constexpr unsigned mapsAtLeast = 4096, traceAtLeast = 512;
                const unsigned maps = std::max(own(2), mapsAtLeast);
                const auto ownCC = [&](unsigned k) {
                    OptixStackSizes one{};
                    ox(optixUtilAccumulateStackSizes(groups[CallablesFirst + k], &one));
                    return one.cssCC;
                };
                a = 0;
                b = std::max({own(0) + std::max({maps, own(12), own(13)}), own(1) + maps, stacks.dssDC});
                c = stacks.cssRG + stacks.cssCC + std::max({stacks.cssCH, stacks.cssMS, stacks.cssIS + stacks.cssAH, traceAtLeast});
                stackBase = c - stacks.cssCC;
                ccVolume = std::max({ownCC(9), ownCC(11), ownCC(18)});
                ccSubsurface = ownCC(10);
                fitted = stacks.cssCC == std::max(ccVolume, ccSubsurface);
            }
            stackA = a;
            stackB = b;
            stackC = c;
            if (std::getenv("XPU2_ASYNC_STACKS")) {
                for (unsigned i = 0; i < count; ++i) {
                    OptixStackSizes one{};
                    ox(optixUtilAccumulateStackSizes(groups[i], &one));
                    const char* name = i >= CallablesFirst && i - CallablesFirst < sizeof(callables) / sizeof(callables[0]) ? callables[i - CallablesFirst].entry : "";
                    std::fprintf(stderr, "stack of %u %s: rg %u ms %u ch %u dc %u cc %u is %u ah %u\n", i, name, one.cssRG, one.cssMS, one.cssCH, one.dssDC, one.cssCC, one.cssIS, one.cssAH);
                }
                std::fprintf(stderr, "stacks asked: from traversal %u, from state %u, continuation %u\n", a, b, c);
            }
            ox(optixPipelineSetStackSize(pipeline, a, b, c, 2));
            auto host = lane.pinned(sizeof(Record) * count);
            for (unsigned i = 0; i < count; ++i) ox(optixSbtRecordPackHeader(groups[i], static_cast<Record*>(host->host()) + i));
            records = lane.device(sizeof(Record) * count);
            gpu::Driver::check(lane.driver.cuMemcpyHtoDAsync_v2_p(records->pointer, host->host(), host->bytes, nullptr));
            lane.driver.drain();
            sbt.raygenRecord = records->pointer;
            sbt.missRecordBase = records->pointer + sizeof(Record);
            sbt.missRecordStrideInBytes = sizeof(Record);
            sbt.missRecordCount = 2;
            sbt.hitgroupRecordBase = records->pointer + 3 * sizeof(Record);
            sbt.hitgroupRecordStrideInBytes = sizeof(Record);
            sbt.hitgroupRecordCount = 4;
            if (split) {
                sbt.callablesRecordBase = records->pointer + CallablesFirst * sizeof(Record);
                sbt.callablesRecordStrideInBytes = sizeof(Record);
                sbt.callablesRecordCount = callableCount;
            }
            if (exceptions()) sbt.exceptionRecord = records->pointer + (count - 1) * sizeof(Record);
        }
        // The continuation stack a scene needs: the volume module's callables only where the scene can call them, the
        // volume ones where it has volumes, the subsurface one where a material scatters under its surface (cpu::Plan's
        // subsurface: a DwaBase-family material whose subsurface model is not none; any DwaBase material kept it before, some
        // 3.5 GB of the card on a 24 GB one). The driver keeps a stack for every hardware thread, so the frame left out is saved on each.
        // Between frames only (begin): no launch in flight.
        void fitStack(bool volumes, bool subsurface) {
            if (!fitted) return;
            const unsigned c = stackBase + std::max(volumes ? ccVolume : 0u, subsurface ? ccSubsurface : 0u);
            if (c == stackC) return;
            ox(optixPipelineSetStackSize(pipeline, stackA, stackB, c, 2));
            if (std::getenv("XPU2_ASYNC_STACKS")) std::fprintf(stderr, "continuation stack %u for the scene (volumes %d, subsurface %d)\n", c, int(volumes), int(subsurface));
            stackC = c;
        }
        void destroy() noexcept {
            if (pipeline) optixPipelineDestroy(pipeline);
            for (auto& g : groups)
                if (g) optixProgramGroupDestroy(g);
            if (ownsModules)
                for (auto& m : modules)
                    if (m) optixModuleDestroy(m);
            pipeline = nullptr;
            for (auto& m : modules) m = nullptr;
            for (auto& g : groups) g = nullptr;
            records.reset();
        }
    };
    struct ParamSlot { B host, device; std::shared_ptr<gpu::Event> done; };
    struct Readback { B state, requests, touched; std::shared_ptr<gpu::Event> done; std::shared_ptr<TimingEvent> begun, ended; };
    struct Flight {
        U64 launch, firstChunk;
        U32 chunks;
        U64 parked, mayPark;
        gpu::StartClock::time_point started;
        Readback* readback;
        bool chain = false;
        U64 refines = 0, mayRefine = 0;  // provisional samples it renders again; the most it may make
        U64 samples = 0;                 // its chunks' samples
    };
    // The pass chain's kernels and buffers (beginChain).
    struct Chain {
        CUmodule module = nullptr;
        CUfunction errors = nullptr, trees = nullptr, masks = nullptr, compact = nullptr;
        B regions, nodes, nodeOffset, nodeCapacity, sat, satOffset, regionDone, tileMask, inImage, tileOrder,
            activeTiles, activeMasks, planned, base, params, hostPlanned, params_staging, staging;
        ChainArgs args{};
        U32 maxStep = 0;
        std::vector<std::shared_ptr<TimingEvent>> marks;
    };
    // Queues one of the chain's CUDA kernels with the chain's arguments first.
    template <class... A>
    void kernel(CUfunction f, U32 blocks, U32 threads, A&... extra) {
        void* args[] = {&chain_.args, &extra...};
        gpu::Driver::check(driver.cuLaunchKernel_p(f, blocks, 1, 1, threads, 1, 1, 0, stream_, args, nullptr));
    }
    // Reads back what the host needs after a launch, then clears the touch marks.
    void readBack(Readback& r) {
        copyOut(r.state, state_, sizeof(DeviceState));
        if (fine_.tiles) {
            ensure(r.requests, fine_.requestBytes());
            ensure(r.touched, fine_.tiles + 1);
            copyOut(r.requests, fine_.requests, fine_.requestBytes());
            copyOut(r.touched, fine_.touched, fine_.tiles + 1);
            gpu::Driver::check(driver.cuMemsetD8Async_p(fine_.touched->pointer, 0, fine_.tiles + 1, stream_));
        }
        if (!r.done) r.done = std::make_shared<gpu::Event>(driver);
        gpu::Driver::check(driver.cuEventRecord_p(r.done->value, stream_));
    }
    // A triangle mesh as the GPU holds it (prepare): its buffers and acceleration structure, its geometry and attribute
    // records; kept from one plan to the next while the plan's mesh is the same MeshData.
    struct MeshEntry {
        std::shared_ptr<const cpu::MeshData> data;  // held, so its address names no other mesh while cached
        std::vector<B> keep;
        OptixTraversableHandle handle = 0;
        mixed::Geometry geometry{};
        shade::MeshAttributes attributes{};
        MeshMotion motion{};
    };
    std::map<const cpu::MeshData*, MeshEntry> meshCache_;
    size_t movedLightsScratch_ = 0;
    U32 movedLightSlots_ = 0;
    // A moving mesh's time steps on the device (MeshMotion): tables of device pointers to arrays there, kept with it.
    MeshMotion uploadMotion(const cpu::MeshData& m, const shade::MeshAttributes& attributes, const std::vector<U64L>& vertexKeys, std::vector<B>& keep) {
        auto table = [&](const std::vector<U64L>& pointers) {
            B b = upload(pointers.data(), pointers.size() * sizeof(U64L));
            keep.push_back(b);
            return reinterpret_cast<const surface::V3* const*>(b->pointer);
        };
        auto steps = [&](const surface::V3* first, const std::vector<std::vector<surface::V3>>& later) {
            std::vector<U64L> pointers{reinterpret_cast<U64L>(first)};
            for (const auto& s : later) {
                B b = upload(s.data(), s.size() * sizeof(surface::V3));
                keep.push_back(b);
                pointers.push_back(b->pointer);
            }
            return table(pointers);
        };
        MeshMotion motion;
        motion.steps = unsigned(vertexKeys.size());
        motion.kind = m.subdSteps ? MotionSubd : m.stepShading ? MotionStepped : MotionPolygon;
        motion.positions = table(vertexKeys);
        if (!m.stepNormals.empty()) motion.normals = steps(attributes.normals, m.stepNormals);
        if (!m.stepDpds.empty()) {
            motion.dpds = steps(attributes.dpds, m.stepDpds);
            motion.dpdt = steps(attributes.dpdt, m.stepDpdt);
        }
        if (!m.normals1.empty()) {
            B b = upload(m.normals1.data(), m.normals1.size() * sizeof(surface::V3));
            keep.push_back(b);
            motion.normals1 = reinterpret_cast<const surface::V3*>(b->pointer);
        }
        return motion;
    }
    size_t meshesReused_ = 0;
    // What a placement-only edit changes (updatePlacements): every mesh's instance and attribute record as the card has
    // them, the top trees' buffers (replaced, not kept), each light's tables on the card by their host view, and a moved
    // instance's primitive attributes.
    std::vector<OptixInstance> instances_;
    std::vector<shade::MeshAttributes> meshAttributesHost_;
    std::vector<B> tops_;
    std::map<const void*, const void*> lightTables_;
    std::map<unsigned, std::vector<B>> placedKeep_;
    // The surfaces' tree over the instances, and the volumes' bounds' (the same geometry, each bounding mesh with its
    // visibility, both sides).
    void buildTops(const cpu::Plan& plan) {
        std::vector<B> made;
        auto instanceBuffer = upload(instances_.data(), instances_.size() * sizeof(OptixInstance));
        OptixBuildInput input{};
        input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances = instanceBuffer->pointer;
        input.instanceArray.numInstances = unsigned(instances_.size());
        auto top = build(input);
        scene_.traversable = top.second;
        made.insert(made.end(), {instanceBuffer, top.first});
        if (!plan.volumes.empty()) {
            std::vector<OptixInstance> bounds = instances_;
            for (unsigned i = 0; i < bounds.size(); ++i) {
                bounds[i].visibilityMask = plan.meshes[i].volume >= 0 ? (plan.meshes[i].visibility & 0xffu) : 0u;
                bounds[i].flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
            }
            auto boundsBuffer = upload(bounds.data(), bounds.size() * sizeof(OptixInstance));
            OptixBuildInput vinput{};
            vinput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            vinput.instanceArray.instances = boundsBuffer->pointer;
            vinput.instanceArray.numInstances = unsigned(bounds.size());
            auto vtop = build(vinput);
            scene_.volumeTraversable = vtop.second;
            made.insert(made.end(), {boundsBuffer, vtop.first});
        }
        tops_ = std::move(made);
    }
    // The lights' records (their tables already on the card: deviceLights point to them), a portal's reference copied as
    // the card has it, the mesh lights, light sets and the lights' trees.
    void uploadLights(const cpu::Plan& plan, std::vector<lights::Light>& deviceLights) {
        lightKeep_.clear();
        for (auto& l : deviceLights) {
            if (l.type != lights::Portal) continue;
            B b = upload(&deviceLights[l.refIndex], sizeof(lights::Light));
            lightKeep_.push_back(b);
            l.ref = reinterpret_cast<const lights::Light*>(b->pointer);
        }
        scene_.lights = upload(deviceLights.data(), deviceLights.size() * sizeof(lights::Light));
        scene_.meshLights = upload(plan.meshLights.data(), plan.meshLights.size() * sizeof(U32));
        scene_.lightSetBits = plan.lightSetBits.empty() ? nullptr : upload(plan.lightSetBits.data(), plan.lightSetBits.size() * sizeof(U32));
        scene_.lightMotion = scene_.lightMoved = nullptr;
        if (!plan.lightMotion.empty()) {
            scene_.lightMotion = upload(plan.lightMotion.data(), plan.lightMotion.size() * sizeof(lights::LightMotion));
            std::vector<U32> movedAt(plan.lights.size(), ~0u);
            for (size_t k = 0; k < plan.lightMotion.size(); ++k) movedAt[plan.lightMotion[k].light] = U32(k);
            scene_.lightMoved = upload(movedAt.data(), movedAt.size() * sizeof(U32));
        }
        scene_.lightTree = scene_.lightTreeRoots = nullptr;
        if (!plan.lightTree.empty()) {
            scene_.lightTree = upload(plan.lightTree.data(), plan.lightTree.size() * sizeof(lights::TreeNode));
            scene_.lightTreeRoots = upload(plan.lightTreeRoots.data(), plan.lightTreeRoots.size() * sizeof(int));
        }
    }
    std::vector<B> lightKeep_;
    // The grids on the device: dense ones (mapped BaseVolumes' baked extinction) and NanoVDB ones (VDB volumes); each grid's
    // host data as sent (a moved volume's baked grid made again is another, updatePlacements).
    std::vector<const void*> gridsSent_;
    // A grid's host data, dense or NanoVDB, by which a grid made again is told from the one sent.
    static const void* gridHeld(const volume::Grid& g) { return g.dense ? static_cast<const void*>(g.dense) : g.data; }
    // The glowing volumes' emissive regions and their distributions' pool (volume_emission.h), for the path tracer's
    // light samples toward them; none without (the integrator then finds no regions).
    void uploadRegions(const cpu::Plan& plan) {
        scene_.emissiveRegions = nullptr;
        scene_.emissionPool = nullptr;
        scene_.emissiveRegionCount = 0;
        if (plan.emissiveRegions.empty() || !volume::GpuEmission) return;
        gpu::Allocating what("the glowing volumes' regions");
        scene_.emissiveRegions = upload(plan.emissiveRegions.data(), plan.emissiveRegions.size() * sizeof(volume::Region));
        const std::vector<float> pool = plan.emissionPool.empty() ? std::vector<float>{0.0f} : plan.emissionPool;
        scene_.emissionPool = upload(pool.data(), pool.size() * sizeof(float));
        scene_.emissiveRegionCount = U32(plan.emissiveRegions.size());
    }
    void uploadGrids(const cpu::Plan& plan) {
        scene_.gridData.clear();
        scene_.volumeGrids = nullptr;
        scene_.volumeGridCount = 0;
        gridsSent_.clear();
        if (plan.volumeGridViews.empty()) return;
        std::vector<volume::Grid> views(plan.volumeGridViews.begin(), plan.volumeGridViews.end());
        for (auto& g : views) {
            gridsSent_.push_back(gridHeld(g));
            if (!g.dense) {
                // A VDB grid: its NanoVDB buffer whole (it holds its nodes by offsets, so it reads where it lands), for
                // volume_grid.h to sample on the card as the host does.
                const void* host = g.data;
                g.data = nullptr;
                if (!host || !volume::GpuVdb) continue;  // not sent while the card does not render VDB volumes
                const size_t bytes = size_t(static_cast<const nanovdb::GridData*>(host)->mGridSize);
                if (!bytes) continue;
                gpu::Allocating what("a VDB volume's grid");
                B data = upload(host, bytes);
                g.data = reinterpret_cast<const void*>(data->pointer);
                scene_.gridData.push_back(data);
                continue;
            }
            g.data = nullptr;
            const size_t floats = size_t(g.dims[0]) * size_t(g.dims[1]) * size_t(g.dims[2]) * 3;
            if (!floats) {
                g.dense = nullptr;
                continue;
            }
            B data = upload(g.dense, floats * sizeof(float));
            g.dense = reinterpret_cast<const float*>(data->pointer);
            scene_.gridData.push_back(data);
        }
        scene_.volumeGrids = upload(views.data(), views.size() * sizeof(volume::Grid));
        scene_.volumeGridCount = U32(views.size());
    }
    struct Scene {
        std::vector<B> keep;
        U64 traversable = 0, volumeTraversable = 0, standInTraversable = 0;
        bool defersVolumes = false;  // volumes the GPU defers to the CPU lane (standIns)
        B volumeGrids;               // the volumes' grids as the device reads them, and the dense grids' data
        std::vector<B> gridData;
        U32 volumeGridCount = 0;
        B emissiveRegions, emissionPool;  // the glowing volumes' regions (uploadRegions)
        U32 emissiveRegionCount = 0;
        B provSums, provCounts, hostProvSums, hostProvCounts;
        B geometry, materials, lights, meshLights, lightSetBits, lightTree, lightTreeRoots, lightMotion, lightMoved, volumes, meshVolumes, walkSequences, meshMaterials, meshAttributes, meshMotion, materialPool, scratch, config, maps, textures, sums, odd, counts,
            hostSums,
            hostOdd, hostCounts, aovTables, geomLabels, aovs, hostAovs, aovClosest, hostAovClosest, hostAovDepths, heat, hostHeat, cryptoHeads,
            cryptoNodes, cryptoNodeCount;
        B passMask[PassSlots], passBase[PassSlots], passStaging[PassSlots], errors, hostErrors;
        std::shared_ptr<gpu::Event> filmDone, errorsDone, previewDone, aovInitDone;
        size_t mipBytes = 0;
    };
    // The fine-detail cache: every level-0 tile of the scene's textures has an
    // id; resident ones occupy a slot of the slab.
    struct Fine {
        U32 tiles = 0, slots = 0, requestCapacity = 0;
        U64 requestHead = 0;
        size_t loaded = 0, evicted = 0;
        B tileState, requests, touched, pageEntry, slab;
        // A level-0 tile's host level, its page there and its size: its texels decoded on the host when first sent up
        // (tex::Mip::load) where the level is read on demand.
        struct Source { const tex::Mip* mip; U32 page; U32 bytes; U32 kind; };
        std::vector<Source> sources;      // host texels of each tile id
        // The cache's slots by size, one kind for each size the scene's formats have (an 8-bit grey map's tile takes 1 KB, a
        // float one's 12 KB), numbered on from first: its own share of the budget (slots), its free slots and its eviction.
        // Its slots are made as tiles come to need them, a block of 64 MB at a time (made of them so far), so a scene whose
        // tiles are mostly never read, or flat, holds only what it uses.
        struct Kind {
            size_t slotBytes = 0;
            U32 slots = 0, first = 0, tiles = 0, made = 0, blockSlots = 1;
            std::vector<U32> freeSlots;
            std::vector<B> blocks;
        };
        std::vector<Kind> classes;
        size_t slabBytes = 0;             // the slots' bytes made so far
        size_t limitBytes = 0;            // and at most (the budget)
        size_t flat = 0;                  // flat tiles held as one value (flatPool)
        B flatPool;                       // each tile id's 16 bytes: a flat tile's value
        // Wide tiles (tex::widePage: a float file's tile past half floats' range, in floats), each in memory of its own for the
        // session: a sun's few tiles in an HDR.
        std::vector<B> wide;
        std::vector<U32> slotTile, tileSlot;
        std::vector<U64> lastUse;
        std::vector<U32> pending;
        size_t requestBytes() const { return size_t(requestCapacity) * sizeof(U32); }
    };

    static void ox(OptixResult e) { gpu::ox(e); }
    B device(size_t bytes) { return std::make_shared<gpu::Buffer>(driver, bytes); }
    B pinned(size_t bytes) { return std::make_shared<gpu::Buffer>(driver, bytes, true); }
    void ensure(B& b, size_t bytes) {
        if (!b || b->bytes < bytes) b = pinned(bytes);
    }
    void copyIn(const B& d, const void* host, size_t bytes) {
        gpu::Driver::check(driver.cuMemcpyHtoDAsync_v2_p(d->pointer, host, bytes, stream_));
    }
    void copyOut(const B& h, const B& d, size_t bytes) {
        gpu::Driver::check(driver.cuMemcpyDtoHAsync_v2_p(h->host(), d->pointer, bytes, stream_));
    }
    void memset32(CUdeviceptr p, unsigned value, size_t words) {
        if (words) gpu::Driver::check(driver.cuMemsetD32Async_p(p, value, words, stream_));
    }
    B upload(const void* data, size_t bytes) {
        auto h = pinned(std::max<size_t>(bytes, 4)), d = device(std::max<size_t>(bytes, 4));
        std::memcpy(h->host(), data, bytes);
        copyIn(d, h->host(), bytes);
        uploads_.push_back(h);
        return d;
    }
    // An acceleration structure, compacted (OptiX's compaction), its temporary memory and the uploads it read let go as
    // soon as it is made. Before, every build's temporary memory and every upload's pinned copy were held until the whole
    // scene was ready, so the card's peak was their sum (about 23 bytes of the card a byte of geometry).
    std::pair<B, OptixTraversableHandle> build(const OptixBuildInput& input, unsigned keys = 1) {
        OptixAccelBuildOptions options{};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        if (keys > 1) {  // geometry in motion: its keys evenly over the shutter, the samples' times 0 to 1
            options.motionOptions.numKeys = static_cast<unsigned short>(keys);
            options.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
            options.motionOptions.timeBegin = 0.0f;
            options.motionOptions.timeEnd = 1.0f;
        }
        OptixAccelBufferSizes sizes{};
        ox(optixAccelComputeMemoryUsage(optix_, &options, &input, 1, &sizes));
        gpu::Allocating what("geometry's ray-tracing structures");
        auto temporary = device(std::max<size_t>(sizes.tempSizeInBytes, 16));
        auto output = device(std::max<size_t>(sizes.outputSizeInBytes, 16));
        auto compactedSize = device(8);
        auto sizeHost = pinned(8);
        OptixAccelEmitDesc emit{};
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = compactedSize->pointer;
        OptixTraversableHandle handle = 0;
        ox(optixAccelBuild(optix_, stream_, &options, &input, 1, temporary->pointer, temporary->bytes, output->pointer,
                           output->bytes, &handle, &emit, 1));
        copyOut(sizeHost, compactedSize, 8);
        gpu::Driver::check(driver.cuCtxSynchronize_p());
        temporary.reset();
        uploads_.clear();  // every copy so far is done: their pinned staging goes
        uint64_t compacted = 0;
        std::memcpy(&compacted, sizeHost->host(), 8);
        builtBytes_ += output->bytes;
        if (compacted && compacted < output->bytes - output->bytes / 8) {
            auto small = device(size_t(compacted));
            OptixTraversableHandle smallHandle = 0;
            ox(optixAccelCompact(optix_, stream_, handle, small->pointer, small->bytes, &smallHandle));
            gpu::Driver::check(driver.cuCtxSynchronize_p());
            compactedBytes_ += small->bytes;
            return {small, smallHandle};
        }
        compactedBytes_ += output->bytes;
        return {output, handle};
    }
    // The acceleration structures' memory as built and as kept (the lane's memory line).
    size_t builtBytes_ = 0, compactedBytes_ = 0;

    // One more block of a kind's slots, where its budget has room and the card has the memory; false where neither.
    bool growKind(Fine::Kind& c) {
        if (c.made >= c.slots) return false;
        const U32 n = std::min(c.blockSlots, c.slots - c.made);
        try {
            gpu::Allocating what("the fine-detail texture cache");
            c.blocks.push_back(device(size_t(n) * c.slotBytes));
        } catch (const std::exception&) {
            c.slots = c.made;  // the card is full: the kind keeps the slots it has, and evicts
            return false;
        }
        for (U32 s = n; s-- > 0;) c.freeSlots.push_back(c.first + c.made + s);
        c.made += n;
        fine_.slabBytes += size_t(n) * c.slotBytes;
        return true;
    }
    // A slot's place on the card: its block, and its place in it.
    CUdeviceptr slotAddress(const Fine::Kind& c, U32 s) const {
        const U32 local = s - c.first;
        return c.blocks[local / c.blockSlots]->pointer + size_t(local % c.blockSlots) * c.slotBytes;
    }
    // A flat tile's tileSlot: resident, in no slot (its value in the flat pool).
    static constexpr U32 FlatSlot = NoSlot - 1;
    // The slot kind of a level-0 tile of this format, made as first met: its whole tile's bytes, at 16-byte steps.
    U32 kindOf(unsigned format) {
        const size_t bytes = (size_t(format == 2 ? tex::tileBytes(2, 0) : tex::Tile * tex::Tile * tex::texelBytes(format)) + 15) / 16 * 16;
        for (size_t k = 0; k < fine_.classes.size(); ++k)
            if (fine_.classes[k].slotBytes == bytes) return U32(k);
        fine_.classes.push_back({bytes});
        return U32(fine_.classes.size() - 1);
    }
    void prepareTextures(const cpu::Plan& plan) {
        std::vector<std::shared_ptr<texture::File>> files;
        for (auto* bindings : {&plan.diffuseMaps, &plan.emissionMaps})
            for (auto& pair : *bindings)
                if (std::find(files.begin(), files.end(), pair.second.file) == files.end()) files.push_back(pair.second.file);
        for (auto& file : plan.textures)
            if (std::find(files.begin(), files.end(), file) == files.end()) files.push_back(file);
        fine_ = {};
        fine_.sources.push_back({nullptr, 0, 0, 0});  // id 0 is untracked
        singleLevel_ = false;
        std::vector<std::pair<std::shared_ptr<texture::File>, B>> images;
        std::vector<std::pair<B, std::vector<U32>>> levelZeroPages;  // per file: page array and its tile ids
        size_t cardFree = 0, cardTotal = 0, mipBytes = 0;
        const bool cardKnown = memoryInfo(cardFree, cardTotal);
        for (auto& file : files) {
            cpu::need(bool(file->pyramid), "Missing host mip pyramid");
            // A realtime plan's from its first level of 1024 pixels or less (Pyramid::view).
            const auto& host = file->pyramid->view(plan.realtime);
            tex::Image descriptor{};
            descriptor.count = host.count;
            if (host.count < 2) singleLevel_ = true;  // no level up for partial shading (sizeParked)
            // Levels 1 and up are resident for the whole session, in their format (tex::Mip::format), each tile at a
            // 16-byte boundary so that a float level's texels are aligned whatever came before.
            std::vector<unsigned char> packed;
            std::vector<std::vector<size_t>> offsets(host.count);
            for (unsigned level = 1; level < host.count; ++level) {
                const auto m = host.levels[level];
                const unsigned nx = (m.width + tex::Tile - 1) / tex::Tile, ny = (m.height + tex::Tile - 1) / tex::Tile;
                for (unsigned i = 0; i < nx * ny; ++i) {
                    const unsigned tx = i % nx, ty = i / nx;
                    const size_t count = size_t(std::min(tex::Tile, m.width - tx * tex::Tile)) *
                                         std::min(tex::Tile, m.height - ty * tex::Tile);
                    packed.resize((packed.size() + 15) / 16 * 16);
                    offsets[level].push_back(packed.size());
                    const auto* from = reinterpret_cast<const unsigned char*>(m.pages[i]);
                    packed.insert(packed.end(), from, from + tex::tileBytes(m.format, U32(count)));
                }
            }
            // Levels 1 and up that would not fit, the rest of the frame besides (the session's texture overflow mode).
            mipBytes += packed.size();
            if (cardKnown && mipBytes + std::max<size_t>(cardTotal / 20, 256ull << 20) > cardFree)
                throw std::runtime_error("the textures do not fit on the card (texture overflow)");
            gpu::Allocating what("a texture's coarser levels");
            B pixels = packed.empty() ? B() : upload(packed.data(), packed.size());
            scene_.mipBytes += packed.size();
            for (unsigned level = 1; level < host.count; ++level) {
                std::vector<const surface::V3*> pages;
                for (size_t o : offsets[level]) pages.push_back(reinterpret_cast<const surface::V3*>(pixels->pointer + o));
                auto table = upload(pages.data(), pages.size() * sizeof(pages[0]));
                descriptor.levels[level] = {host.levels[level].width, host.levels[level].height,
                                            reinterpret_cast<const surface::V3* const*>(table->pointer), 0, host.levels[level].format};
                scene_.keep.push_back(table);
            }
            // Level 0 starts empty; its tiles are numbered for the cache.
            const auto m = host.levels[0];
            const unsigned nx = (m.width + tex::Tile - 1) / tex::Tile, ny = (m.height + tex::Tile - 1) / tex::Tile;
            std::vector<const surface::V3*> none(size_t(nx) * ny, nullptr);
            auto table = upload(none.data(), none.size() * sizeof(none[0]));
            const U32 base = U32(fine_.sources.size());
            std::vector<U32> ids;
            for (unsigned i = 0; i < nx * ny; ++i) {
                const unsigned tx = i % nx, ty = i / nx;
                const U32 count = std::min(tex::Tile, m.width - tx * tex::Tile) * std::min(tex::Tile, m.height - ty * tex::Tile);
                fine_.sources.push_back({&host.levels[0], i, tex::tileBytes(m.format, count), kindOf(m.format)});
                ++fine_.classes[fine_.sources.back().kind].tiles;
                ids.push_back(base + i);
            }
            descriptor.levels[0] = {m.width, m.height, reinterpret_cast<const surface::V3* const*>(table->pointer), base, m.format};
            levelZeroPages.push_back({table, ids});
            scene_.keep.push_back(table);
            if (pixels) scene_.keep.push_back(pixels);
            images.push_back({file, upload(&descriptor, sizeof(descriptor))});
        }
        fine_.tiles = U32(fine_.sources.size() - 1);
        fine_.requestCapacity = fine_.tiles + 1;
        // Each tile in a slot of its format's size: an 8-bit grey map's in a twelfth of a float one's, so the same budget holds
        // twelve times its tiles. The budget shared by the kinds as their tiles would take it. A scene without textures keeps one
        // unused slot, so the buffers below are never empty.
        if (fine_.classes.empty()) fine_.classes.push_back({size_t(tex::Tile) * tex::Tile * sizeof(surface::V3)});
        size_t need = 0;
        for (const auto& c : fine_.classes) need += size_t(std::max<U32>(c.tiles, 1)) * c.slotBytes;
        fineBudget_ = fineBudget(need);
        const double share = std::min(1.0, double(fineBudget_) / double(need));
        for (auto& c : fine_.classes) {
            c.slots = std::max<U32>(1, std::min<U32>(std::max<U32>(c.tiles, 1), U32(double(std::max<U32>(c.tiles, 1)) * share)));
            c.first = fine_.slots;
            c.blockSlots = U32(std::max<size_t>(1, (64ull << 20) / c.slotBytes));
            fine_.slots += c.slots;
            fine_.limitBytes += size_t(c.slots) * c.slotBytes;
        }
        if (fine_.tiles)
            std::printf("XPU2: fine-detail texture cache up to %.0f MB of the %.0f MB its tiles would take, made as tiles are read (%s)\n",
                        double(fine_.limitBytes) / (1 << 20), double(need) / (1 << 20), fineBudgetWhy_.c_str());
        cpu::need(fine_.slots > 0, "Fine-detail cache budget");
        fine_.tileState = device((fine_.tiles + 1) * sizeof(U32));
        fine_.touched = device(fine_.tiles + 1);
        fine_.requests = device(fine_.requestBytes());
        fine_.flatPool = device(size_t(fine_.tiles + 1) * 16);
        memset32(fine_.tileState->pointer, 0, fine_.tiles + 1);
        gpu::Driver::check(driver.cuMemsetD8Async_p(fine_.touched->pointer, 0, fine_.tiles + 1, stream_));
        std::vector<U64L> entries(fine_.tiles + 1, 0);
        for (auto& [table, ids] : levelZeroPages)
            for (size_t i = 0; i < ids.size(); ++i) entries[ids[i]] = table->pointer + i * sizeof(void*);
        fine_.pageEntry = upload(entries.data(), entries.size() * sizeof(U64L));
        fine_.slotTile.assign(fine_.slots, 0);
        fine_.tileSlot.assign(fine_.tiles + 1, NoSlot);
        fine_.lastUse.assign(fine_.tiles + 1, 0);

        std::vector<tex::Maps> maps;
        for (const auto& mesh : plan.meshes) {
            tex::Maps pair;
            auto fill = [&](tex::Binding& b, const auto& bindings) {
                auto it = bindings.find(mesh.material);
                if (it == bindings.end()) return;
                b = it->second.descriptor();
                for (auto& image : images)
                    if (image.first == it->second.file) {
                        b.image = reinterpret_cast<const tex::Image*>(image.second->pointer);
                        return;
                    }
                throw std::runtime_error("Unmapped texture residency");
            };
            fill(pair.diffuse, plan.diffuseMaps);
            fill(pair.emission, plan.emissionMaps);
            maps.push_back(pair);
        }
        scene_.maps = upload(maps.data(), maps.size() * sizeof(tex::Maps));
        // The maps' texture table, in the plan's order (shade_mapnodes.h).
        if (!plan.textures.empty()) {
            std::vector<const tex::Image*> table;
            for (auto& file : plan.textures) {
                const tex::Image* image = nullptr;
                for (auto& entry : images)
                    if (entry.first == file) image = reinterpret_cast<const tex::Image*>(entry.second->pointer);
                if (!image) throw std::runtime_error("Unmapped texture residency");
                table.push_back(image);
            }
            scene_.textures = upload(table.data(), table.size() * sizeof(table[0]));
        }
        for (auto& image : images) scene_.keep.push_back(image.second);
    }

    // Loads every requested tile from host RAM into a slot of its kind, evicting that kind's least recently used where its
    // slots are full, and queues the page-table updates ahead of the next launch on the same stream. A flat tile (all its
    // texels one value, tex::flatPage) takes no slot: its value goes to its place in the flat pool, for good.
    void commitTiles() {
        if (fine_.pending.empty()) return;
        std::vector<TileUpdate> updates;
        // Tiles not yet decoded on the host (a level read on demand) decoded first, on several threads at once.
        {
            std::vector<U32> missing;
            for (size_t k = 0; k < fine_.pending.size(); ++k) {
                const auto& source = fine_.sources[fine_.pending[k]];
                if (source.mip && source.mip->load && !source.mip->pages[source.page]) missing.push_back(fine_.pending[k]);
            }
            const unsigned threads = unsigned(std::min<size_t>(missing.size(), std::max(1u, std::thread::hardware_concurrency() / 2)));
            std::atomic<size_t> next{0};
            // A tile that cannot be read is the frame's failure, said with its reason, once every thread has left: thrown past
            // threads still running, on this thread or theirs, it ended the process with nothing said ("terminate called").
            std::mutex failing;
            std::exception_ptr failed;
            auto work = [&] {
                try {
                    for (size_t k; (k = next++) < missing.size();) {
                        const auto& source = fine_.sources[missing[k]];
                        source.mip->load(source.mip->loader, 0, source.page);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failing);
                    if (!failed) failed = std::current_exception();
                    next = missing.size();
                }
            };
            std::vector<std::thread> pool;
            try {
                for (unsigned k = 1; k < threads; ++k)
                    pool.emplace_back([&] {
                        multi::OnNode near(options_.node);
                        work();
                    });
            } catch (const std::system_error&) {
                // No more threads to be had: those there are, and this one, do the work.
            }
            if (threads) work();
            for (auto& t : pool) t.join();
            if (failed) std::rethrow_exception(failed);
        }
        // Each kind's least recently used resident tiles, not the ones asked for now, as many as its slots are short.
        std::vector<size_t> wanted(fine_.classes.size(), 0);
        for (U32 tile : fine_.pending) {
            const auto& source = fine_.sources[tile];
            const void* page = source.mip->pages[source.page];
            if (!tex::flatPage(page) && !tex::widePage(page)) ++wanted[source.kind];  // let go meanwhile: taken as needing a slot
        }
        for (size_t k = 0; k < fine_.classes.size(); ++k) {
            auto& c = fine_.classes[k];
            const size_t room = c.freeSlots.size() + (c.slots - c.made);  // free, or still to be made
            if (wanted[k] <= room) continue;
            std::vector<U32> resident;
            for (U32 s = c.first; s < c.first + c.made; ++s)
                if (fine_.slotTile[s]) resident.push_back(fine_.slotTile[s]);
            std::sort(resident.begin(), resident.end(), [&](U32 a, U32 b) { return fine_.lastUse[a] < fine_.lastUse[b]; });
            const size_t need = wanted[k] - room;
            for (size_t i = 0; i < resident.size() && i < need; ++i) {
                const U32 tile = resident[i], s = fine_.tileSlot[tile];
                fine_.tileSlot[tile] = NoSlot;
                fine_.slotTile[s] = 0;
                c.freeSlots.push_back(s);
                updates.push_back({tile, 0, 0});
                ++fine_.evicted;
            }
        }
        size_t most = 16;
        for (const auto& c : fine_.classes) most = std::max(most, c.slotBytes);
        gpu::Allocating what("texture tiles on their way to the card");
        auto staging = pinned(std::max<size_t>(fine_.pending.size(), 1) * most);
        size_t n = 0;
        std::vector<U32> deferred;
        std::vector<B> extra;  // wide tiles' staging, kept until their copies are done
        for (U32 tile : fine_.pending) {
            const auto& source = fine_.sources[tile];
            auto& c = fine_.classes[source.kind];
            char* at = static_cast<char*>(staging->host()) + n * most;
            tex::spill::Guard guard;  // a tile may be let go under memory pressure, or have been since it was decoded
            const surface::V3* texels = source.mip->pages[source.page];
            while (!texels && source.mip->load) {
                source.mip->load(source.mip->loader, 0, source.page);
                texels = source.mip->pages[source.page];
            }
            cpu::need(texels != nullptr, "A fine-detail tile with no texels");
            CUdeviceptr to = 0;
            size_t bytes = 0;
            U32 s = FlatSlot;
            if (tex::flatPage(texels)) {
                to = fine_.flatPool->pointer + size_t(tile) * 16;
                bytes = sizeof(surface::V3);
                std::memcpy(at, tex::flatValue(texels), bytes);
                ++fine_.flat;
            } else if (tex::widePage(texels)) {
                const U32 count = source.bytes / tex::texelBytes(source.mip->format);
                bytes = size_t(count) * tex::texelBytes(tex::wideFormat(source.mip->format));
                auto own = pinned(bytes);
                std::memcpy(own->host(), tex::pageAddress(texels), bytes);
                fine_.wide.push_back(device(bytes));
                gpu::Driver::check(driver.cuMemcpyHtoDAsync_v2_p(fine_.wide.back()->pointer, own->host(), bytes, stream_));
                extra.push_back(own);
                fine_.tileSlot[tile] = FlatSlot;
                fine_.lastUse[tile] = launches_ + 1;
                updates.push_back({tile, 0, U64L(fine_.wide.back()->pointer) | 2ull});
                ++fine_.loaded;
                continue;
            } else {
                if (c.freeSlots.empty()) growKind(c);
                if (c.freeSlots.empty()) {  // more asked for than its kind's slots hold: next time
                    deferred.push_back(tile);
                    continue;
                }
                s = c.freeSlots.back();
                c.freeSlots.pop_back();
                to = slotAddress(c, s);
                bytes = size_t(source.bytes);
                std::memcpy(at, texels, bytes);
                fine_.slotTile[s] = tile;
            }
            gpu::Driver::check(driver.cuMemcpyHtoDAsync_v2_p(to, at, bytes, stream_));
            fine_.tileSlot[tile] = s;
            fine_.lastUse[tile] = launches_ + 1;
            updates.push_back({tile, 0, U64L(to) | (s == FlatSlot ? 1ull : 0ull)});
            ++fine_.loaded;
            ++n;
        }
        fine_.pending = std::move(deferred);
        if (!updates.empty()) {
            auto list = pinned(updates.size() * sizeof(TileUpdate));
            std::memcpy(list->host(), updates.data(), updates.size() * sizeof(TileUpdate));
            auto deviceList = device(updates.size() * sizeof(TileUpdate));
            copyIn(deviceList, list->host(), list->bytes);
            LaunchParams p = baseParams();
            p.mode = 1;
            p.updates = reinterpret_cast<const TileUpdate*>(deviceList->pointer);
            p.updateCount = U32(updates.size());
            auto& slot = takeParams(p);
            ox(optixLaunch(program_->pipeline, stream_, slot.device->pointer, sizeof(LaunchParams), &program_->sbt,
                           p.updateCount, 1, 1));
            auto done = std::make_shared<gpu::Event>(driver);
            gpu::Driver::check(driver.cuEventRecord_p(done->value, stream_));
            slot.done = done;
            extra.insert(extra.end(), {staging, list, deviceList});
            retire_.push_back({done, std::move(extra)});
        }
        while (!retire_.empty() && retire_.front().first->ready()) retire_.pop_front();
    }

    LaunchParams baseParams() const {
        LaunchParams p{};
        p.config = reinterpret_cast<const mixed::Config*>(scene_.config->pointer);
        p.materials = reinterpret_cast<const surface::Lambert*>(scene_.materials->pointer);
        p.lights = scene_.lights ? reinterpret_cast<const lights::Light*>(scene_.lights->pointer) : nullptr;
        p.meshLights = scene_.meshLights ? reinterpret_cast<const U32*>(scene_.meshLights->pointer) : nullptr;
        p.lightSetBits = scene_.lightSetBits ? reinterpret_cast<const U32*>(scene_.lightSetBits->pointer) : nullptr;
        p.lightSetWords = plan_->lightSetWords;
        p.lightTree = scene_.lightTree ? reinterpret_cast<const lights::TreeNode*>(scene_.lightTree->pointer) : nullptr;
        p.lightTreeRoots = scene_.lightTreeRoots ? reinterpret_cast<const int*>(scene_.lightTreeRoots->pointer) : nullptr;
        p.volumes = scene_.volumes ? reinterpret_cast<const volume::Shader*>(scene_.volumes->pointer) : nullptr;
        p.meshVolumes = scene_.meshVolumes ? reinterpret_cast<const int*>(scene_.meshVolumes->pointer) : nullptr;
        p.volumeGrids = scene_.volumeGrids ? reinterpret_cast<const volume::Grid*>(scene_.volumeGrids->pointer) : nullptr;
        p.volumeGridCount = scene_.volumeGrids ? scene_.volumeGridCount : 0u;
        p.emissiveRegions = scene_.emissiveRegions ? reinterpret_cast<const volume::Region*>(scene_.emissiveRegions->pointer) : nullptr;
        p.emissionPool = scene_.emissionPool ? reinterpret_cast<const float*>(scene_.emissionPool->pointer) : nullptr;
        p.emissiveRegionCount = scene_.emissiveRegions ? scene_.emissiveRegionCount : 0u;
        p.walkSequences = scene_.walkSequences ? reinterpret_cast<const float*>(scene_.walkSequences->pointer) : nullptr;
        p.meshMaterials = scene_.meshMaterials ? reinterpret_cast<const shade::MaterialRecord*>(scene_.meshMaterials->pointer) : nullptr;
        p.meshAttributes = scene_.meshAttributes ? reinterpret_cast<const shade::MeshAttributes*>(scene_.meshAttributes->pointer) : nullptr;
        p.meshMotion = scene_.meshMotion ? reinterpret_cast<const MeshMotion*>(scene_.meshMotion->pointer) : nullptr;
        p.lightMotion = scene_.lightMotion ? reinterpret_cast<const lights::LightMotion*>(scene_.lightMotion->pointer) : nullptr;
        p.lightMoved = scene_.lightMoved ? reinterpret_cast<const U32*>(scene_.lightMoved->pointer) : nullptr;
        p.movedLightSlots = scene_.lightMotion ? movedLightSlots_ : 0u;
        p.materialPool = scene_.materialPool ? reinterpret_cast<const float*>(scene_.materialPool->pointer) : nullptr;
        p.scratch = scene_.scratch ? reinterpret_cast<unsigned char*>(scene_.scratch->pointer) : nullptr;
        p.scratchStride = scratchStride(plan_->dwaSlots, plan_->glitter) + aovScratch_;
        p.aovOffset = scratchStride(plan_->dwaSlots, plan_->glitter);
        p.aovTables = reinterpret_cast<const aov::Tables*>(scene_.aovTables->pointer);
        p.geomLabels = scene_.geomLabels ? reinterpret_cast<const int*>(scene_.geomLabels->pointer) : nullptr;
        p.aovs = scene_.aovs ? reinterpret_cast<float*>(scene_.aovs->pointer) : nullptr;
        p.aovClosest = scene_.aovClosest ? reinterpret_cast<U64L*>(scene_.aovClosest->pointer) : nullptr;
        p.aovClosestChannels = aovClosestChannels_;
        p.heat = scene_.heat ? reinterpret_cast<double*>(scene_.heat->pointer) : nullptr;
        p.secondsPerTick = secondsPerTick_;
        p.movedLightsOffset = p.scratchStride + cryptoScratch_;
        p.scratchStride += cryptoScratch_ + movedLightsScratch_;
        p.registersOffset = p.scratchStride;
        p.mapRegisters = plan_->mapRegisters;
        p.scratchStride += registersScratch_;
        if (cryptoEvents_) {
            p.cryptoOffset = scratchStride(plan_->dwaSlots, plan_->glitter) + aovScratch_;
            p.cryptoEvents = cryptoEvents_;
            p.cryptoHeads = reinterpret_cast<U32*>(scene_.cryptoHeads->pointer);
            p.cryptoNodes = reinterpret_cast<U32*>(scene_.cryptoNodes->pointer);
            p.cryptoNodeCount = reinterpret_cast<U32*>(scene_.cryptoNodeCount->pointer);
            p.cryptoCategories = cryptoCategories_;
            p.cryptoMultiPresence = cryptoMultiPresence_;
            p.cryptoExtras = cryptoExtras_;
            p.cryptoNodeWords = cryptoNodeWords_;
            p.cryptoNodeCapacity = cryptoNodeCapacity_;
        }
        p.glitterOffset = plan_->glitter ? glitterOffset(plan_->dwaSlots) : 0;
        p.traversable = scene_.traversable;
        p.volumeTraversable = scene_.volumeTraversable;
        p.standInTraversable = scene_.standInTraversable;
        p.geometry = reinterpret_cast<const mixed::Geometry*>(scene_.geometry->pointer);
        p.maps = scene_.maps ? reinterpret_cast<const tex::Maps*>(scene_.maps->pointer) : nullptr;
        p.textures = scene_.textures ? reinterpret_cast<const tex::Image* const*>(scene_.textures->pointer) : nullptr;
        p.textureCount = scene_.textures ? U32(plan_->textures.size()) : 0;
        p.ring = reinterpret_cast<const Chunk*>(ring_->pointer);
        p.ringCapacity = options_.ringChunks;
        p.itemsPerChunk = itemsPerChunk_;
        p.maxTries = 8;
        p.parked = reinterpret_cast<Parked*>(parked_->pointer);
        p.parkedCapacity = parkedCapacity_;
        p.deferred = deferred_ ? reinterpret_cast<Parked*>(deferred_->pointer) : nullptr;
        p.deferredCapacity = deferredCapacity_ ? deferredCapacity_ : 1u;
        p.state = reinterpret_cast<DeviceState*>(state_->pointer);
        p.partial = partial_ ? 1u : 0u;
        p.profile = std::getenv("XPU2_PROFILE") ? 1u : 0u;
        p.refineCapacity = refineCapacity_;
        p.lowFlags = reinterpret_cast<U32*>(lowFlags_->pointer);
        p.provSums = reinterpret_cast<double*>(scene_.provSums->pointer);
        p.provCounts = reinterpret_cast<U32*>(scene_.provCounts->pointer);
        p.refines = reinterpret_cast<Refine*>(refines_->pointer);
        p.sums = reinterpret_cast<double*>(scene_.sums->pointer);
        p.counts = reinterpret_cast<U32*>(scene_.counts->pointer);
        p.adaptive = adaptive_ ? 1u : 0u;
        p.pixels = plan_->width * plan_->height;
        p.odd = reinterpret_cast<double*>(scene_.odd->pointer);
        p.errors = scene_.errors ? reinterpret_cast<float*>(scene_.errors->pointer) : nullptr;
        for (U32 s = 0; s < PassSlots; ++s) {
            p.passMask[s] = reinterpret_cast<const U64L*>(scene_.passMask[s]->pointer);
            p.passBase[s] = reinterpret_cast<const U32*>(scene_.passBase[s]->pointer);
        }
        if (fine_.tiles) {
            p.tileState = reinterpret_cast<U32*>(fine_.tileState->pointer);
            p.requests = reinterpret_cast<U32*>(fine_.requests->pointer);
            p.requestCapacity = fine_.requestCapacity;
            p.touched = reinterpret_cast<unsigned char*>(fine_.touched->pointer);
            p.pageEntry = reinterpret_cast<const U64L*>(fine_.pageEntry->pointer);
            p.slab = nullptr;  // slots are in blocks; each update carries its address
        }
        return p;
    }
    bool paramsFree() const {
        for (auto& p : params_)
            if (!p.done || p.done->ready()) return true;
        return false;
    }
    ParamSlot& takeParams(const LaunchParams& p) {
        for (auto& slot : params_)
            if (!slot.done || slot.done->ready()) {
                std::memcpy(slot.host->host(), &p, sizeof(p));
                copyIn(slot.device, slot.host->host(), sizeof(p));
                slot.done.reset();
                return slot;
            }
        throw std::runtime_error("No free launch parameter slot");
    }
    // The parked queue for this frame. Only a texture's missing tile parks a sample, so a scene without textures holds no
    // real queue; with them, each of the frame's samples may wait on the queue and run again, twice over at most, up to
    // 2^26 (1 GB of the card). With partial shading a missing tile is read a level up, always on the card, so nothing parks
    // but where a texture has no level up (one of a single pixel): such a frame keeps the small queue too, and the card its
    // gigabyte. The pass chain, which parks, sizes it as before.
    void sizeParked(bool partial) {
        mayPark_ = fine_.tiles > 0 && (!partial || singleLevel_);
        const U64 tiles = U64((plan_->width + 7) / 8) * ((plan_->height + 7) / 8);
        const U32 wanted = mayPark_ ? U32(std::min<U64>(2 * tiles * TilePixels * plan_->spp + 1024, 1ull << 26)) : 1024u;
        if (parked_ && parkedCapacity_ >= wanted) return;
        parked_.reset();
        parkedCapacity_ = wanted;
        gpu::Allocating what("samples waiting on texture tiles (parked queue)");
        parked_ = device(size_t(parkedCapacity_) * sizeof(Parked));
    }
    // The highest the parked queue can have reached: the last readback plus
    // everything the launches still in flight could park.
    U64 parkedBound() const {
        U64 bound = parkedTail_;
        for (auto& f : inFlight_) bound += f.mayPark;
        return bound;
    }
    U64 oldestChunk() const { return inFlight_.empty() ? published_ : inFlight_.front().firstChunk; }
    // The most provisional samples the launches in flight may still make (the host reads the ring after each).
    U64 refineBound() const {
        U64 bound = 0;
        for (auto& f : inFlight_) bound += f.mayRefine;
        return bound;
    }

    void cleanup() noexcept {
        try {
            if (driver.context) driver.drain();
        } catch (...) {
        }
        inFlight_.clear();
        retire_.clear();
        uploads_.clear();
        scene_ = {};
        tops_.clear();
        lightKeep_.clear();
        placedKeep_.clear();
        fine_ = {};
        for (auto& p : params_) p = {};
        for (auto& r : readbacks_) r = {};
        for (auto& s : staging_) s.reset();
        ring_.reset();
        state_.reset();
        parked_.reset();
        lowFlags_.reset();
        refines_.reset();
        for (auto& r : refineIn_) r.reset();
        if (stream_) driver.cuStreamDestroy_v2_p(stream_);
        stream_ = nullptr;
        if (chain_.module) driver.cuModuleUnload_p(chain_.module);
        chain_ = {};
        mapped_.destroy();
        for (auto& p : programs_) p.destroy();
        if (optix_) optixDeviceContextDestroy(optix_);
        optix_ = nullptr;
    }

    GpuOptions options_;
    OptixDeviceContext optix_ = nullptr;
    Program programs_[2];  // uniform, adaptive
    Program mapped_;       // programs_[0] with the mapped volume module (made by begin where a scene needs it)
    bool volumeMaps_ = false;  // the scene has volumes whose maps the card runs
    Program* program_ = &programs_[0];
    bool single_ = false;  // one program for every frame
    Chain chain_;
    DeviceState chainState_{};
    double chainStages_[4] = {};
    std::vector<double> chainRenders_;
    U64 chainsLaunched_ = 0, chainsCollected_ = 0;
    CUstream stream_ = nullptr;
    B ring_, state_, parked_;
    // The deferred samples' ring (pinned), the host's place in it, and those read but not yet taken.
    B deferred_;
    U32 deferredCapacity_ = 0;
    U64 deferredHead_ = 0;
    std::vector<Parked> deferredOut_;
    // Partial shading: on this frame; each launch thread's mark; the provisional samples' ring (pinned), the host's
    // place in it and those read but not yet taken; each launch slot's provisional samples to render again.
    bool partial_ = false;
    bool exceptionLogged_ = false;  // the diagnostic's first exception logged
    U64L profile_[ProfileStages] = {};
    B lowFlags_, refines_;
    U32 refineCapacity_ = 1u << 22;
    U64 refineHead_ = 0;
    std::vector<Refine> refinesOut_;
    std::array<B, Slots> refineIn_;
    std::array<ParamSlot, 8> params_;
    std::array<Readback, 4> readbacks_;
    std::array<B, 4> staging_;
    std::vector<B> uploads_;
    std::deque<std::pair<std::shared_ptr<gpu::Event>, std::vector<B>>> retire_;
    std::deque<Flight> inFlight_;
    cpu::PlanView plan_;
    Scene scene_;
    Fine fine_;
    U32 itemsPerChunk_ = 0, parkedCapacity_ = 0;
    bool mayPark_ = false;  // the frame may park samples (sizeParked)
    bool singleLevel_ = false;  // a texture of one level, which partial shading cannot read a level up
    bool adaptive_ = false;
    // The AOV film's channels and closest depths a pixel, a sample's scratch for them, and the clock's seconds a tick.
    U32 aovChannels_ = 0, aovClosest_ = 0, aovClosestChannels_ = 0;
    std::vector<aov::Entry> aovEntries_;  // the tables' entries, for reading the closest filter's words
    mutable bool closestLanded_ = true;   // the downloaded words read into the AOV film and its depths
    size_t aovScratch_ = 0;
    U32 cardThreads_ = 0;  // the launch width by the card (constructor); prepare's is at most this
    size_t registersScratch_ = 0;  // a sample's map registers (X2_MAP_REGISTERS)
    // Cryptomatte: events a thread and their scratch, categories, multi presence, extras, node size and pool.
    U32 cryptoEvents_ = 0, cryptoCategories_ = 1, cryptoMultiPresence_ = 0, cryptoExtras_ = 0, cryptoNodeWords_ = 5, cryptoNodeCapacity_ = 0;
    size_t cryptoScratch_ = 0;
    double secondsPerTick_ = 0.0;
    std::vector<U32> passDone_;
    // The multi-device extension (follow).
    std::vector<std::shared_ptr<GpuLane>> followers_;  // shared: a denoiser on one keeps it (xpu2_session.cc)
    mutable std::vector<U32> passSum_;
    std::vector<float> pooled_;
    gpu::B errorSums_, errorOdd_, errorCounts_;  // the film read back with the errors, to pool them (poolErrors)
    bool countsWithErrors_ = false, filmAdded_ = false, previewAdded_ = false, errorsPooled_ = false;
    U64 published_ = 0, launches_ = 0;
    U64 parkedTail_ = 0, parkedIssued_ = 0;
    Completion completed_;
};

}  // namespace xpu2::interactive::async

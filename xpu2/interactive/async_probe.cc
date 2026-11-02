// Timing and image worker for the asynchronous renderer. No Modo API.
//
// async_probe scene runtime ptx assets output threads repeats mode
//             [cellSize samplesPerChunk windowMs seed sampling override]
// seed and sampling may be "-" for the scene's own; override is an .rdla read
// over the scene (such as adaptive sampling settings), or "-". The adaptive GPU
// program and the pass chain's kernels are read from adaptive/ beside the given
// program. XPU2_ASYNC_HOST_PASSES=1 makes adaptive passes on the host instead.
// mode: 0 CPU only, 1 GPU only, 2 both. Writes run-<i>.rgba (raw float32 RGBA)
// into output and prints one JSON report.
#include "async_render.h"
#include <fstream>
#include <iostream>
#include <json/json.h>
using namespace xpu2::interactive;
namespace rdl = scene_rdl2::rdl2;
using Clock = std::chrono::steady_clock;

static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    try {
        cpu::need(argc >= 9 && argc <= 15, "scene runtime ptx assets output threads repeats mode [cell spc window seed sampling override]");
        const auto began = Clock::now();
        const fs::path out(argv[5]);
        const unsigned threads = unsigned(std::stoul(argv[6])), repeats = unsigned(std::stoul(argv[7]));
        const int mode = std::stoi(argv[8]);
        cpu::need(threads >= 1 && threads <= 64 && repeats >= 1 && repeats <= 16 && mode >= 0 && mode <= 2, "Probe bounds");
        if (std::getenv("XPU2_SESSION_SPILL_MB")) texture::watchPressure(0);
        async::RenderOptions options;
        options.threads = threads;
        options.mode = mode;
        if (argc > 9) options.grid.cellSize = unsigned(std::stoul(argv[9]));
        if (argc > 10) options.grid.samplesPerChunk = unsigned(std::stoul(argv[10]));
        if (argc > 11) options.windowMs = std::stod(argv[11]);
        if (const char* v = std::getenv("XPU2_ASYNC_NO_RESERVE")) options.reserveFeederCore = std::string(v) != "1";
        if (const char* v = std::getenv("XPU2_ASYNC_TREE_THREADS")) options.treeThreads = unsigned(std::stoul(v));
        if (const char* v = std::getenv("XPU2_ASYNC_HOST_PASSES")) options.deviceChain = std::string(v) != "1";

        std::ifstream file(argv[3], std::ios::binary);
        cpu::need(bool(file), "PTX read");
        const std::string ptx((std::istreambuf_iterator<char>(file)), {});
        // Its material module sits beside it, in material/ (build_async.py).
        std::string materialPtx, chainPtx;
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "material" / "async_material.ptx", std::ios::binary})
            materialPtx.assign(std::istreambuf_iterator<char>(a), {});
        std::string mapsPtx, lightsPtx, lobesPtx, volumesPtx, dwaPtx, dwaLobesPtx, extrasPtx;
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "maps" / "async_maps.ptx", std::ios::binary})
            mapsPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "lights" / "async_lights.ptx", std::ios::binary})
            lightsPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "lobes" / "async_lobes.ptx", std::ios::binary})
            lobesPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "volumes" / "async_volumes.ptx", std::ios::binary})
            volumesPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "dwa" / "async_dwa.ptx", std::ios::binary})
            dwaPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "dwalobes" / "async_dwalobes.ptx", std::ios::binary})
            dwaLobesPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "extras" / "async_extras.ptx", std::ios::binary})
            extrasPtx.assign(std::istreambuf_iterator<char>(a), {});
        if (std::ifstream a{fs::path(argv[3]).parent_path() / "adaptive" / "async_adaptive.ptx", std::ios::binary})
            chainPtx.assign(std::istreambuf_iterator<char>(a), {});

        auto stamp = Clock::now();
        rdl::SceneContext source;
        source.setProxyModeEnabled(true);
        source.setDsoPath(argv[2]);
        rdl::readSceneFromFile(argv[1], source);
        // An optional second file applied over the scene, such as adaptive sampling settings.
        if (argc > 14 && std::string(argv[14]) != "-") rdl::readSceneFromFile(argv[14], source);
        const double readSeconds = since(stamp);

        stamp = Clock::now();
        CanonicalSession canonical(argv[2], {3301, 0, 0, 0});
        auto frame = canonical.prepare(source, out / "canonical", {3301, 1, 1, 1});
        // MoonRay's path tracer (PATH_TRACER.md).
        // MoonRay's sampler tables (a random walk's sequences) sit beside the GPU program in a build.
        cpu::sequenceDirectory() = fs::path(argv[3]).parent_path();
        auto plan = cpu::prepare(canonical, frame, {}, cpu::Limits{}, argv[4], cpu::Domain::SurfaceTextures, true, true);
        auto chosen = std::make_shared<cpu::Plan>(*plan);
        // "-" keeps the scene's own seed and the default sampling policy.
        if (argc > 12 && std::string(argv[12]) != "-") chosen->seed = unsigned(std::stoul(argv[12]));
        if (argc > 13 && std::string(argv[13]) != "-") chosen->sampling = unsigned(std::stoul(argv[13]));
        cpu::need(chosen->sampling <= 3, "Sampling policy");
        plan = chosen;
        cpu::need(canonical.acknowledge(frame) && canonical.publish(frame), "Canonical publish");
        const double planSeconds = since(stamp);

        stamp = Clock::now();
        async::CpuScene cpuScene(plan);
        const double embreeSeconds = since(stamp);

        stamp = Clock::now();
        std::unique_ptr<async::GpuLane> gpu;
        if (mode != 0) {
            async::GpuOptions gpuOptions;
            if (const char* v = std::getenv("XPU2_ASYNC_PROFILE_CHAIN")) gpuOptions.profileChain = std::string(v) == "1";
            gpuOptions.mapsPtx = mapsPtx;
            gpuOptions.lightsPtx = lightsPtx;
            gpuOptions.lobesPtx = lobesPtx;
            gpuOptions.volumesPtx = volumesPtx;
            gpuOptions.dwaPtx = dwaPtx;
            gpuOptions.dwaLobesPtx = dwaLobesPtx;
            gpuOptions.extrasPtx = extrasPtx;
            gpu = std::make_unique<async::GpuLane>(ptx, materialPtx, chainPtx, gpuOptions);
            gpu->prepare(plan);
        }
        const double gpuSeconds = since(stamp);

        Json::Value report;
        report["renderer"] = "asynchronous: cellular grid, CPU and GPU lanes, device film";
        report["width"] = plan->width;
        report["height"] = plan->height;
        report["spp"] = plan->spp;
        report["threads"] = threads;
        report["mode"] = mode;
        report["cell_size"] = options.grid.cellSize;
        report["samples_per_chunk"] = options.grid.samplesPerChunk;
        report["window_ms"] = options.windowMs;
        report["seed"] = plan->seed;
        report["sampling_policy"] = plan->sampling;
        report["adaptive"] = plan->adaptive;
        if (plan->adaptive) {
            report["min_adaptive_samples"] = plan->minSamples;
            report["max_adaptive_samples"] = plan->maxSamples;
            report["target_adaptive_error"] = plan->targetError;
        }
        report["read_seconds"] = readSeconds;
        report["plan_seconds"] = planSeconds;
        report["embree_seconds"] = embreeSeconds;
        report["gpu_prepare_seconds"] = gpuSeconds;
        report["prepare_seconds"] = since(began);
        if (gpu) {
            report["fine_tiles"] = Json::UInt64(gpu->fineTiles());
            report["resident_mip_bytes"] = Json::UInt64(gpu->residentMipBytes());
        }

        async::Renderer renderer(plan, cpuScene, gpu.get());
        for (unsigned i = 0; i < repeats; ++i) {
            const auto r = renderer.render(options);
            cpu::need(!r.failed, r.error.empty() ? "Render failed" : r.error);
            const fs::path raw = out / ("run-" + std::to_string(i) + ".rgba");
            std::ofstream o(raw, std::ios::binary);
            o.write(reinterpret_cast<const char*>(r.rgba.data()), r.rgba.size() * sizeof(float));
            cpu::need(bool(o), "Image write");
            std::ofstream oc(out / ("run-" + std::to_string(i) + ".counts"), std::ios::binary);
            oc.write(reinterpret_cast<const char*>(r.counts.data()), r.counts.size() * sizeof(r.counts[0]));
            if (!r.odd.empty()) {
                std::ofstream oo(out / ("run-" + std::to_string(i) + ".odd"), std::ios::binary);
                oo.write(reinterpret_cast<const char*>(r.odd.data()), r.odd.size() * sizeof(float));
            }
            Json::Value row;
            row["index"] = i;
            row["sampling_seconds"] = r.seconds;
            row["cpu_samples"] = Json::UInt64(r.cpuSamples);
            row["gpu_samples"] = Json::UInt64(r.gpuSamples);
            row["gpu_share"] = double(r.gpuSamples) / double(r.cpuSamples + r.gpuSamples);
            row["launches"] = Json::UInt64(r.launches);
            row["parked_run"] = Json::UInt64(r.parkedRun);
            row["cpu_steals"] = Json::UInt64(r.cpuSteals);
            row["cpu_moves"] = Json::UInt64(r.cpuMoves);
            row["gpu_stolen_chunks"] = Json::UInt64(r.gpuStolenChunks);
            row["fine_loaded"] = Json::UInt64(r.fineLoaded);
            row["fine_evicted"] = Json::UInt64(r.fineEvicted);
            row["gpu_items_per_second"] = r.gpuItemsPerSecond;
            row["cpu_samples_per_second"] = r.cpuSamplesPerSecond;
            row["cpu_workers"] = r.cpuWorkers;
            row["adaptive_passes"] = r.adaptivePasses;
            row["tree_updates"] = r.treeUpdates;
            row["mean_samples"] = r.meanSamples;
            row["tree_seconds"] = r.treeSeconds;
            row["error_seconds"] = r.errorSeconds;
            row["chain_stalls"] = r.chainStalls;
            Json::Value stages(Json::arrayValue);
            for (double v : r.chainStages) stages.append(v);
            row["chain_stages"] = stages;
            Json::Value renders(Json::arrayValue);
            for (double v : r.chainRenders) renders.append(v);
            row["chain_renders"] = renders;
            Json::Value passes(Json::arrayValue);
            for (const auto& t : r.passes) {
                Json::Value p(Json::arrayValue);
                p.append(Json::UInt64(t.samples));
                for (double v : {t.generated, t.launched, t.completed, t.asked, t.errors, t.trees}) p.append(v);
                passes.append(p);
            }
            row["passes"] = passes;
            Json::Value launches(Json::arrayValue);
            for (const auto& l : r.launchTrace) {
                Json::Value v(Json::arrayValue);
                v.append(l.queued);
                v.append(l.collected);
                v.append(l.deviceSeconds);
                v.append(l.claimSeconds);
                v.append(l.accountSeconds);
                v.append(l.launchSeconds);
                v.append(Json::UInt64(l.chunks));
                v.append(Json::UInt64(l.parked));
                launches.append(v);
            }
            row["launch_trace"] = launches;
            row["cpu_busy_seconds"] = r.cpuBusySeconds;
            row["cpu_slowest_chunk"] = r.cpuSlowestChunk;
            row["cpu_first_claim"] = r.cpuFirstClaim;
            row["cpu_last_claim"] = r.cpuLastClaim;
            Json::Value timeline;
            timeline["first_launch"] = r.timeline.firstLaunch;
            timeline["grid_empty"] = r.timeline.gridEmpty;
            timeline["cpu_done"] = r.timeline.cpuDone;
            timeline["last_gpu_completion"] = r.timeline.lastGpuCompletion;
            timeline["film_landed"] = r.timeline.filmLanded;
            row["timeline"] = timeline;
            row["raw_file"] = raw.string();
            report["runs"].append(row);
        }
        if (gpu) {
            report["peak_device_bytes"] = Json::UInt64(gpu->driver.peakDevice);
            report["peak_pinned_bytes"] = Json::UInt64(gpu->driver.peakPinned);
        }
        report["status"] = "passed";
        Json::StreamWriterBuilder b;
        b["indentation"] = "";
        std::cout << Json::writeString(b, report) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

// The asynchronous renderer's coordinator (spec §4, §8, §15 throughput mode).
//
// The CPU lane's workers claim their own chunks from the grid. This thread is
// the GPU lane's feeder and broker: it keeps two launches queued, sizes each
// from the GPU's measured rate (the donation forecast, CA-08), claims the GPU's
// own cells first and then the CPU's hottest (stealing), and near the end
// splits what is left so both devices finish together (the lookahead). It
// never waits on the GPU; it polls. At quiescence (CA-16) the two films merge
// and every pixel's sample count is audited (ACC-07).
#pragma once
#include "async_adaptive.h"
#include "async_cpu.h"
#include "async_gpu.h"
#include "device_suit.h"
#include <functional>
#include <limits>
#include <map>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>

namespace xpu2::interactive::async {

// Realtime's first sample in two passes, a pixel in each 2x2 of every tile and then the other three, so that a quarter of
// the samples showed the whole picture (filled between, then denoised at half size and enlarged by OptiX). Off since
// 2026-10-30 (asked): the filled picture flashed as scanlines before it settled, for little gain; each picture is the
// whole first pass. XPU2_SESSION_RT_SPARSE=1 brings it back.
inline bool sparseStart(const cpu::Plan& p) {
    static const bool on = [] { const char* v = std::getenv("XPU2_SESSION_RT_SPARSE"); return v && v[0] == '1'; }();
    return on && p.realtime;
}

struct RenderOptions {
    unsigned threads = 12;  // CPU lane workers
    int mode = 2;           // 0 CPU only, 1 GPU only, 2 both
    GridOptions grid{};
    double windowMs = 10;   // target duration of one GPU launch (throughput mode, THR-03)
    U32 firstChunks = 2048; // the first launches, before the GPU's rate is known
    // With both devices, one of the threads is the GPU feeder's own core (spec §4
    // reserves cores for the coordinator and feeder): it polls without sleeping.
    bool reserveFeederCore = true;
    unsigned treeThreads = ~0u;  // adaptive frames: pool threads besides the feeder for the error trees (~0u: one per region)
    // Adaptive frames with a GPU: make every pass on the device (the pass chain),
    // or, false, on the host between the GPU's passes.
    bool deviceChain = true;
    // Adaptive frames: cellular generations decide which tiles sample each pass (adaptive::Cellular; the passes are
    // made on the host), or, false, MoonRay's error trees.
    bool cellular = false;
    // Frames by sampling generations (cellular, uniform or adaptive) with a GPU: the CPU lane takes chunks too, by the
    // grid's rights of passage, and its samples join the GPU's errors (Adaptive::merged); false, the GPU takes them all.
    bool generationsCpu = true;
    // A preview session's watch over the frame (none: the frame runs to its end).
    struct Watch* watch = nullptr;
    // Partial shading: a GPU sample whose fine-detail tile is not resident finishes from the texture's next level up
    // and stays provisional (shown in the pictures, kept out of the film) until it is rendered again in full, by the
    // CPU lane or by the GPU once the tile has landed; false parks it until the tile lands, as before. Never in the
    // pass chain, which runs on the device alone.
    bool partial = true;
    // Important parts first: the pictures so far score the cells, the important ones stay their owner's and may run a
    // range ahead, the rest are anyone's (async_grid.h).
    bool important = true;
    // The GPU's cellular workers: shares of each launch, each in a cell of its own, moving by the cellular rules.
    unsigned gpuWorkers = 32;
};

// What a preview session asks of a frame, on the feeder's thread: whether to stop it (its samples are dropped,
// the frame is cancelled) or drain it (the work in hand finishes and the frame ends with what it has, every
// pixel the mean of the samples it got), and a picture of it now and then.
struct Watch {
    std::function<bool()> stop, drain;
    double previewSeconds = 0.1;
    // The picture so far (RGB, the film's pixel order) and the share of the frame's samples in it.
    std::function<void(const std::vector<float>& rgb, U32 width, U32 height, double fraction)> preview;
    // An output to show instead of the beauty (null: the beauty), asked at each picture; its picture comes as the
    // frame so far, its AOVs resolved as the frame's end resolves them.
    std::function<bool()> previewAov;
    std::function<void(const struct RenderResult& sofar, double fraction)> previewResult;
    // Checkpoints (Safe Denoising): the frame so far, as its end would make it (Cryptomatte too), every
    // checkpointSeconds (0 or no checkpoint: none).
    double checkpointSeconds = 0;
    std::function<void(struct RenderResult&& sofar)> checkpoint;
    // Each sampling generation as it goes out (the MoonRay Terminal): its number, its samples, the tiles still
    // sampling and all the image's tiles, the samples a pixel on average once it is done, and the seconds into the frame.
    std::function<void(unsigned generation, unsigned long long samples, size_t liveTiles, size_t tiles, double perPixel, double seconds)> generationLog;
};

// When things happened in a frame, in seconds from its start (TEL-01).
struct Timeline {
    double firstLaunch = -1, gridEmpty = -1, lastGpuCompletion = -1, cpuDone = -1, filmLanded = -1;
};

// One GPU launch: when it was queued and collected, in seconds from the frame's
// start, its chunks and parked samples, and its own duration on the device.
struct LaunchTrace {
    double queued = -1, collected = -1, deviceSeconds = 0;
    double claimSeconds = 0, accountSeconds = 0, launchSeconds = 0;  // the feeder's time to claim, account and queue it
    U64 chunks = 0, parked = 0;
};

// One adaptive pass: its samples, and when it went out, completed, and when the
// errors after it were asked for, arrived and became trees (-1: did not happen).
struct PassTrace {
    U64 samples = 0;
    double generated = -1, launched = -1, completed = -1, asked = -1, errors = -1, trees = -1;
};

struct RenderResult {
    std::vector<float> rgba;
    std::vector<U32> counts;  // samples each pixel received
    std::vector<float> odd;   // adaptive frames and aux outputs: odd-index sums over all samples (MoonRay's beauty aux)
    // The AOV schema's values a pixel (aov_core.h), resolved as MoonRay's snapshots resolve them: averages over the
    // pixel's samples, sums, minima, maxima and the closest as they are; and the seconds each pixel took.
    std::vector<float> aov;
    std::vector<float> heat;
    // Cryptomatte's fragments, finalized (aov_crypto.h), when an output asks for them.
    std::shared_ptr<aov::CryptoBuffer> crypto;
    // The display filter outputs' pixels by output name (aov_display.h), three floats a pixel, rows from the bottom.
    std::map<std::string, std::vector<float>> displayFilters;
    U64 cpuSamples = 0, gpuSamples = 0, parkedRun = 0, launches = 0, cpuSteals = 0, cpuMoves = 0,
        gpuStolenChunks = 0, fineLoaded = 0, fineEvicted = 0;
    double seconds = 0, gpuItemsPerSecond = 0, cpuSamplesPerSecond = 0;
    unsigned cpuWorkers = 0;
    Timeline timeline;
    // Adaptive frames: passes generated, tree updates, and samples a pixel on average.
    U32 adaptivePasses = 0, treeUpdates = 0;
    U64 deferred = 0;  // samples the GPU deferred to the CPU lane (stand-ins)
    // Partial shading: provisional samples the GPU made (one again each time one rendered again still read a coarser
    // level), and those rendered again in full on the GPU and on the CPU.
    U64 provisional = 0, refinedGpu = 0, refinedCpu = 0;
    // The GPU profile (XPU2_PROFILE): clock ticks in each stage (ProfileStage), summed over the GPU's threads.
    U64 gpuProfile[ProfileStages] = {};
    // The cells by suitability of a grid's first frame on both devices (device_suit.h): cells, those the CPU took
    // (of them, those the GPU would mostly defer), the lanes' rate ratio used, the probe's time.
    U32 seededCells = 0, seededCpuCells = 0, seededDeferCells = 0;
    double seededRatio = 0, seedSeconds = 0;
    double meanSamples = 0;
    double treeSeconds = 0, errorSeconds = 0;  // rebuilding the trees; from asking for errors to having them
    U32 chainStalls = 0;  // times the GPU's pass chain waited for texture tiles
    double chainStages[4] = {};  // profiled chains: seconds in errors, trees, masks, renders
    std::vector<double> chainRenders;  // profiled chains: each step's render
    std::vector<PassTrace> passes;
    std::vector<LaunchTrace> launchTrace;
    double cpuBusySeconds = 0, cpuSlowestChunk = 0, cpuFirstClaim = 0, cpuLastClaim = 0;
    bool failed = false;
    bool cancelled = false, drained = false;  // stopped or drained by a Watch
    std::string error;
};

// The kit's low-latency rendering (MoonRay Source Overrides, on by default): it starts the renderer below Modo's
// priority and gives it one processor thread fewer than the machine has. XPU2's threads then keep to their process's
// priority, none raised above it (the GPU's feeder and the error trees' pool give way to Modo, the GPU fed a little less
// promptly), and the CPU lane leaves a thread free besides the feeder's whatever thread count it is given. Known from
// the process's priority class, which the kit lowers as the renderer starts; XPU2_SESSION_LOW_LATENCY=0 or 1 says so.
inline bool lowLatency() {
    if (const char* v = std::getenv("XPU2_SESSION_LOW_LATENCY")) return v[0] == '1';
    const DWORD c = GetPriorityClass(GetCurrentProcess());
    return c == BELOW_NORMAL_PRIORITY_CLASS || c == IDLE_PRIORITY_CLASS;
}

// Persistent threads for the adaptive error trees: run(n, f) calls f(0) .. f(n - 1)
// across them and the calling thread, and returns once every call has.
class Pool {
public:
    explicit Pool(unsigned threads) {
        for (unsigned i = 0; i < threads; ++i) threads_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& t : threads_) t.join();
    }
    unsigned size() const { return unsigned(threads_.size()); }
    void run(unsigned n, const std::function<void(unsigned)>& f) {
        {
            // A worker still inside the last run would take this one's indices.
            std::unique_lock<std::mutex> lock(mutex_);
            left_.wait(lock, [&] { return active_ == 0; });
            task_ = &f;
            count_ = n;
            next_.store(0);
            ++generation_;
        }
        wake_.notify_all();
        work(f, n);
        std::unique_lock<std::mutex> lock(mutex_);
        left_.wait(lock, [&] { return active_ == 0 && finished_ == n; });
        finished_ = 0;
        task_ = nullptr;
    }

private:
    void work(const std::function<void(unsigned)>& f, unsigned n) {
        unsigned done = 0;
        for (unsigned i; (i = next_.fetch_add(1)) < n; ++done) f(i);
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ += done;
        left_.notify_all();
    }
    void loop() {
        SetThreadPriority(GetCurrentThread(), lowLatency() ? THREAD_PRIORITY_NORMAL : THREAD_PRIORITY_ABOVE_NORMAL);
        unsigned seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* task;
            unsigned n;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || (task_ && generation_ != seen); });
                if (stop_) return;
                seen = generation_;
                task = task_;
                n = count_;
                ++active_;
            }
            work(*task, n);
            std::lock_guard<std::mutex> lock(mutex_);
            --active_;
            left_.notify_all();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_, left_;
    const std::function<void(unsigned)>* task_ = nullptr;
    unsigned count_ = 0, generation_ = 0, active_ = 0, finished_ = 0;
    std::atomic<unsigned> next_{0};
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

class Renderer {
public:
    Renderer(cpu::PlanView plan, const CpuScene& cpuScene, GpuLane* gpu)
        : plan_(std::move(plan)), cpuScene_(cpuScene), gpu_(gpu) {
        timeBeginPeriod(1);  // millisecond sleeps for the feeder's polling
    }
    ~Renderer() { timeEndPeriod(1); }

    // An adaptive frame's live tiles, the pass being sampled (cellular frames: its generation's cells). Empty
    // between frames and in uniform frames.
    void liveTiles(std::vector<U32>& tiles) const {
        std::lock_guard<std::mutex> lock(activityMutex_);
        tiles = live_;
    }
    // What the frame's workers are on, for a host's outlines of the work, from any thread: each CPU worker's tile
    // (8 pixels a side), and the work grid's cells that the GPU's queued launches hold. Empty between frames.
    void activity(std::vector<U32>& cpuTiles, std::vector<U32>& gpuCells) const {
        cpuTiles.clear();
        gpuCells.clear();
        std::lock_guard<std::mutex> lock(activityMutex_);
        if (activeLane_)
            for (unsigned w = 0; w < activeLane_->threads(); ++w)
                if (const U32 t = activeLane_->activeTile(w); t != ~0u) cpuTiles.push_back(t);
        for (const auto& flight : gpuFlights_) gpuCells.insert(gpuCells.end(), flight.begin(), flight.end());
        std::sort(gpuCells.begin(), gpuCells.end());
        gpuCells.erase(std::unique(gpuCells.begin(), gpuCells.end()), gpuCells.end());
    }
    U32 cellSize() const { return cellSize_; }
    // The GPU lane's samples a second over the CPU lane's, measured by the last frame on both (0: none yet); a new
    // renderer of the session's next scene starts from its predecessor's.
    double deviceRatio() const { return deviceRatio_; }
    void deviceRatio(double r) { deviceRatio_ = r; }

    RenderResult render(const RenderOptions& o) {
        using Clock = std::chrono::steady_clock;
        RenderResult result;
        const auto config = mixed::config(plan_);
        const bool adaptive = plan_->adaptive;
        // In an adaptive frame with a GPU, the CPU rebuilds the error trees and the
        // GPU takes every sample: the chain of passes, each waiting on the trees,
        // bounds the frame, and a CPU chunk would sit on that chain.
        // A scene with GPU stand-ins needs the CPU lane for the samples the GPU defers, whatever the mode.
        const bool standIns = o.mode != 0 && gpu_ && gpu_->standIns();
        // Sampling generations (cellular rules) make each pass on the host: the CPU lane takes chunks the grid's rights of
        // passage let it finish before the GPU would, and its samples join the GPU's errors (Adaptive::merged).
        // (The kit's render settings may leave the CPU out: plan_->generationsCpu.)
        // Realtime leaves the CPU lane to what the GPU defers: cells the CPU owned filled seconds after the rest.
        const bool cpuInGenerations = adaptive && (o.cellular || plan_->sampleBudget != 0) && plan_->generationsCpu && o.generationsCpu &&
                                      !plan_->realtime;
        const bool useGpu = o.mode != 0, useCpu = standIns || (o.mode != 1 && !(adaptive && useGpu && !cpuInGenerations));
        cpu::need(!useGpu || gpu_, "GPU lane required");
        GridOptions gridOptions = o.grid;
        gridOptions.mode = useCpu ? (useGpu ? 2 : 0) : 1;
        // Ownership carried over from earlier frames of the same grid (GRD-03).
        const U32 cells = ((plan_->width + gridOptions.cellSize - 1) / gridOptions.cellSize) *
                          ((plan_->height + gridOptions.cellSize - 1) / gridOptions.cellSize);
        // A grid's first frame on both devices: its cells start on the device suited to what they see (device_suit.h).
        const std::vector<Device>* owners = ownership_.owners(cells);
        suit::Seeded seeded;
        if (!owners && gridOptions.mode == 2) {
            seeded = suit::seed(config, *plan_, cpuScene_, gridOptions.cellSize, deviceRatio_);
            if (seeded.owners.size() == cells) {
                owners = &seeded.owners;
                result.seededCells = seeded.cells;
                result.seededCpuCells = seeded.cpuCells;
                result.seededDeferCells = seeded.deferCells;
                result.seededRatio = seeded.ratio;
                result.seedSeconds = seeded.seconds;
            }
        }
        Grid grid(plan_->width, plan_->height, plan_->spp, gridOptions, owners, adaptive, plan_->viewport);
        // The indexer's probabilities, kept for the later frames of the same grid.
        if (seeded.gpuProbability.size() == cells) suitability_ = seeded.gpuProbability;
        grid.setSuitability(suitability_);
        HostFilm film(plan_->width, plan_->height);
        film.initAovs(cpuScene_.aovTables, plan_->aovs && plan_->aovs->heatMap);
        film.initCrypto(plan_->aovs.get());
        RetryQueue retries;
        RefineQueue refines;
        const U32 itemsPerChunk = gridOptions.samplesPerChunk * TilePixels;
        const bool chainFrame = adaptive && useGpu && o.deviceChain && !o.cellular && !plan_->sampleBudget && !standIns && gpu_->hasChain();
        const bool partial = o.partial && useGpu && !chainFrame;
        if (partial) film.initSettled();
        if (useGpu) gpu_->begin(gridOptions.samplesPerChunk, adaptive, partial);
        Adaptive state(*plan_, gridOptions.samplesPerChunk, o.cellular);
        state.cpuSamples = adaptive && useGpu && useCpu;
        state.onGeneration = [this](const std::vector<U32>& tiles) {
            std::lock_guard<std::mutex> lock(activityMutex_);
            live_ = tiles;
        };
        if (o.watch && o.watch->generationLog) {
            const double pixels = double(plan_->width) * plan_->height;
            const auto began = Clock::now();
            Watch* const told = o.watch;
            state.onGenerationLog = [told, pixels, began, &state](U32 generation, U64 samples, size_t live, double spent) {
                told->generationLog(generation + 1, samples, live, state.tiles, pixels > 0 ? spent / pixels : 0.0,
                                    std::chrono::duration<double>(Clock::now() - began).count());
            };
        }
        if (adaptive) {
            const unsigned trees = o.treeThreads != ~0u ? o.treeThreads : std::max(1u, std::min(o.threads, 16u) - 1);
            if (!pool_ || pool_->size() != trees) pool_ = std::make_unique<Pool>(trees);
            state.start = Clock::now();
            if (chainFrame) state.startChain(*gpu_, tileOrder(gridOptions.cellSize));
            else state.generateReady(grid, useGpu ? gpu_ : nullptr);
        }

        // The feeder runs above the CPU lane's workers, so a busy CPU never starves
        // the GPU of work (spec §4: the GPU feeder runs at high priority); with low latency, at its process's own.
        const bool modest = lowLatency();
        struct Priority {
            int old = GetThreadPriority(GetCurrentThread());
            explicit Priority(bool modest) { if (!modest) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); }
            ~Priority() { SetThreadPriority(GetCurrentThread(), old); }
        } priority(modest);
        // With a reserved core the feeder is pinned to the last logical processor
        // and the workers to the others (spec §4, CPU-01): a spinning high-priority
        // feeder must never sit on a worker's core.
        DWORD_PTR process = 0, system = 0, workerMask = 0;
        struct Pin {
            DWORD_PTR old = 0;
            void to(DWORD_PTR mask) { old = SetThreadAffinityMask(GetCurrentThread(), mask); }
            ~Pin() { if (old) SetThreadAffinityMask(GetCurrentThread(), old); }
        } pin;
        unsigned workers = !useCpu ? 0 : (useGpu && o.reserveFeederCore && o.threads > 1 ? o.threads - 1 : o.threads);
        // Low latency: a processor thread left free for Modo besides the feeder's.
        if (workers && modest) workers = std::min(workers, std::max(1u, std::thread::hardware_concurrency() - (useGpu ? 2u : 1u)));
        result.cpuWorkers = workers;
        if (useGpu && useCpu && o.reserveFeederCore && GetProcessAffinityMask(GetCurrentProcess(), &process, &system) &&
            process && (process & (process - 1))) {
            DWORD_PTR feeder = DWORD_PTR(1) << (8 * sizeof(DWORD_PTR) - 1);  // the highest allowed processor
            while (!(process & feeder)) feeder >>= 1;
            workerMask = process & ~feeder;
            pin.to(feeder);
        }
        if (!cpu_ || cpu_->threads() != workers || cpuAffinity_ != workerMask) {
            cpu_.reset();
            cpu_ = std::make_unique<CpuLane>(cpuScene_, workers, workerMask);
            cpuAffinity_ = workerMask;
        }
        CpuLane& cpuLane = *cpu_;
        const auto start = Clock::now();
        auto at = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
        cpuLane.begin(config, grid, film, retries, adaptive ? &state.tables : nullptr, partial ? &refines : nullptr);
        {
            std::lock_guard<std::mutex> lock(activityMutex_);
            activeLane_ = &cpuLane;
            gpuFlights_.clear();
            cellSize_ = gridOptions.cellSize;
        }
        // Outlines end with the frame, however it ends.
        struct Quiet {
            Renderer& r;
            ~Quiet() {
                std::lock_guard<std::mutex> lock(r.activityMutex_);
                r.activeLane_ = nullptr;
                r.gpuFlights_.clear();
                r.live_.clear();
            }
        } quiet{*this};
        // A failure thrown out of the feeder (a tile that cannot be read, the GPU's): the workers stop before the frame's grid and film go.
        struct Halt {
            CpuLane& lane;
            int uncaught = std::uncaught_exceptions();
            ~Halt() {
                if (std::uncaught_exceptions() > uncaught) {
                    lane.cancel();
                    lane.waitParked();
                }
            }
        } halt{cpuLane};
        // The queued launches' cells: one list a launch, oldest first, as long as the GPU's queue.
        auto flights = [&](const Chunk* chunks, U32 count) {
            std::vector<U32> cells;
            for (U32 i = 0; i < count; ++i) cells.push_back(chunks[i].cell);
            std::sort(cells.begin(), cells.end());
            cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
            std::lock_guard<std::mutex> lock(activityMutex_);
            gpuFlights_.push_back(std::move(cells));
        };
        auto landed = [&] {
            std::lock_guard<std::mutex> lock(activityMutex_);
            while (gpuFlights_.size() > gpu_->inFlight()) gpuFlights_.pop_front();
            while (gpuFlights_.size() < gpu_->inFlight()) gpuFlights_.emplace_back();  // the pass chain's flights
        };
        Watch* const watch = o.watch;
        bool cancelled = false, drained = false;
        // Progressive pictures: the GPU's film is copied behind its queued launches, then merged with the host's.
        enum { PreviewIdle, PreviewCopying } previewState = PreviewIdle;
        // A drain with provisional samples about: no chunk is handed out any more (the grid closes), the samples are
        // rendered in full first, and then the frame ends with what it has.
        bool draining = false;
        // Without a watch they are made for important parts first alone.
        const bool shown = watch && watch->preview;
        double nextPreview = shown ? watch->previewSeconds : o.important && !adaptive ? 0.2 : -1;
        std::vector<float> previewRgb;
        std::vector<unsigned char> covered;
        double nextScore = 0;  // important parts first: the pictures score the cells at most once a second
        // Checkpoints: the next one's time (-1: none), and whether the copy under way is for one.
        double nextCheckpoint = watch && watch->checkpoint && watch->checkpointSeconds > 0 ? watch->checkpointSeconds : -1;
        bool checkpointNow = false;
        const double plannedSamples = double(plan_->width) * plan_->height * plan_->spp;
        bool previewOfAov = false;
        auto preview = [&] {
            const bool checkpointDue = nextCheckpoint >= 0 && at() >= nextCheckpoint && !draining;
            if (previewState == PreviewIdle && !checkpointDue && (nextPreview < 0 || at() < nextPreview)) return;
            if (previewState == PreviewIdle) {
                previewOfAov = shown && watch->previewAov && watch->previewResult && watch->previewAov();
                checkpointNow = checkpointDue;
            }
            if (previewState == PreviewIdle && useGpu) {
                gpu_->downloadPreview(previewOfAov || checkpointNow);
                previewState = PreviewCopying;
                return;
            }
            if (previewState == PreviewCopying && !gpu_->previewReady()) return;
            if (previewOfAov || checkpointNow) {
                RenderResult sofar;
                const U64 seen = snapshotResult(film, sofar, adaptive, useGpu ? gpu_ : nullptr, checkpointNow);
                if (previewOfAov) watch->previewResult(sofar, std::min(1.0, double(seen) / plannedSamples));
                if (checkpointNow) {
                    watch->checkpoint(std::move(sofar));
                    nextCheckpoint = at() + watch->checkpointSeconds;
                    checkpointNow = false;
                }
                if (previewOfAov) {
                    previewState = PreviewIdle;
                    // An output's pictures are slower to make: at most two a second, but realtime's (denoised) as they come.
                    nextPreview = at() + (plan_->realtime ? watch->previewSeconds : std::max(watch->previewSeconds, 0.5));
                    return;
                }
            }
            previewRgb.resize(size_t(plan_->width) * plan_->height * 3);
            covered.resize(size_t(plan_->width) * plan_->height);
            const U64 seen = film.preview(previewRgb.data(), useGpu ? gpu_->sums() : nullptr, useGpu ? gpu_->counts() : nullptr,
                                          partial ? gpu_->provSums() : nullptr, partial ? gpu_->provCounts() : nullptr, covered.data());
            if (o.important && !adaptive && !draining && at() >= nextScore) {
                grid.setScores(cellScores(previewRgb, covered, gridOptions.cellSize));
                nextScore = at() + 1.0;
            }
            previewState = PreviewIdle;
            if (!shown) {
                nextPreview = at() + 1.0;
                return;
            }
            // An adaptive frame stops short of its most samples: its share is its passes done, where the host knows them.
            double fraction = std::min(1.0, double(seen) / plannedSamples);
            if (adaptive && !state.chain && !state.passes.empty())
                fraction = std::max(fraction, double(state.retired) / double(state.passes.size()));
            // A frame by sampling generations runs for uniform's time: its share of it, once the seed has priced it.
            if (plan_->sampleBudget && plan_->budgetBySamples) fraction = std::min(1.0, double(seen) / double(plan_->sampleBudget));
            else if (plan_->sampleBudget && state.budgetSeconds > 0) fraction = std::min(1.0, state.at() / state.budgetSeconds);
            fillUncovered(previewRgb, covered, plan_->width, plan_->height);
            watch->preview(previewRgb, plan_->width, plan_->height, fraction);
            nextPreview = at() + watch->previewSeconds;
        };

        double gpuRate = 0;  // samples a second, timed on the device, smoothed
        U64 queuedSamples = 0, stalls = 0, lastParked = 0;
        // The GPU's cellular workers, each in its cell (async_grid.h choose): a launch's share each.
        std::vector<U32> gpuCells(useGpu ? std::max(1u, o.gpuWorkers) : 0u, ~0u);
        for (U32 w = 0; w < gpuCells.size(); ++w) gpuCells[w] = grid.seed(Device::Gpu, w, U32(gpuCells.size()));
        std::vector<Chunk> claim;
        std::vector<U64> cellSamples(grid.cellCount(), 0);  // the GPU's claims, for the ownership accounting
        // Partial shading: the provisional samples a launch renders again, and a guard against a cache too small for
        // them (the queue no lower over many launches that took them whether or not their tiles had landed).
        std::vector<Refine> refineBatch(partial ? GpuLane::RefineInCapacity : 0);
        std::vector<Refine> made;
        size_t refineLow = ~size_t(0);
        U32 refineStalls = 0;
        std::string failure;
        while (failure.empty()) {
            if (watch) {
                if (watch->stop && watch->stop()) {
                    cancelled = true;
                    break;
                }
                if (!draining && watch->drain && watch->drain()) {
                    if (!partial) {
                        drained = true;
                        break;
                    }
                    draining = true;
                    grid.close();
                }
            }
            preview();
            if (useGpu) {
                Completion c;
                while (gpu_->poll(c)) {
                    // The samples the launch deferred: the CPU lane renders them whole, before new chunks.
                    {
                        std::vector<Parked> deferred;
                        gpu_->takeDeferred(deferred);
                        if (!deferred.empty()) {
                            std::vector<Retry> moved(deferred.size());
                            const U32 tilesX = (plan_->width + TileSize - 1) / TileSize;
                            for (size_t k = 0; k < deferred.size(); ++k) {
                                moved[k] = Retry{deferred[k].pixel, deferred[k].index, deferred[k].pass};
                                const U32 x = deferred[k].pixel % plan_->width, y = deferred[k].pixel / plan_->width;
                                grid.deferredToCpu((y / TileSize) * tilesX + x / TileSize);
                            }
                            retries.push(moved.data(), moved.size());
                            result.deferred += deferred.size();
                        }
                    }
                    // The provisional samples it made: for the CPU lane, or the GPU once their tiles have landed.
                    if (partial) {
                        made.clear();
                        gpu_->takeRefines(made);
                        refines.push(made.data(), made.size());
                        result.provisional += made.size();
                        result.refinedGpu += c.refinesRun;
                    }
                    queuedSamples -= std::min(queuedSamples, c.samples);
                    // Every launch's rate, kept across frames for realtime's launch sizes (launchSize).
                    if (c.deviceSeconds > 0 && c.samples > 0) gpu_->noteRate(double(c.samples + c.parkedRun + c.refinesRun) / c.deviceSeconds);
                    // Launches too small to keep the GPU full understate its rate.
                    if (c.deviceSeconds > 0 && c.samples >= U64(gpu_->launchThreads())) {
                        const double rate = double(c.samples + c.parkedRun + c.refinesRun) / c.deviceSeconds;
                        gpuRate = gpuRate ? 0.7 * gpuRate + 0.3 * rate : rate;
                    }
                    if (c.errors) failure = "GPU sample failure; whole epoch must use fallback";
                    result.timeline.lastGpuCompletion = at();
                    if (result.launches < result.launchTrace.size()) {
                        auto& l = result.launchTrace[result.launches];
                        l.collected = at();
                        l.deviceSeconds = c.deviceSeconds;
                        l.parked = c.parkedRun;
                    }
                    ++result.launches;
                    result.parkedRun += c.parkedRun;
                }
                landed();
                // The rights of passage weigh the lanes' rates (async_grid.h passage).
                {
                    const double elapsed = at();
                    const unsigned workers = std::max(1u, cpuLane.threads());
                    grid.setRates(gpuRate, elapsed > 0.05 && cpuLane.threads() ? double(cpuLane.counters.samples.load()) / elapsed / workers : 0.0,
                                  workers, o.windowMs / 1000.0);
                }
                while (failure.empty() && gpu_->ready()) {
                    const double claimStart = at();
                    // Provisional samples whose tiles have landed go back to the GPU first; with no chunk left and
                    // nothing in flight, any (their tiles are asked for again), so the frame never waits on the CPU.
                    U32 refineCount = 0;
                    if (partial && refines.size()) {
                        gpu_->landTiles();
                        const bool any = grid.empty() && !gpu_->inFlight();
                        refineCount = U32(refines.take([&](U32 tile) { return any || gpu_->tileResident(tile); }, refineBatch.data(),
                                                       refineBatch.size(), size_t(4) * refineBatch.size()));
                        if (any && refineCount) {
                            const size_t left = refines.size() + refineCount;
                            if (left < refineLow) {
                                refineLow = left;
                                refineStalls = 0;
                            } else if (++refineStalls > 512) {
                                failure = "Fine-detail cache cannot hold the provisional samples' working set; use fallback";
                            }
                        }
                    }
                    // Realtime's sparse passes (a pixel in each 2x2, then the other three): a chunk holds 16 or 48 of a tile's
                    // 64 pixel samples, which the launch is sized by.
                    const double realtimeItems = sparseStart(*plan_) && adaptive && state.retired < 2 ? (state.retired == 0 ? 16.0 : 48.0) : 0.0;
                    const U32 want = std::min(launchSize(o, grid, gpuRate, cpuLane, queuedSamples, start, realtimeItems), gpu_->chunkRoom(refineCount));
                    claim.resize(want);
                    U32 got = 0;
                    if (want) {
                        // The GPU's cellular workers take the launch's shares, each from its cell, moving by the cellular
                        // rules where that cell has nothing more it may take now.
                        const U32 share = (want + U32(gpuCells.size()) - 1) / U32(gpuCells.size());
                        for (U32 w = 0; w < gpuCells.size() && got < want; ++w)
                            for (U32 need = std::min(share, want - got), moves = 0; need && moves < 8; ++moves) {
                                const U32 cell = grid.choose(Device::Gpu, gpuCells[w], w);
                                if (cell == ~0u) break;
                                gpuCells[w] = cell;
                                const U32 n = grid.claim(cell, need, claim.data() + got, Device::Gpu);
                                if (n && grid.owner(cell) != Device::Gpu) result.gpuStolenChunks += n;
                                got += n;
                                need -= n;
                            }
                        if (got) grid.gpuClaimed();
                    }
                    const double accountStart = at();
                    // Each chunk's samples: its tile's pixels in the image (an adaptive
                    // pass's mask holds only those) times its sample count.
                    U64 launchSamples = 0;
                    for (U32 i = 0; i < got; ++i) {
                        const Chunk& k = claim[i];
                        const U64 pixels = adaptive ? U64(__builtin_popcountll(state.tables.masks[chunkPass(k) % PassSlots][k.tile]))
                                                    : grid.tilePixels(k.tile);
                        cellSamples[k.cell] += pixels * k.count;
                        launchSamples += pixels * k.count;
                    }
                    for (U32 c = 0; c < cellSamples.size(); ++c)
                        if (cellSamples[c]) {
                            grid.rendered(c, Device::Gpu, cellSamples[c]);
                            cellSamples[c] = 0;
                        }
                    const double launchStart = at();
                    if (!gpu_->launch(claim.data(), got, refineBatch.data(), refineCount, launchSamples)) break;
                    flights(claim.data(), got);
                    if (result.timeline.firstLaunch < 0) result.timeline.firstLaunch = at();
                    result.launchTrace.push_back({at(), -1, 0, accountStart - claimStart, launchStart - accountStart,
                                                  at() - launchStart, got, 0});
                    if (adaptive && got && grid.empty()) state.trace[state.generated - 1].launched = state.at();
                    queuedSamples += launchSamples;
                    // A parked queue that does not shrink across many launches means
                    // the cache cannot hold what the samples need.
                    if (!got && !refineCount && gpu_->parkedOutstanding() >= lastParked && ++stalls > 256)
                        failure = "Fine-detail cache cannot hold the working set; use fallback";
                    lastParked = gpu_->parkedOutstanding();
                }
            }
            if (cpuLane.counters.failures.load()) {
                failure = cpuLane.failure();
                if (failure.empty()) failure = "CPU sample failure; whole epoch must use fallback";
            }
            if (adaptive && !draining) {
                const U64 before = useGpu ? gpu_->launchCount() : 0;
                state.advance(grid, film, useGpu ? gpu_ : nullptr, *pool_);
                if (useGpu) landed();
                if (state.chain && gpu_->launchCount() > before) {  // the pass chain went out: one flight
                    if (result.timeline.firstLaunch < 0) result.timeline.firstLaunch = at();
                    result.launchTrace.push_back({at(), -1, 0, 0, 0, 0, 0, 0});
                }
            }
            const bool empty = grid.empty() && (draining || !adaptive || state.finished(useGpu ? gpu_ : nullptr));
            if (empty && result.timeline.gridEmpty < 0) result.timeline.gridEmpty = at();
            const bool cpuQuiet = cpuLane.idle() == cpuLane.threads() && !retries.size() && !refines.size();
            if (empty && cpuQuiet && result.timeline.cpuDone < 0) result.timeline.cpuDone = at();
            const bool gpuQuiet = !useGpu || (gpu_->inFlight() == 0 && gpu_->parkedOutstanding() == 0);
            if (empty && gpuQuiet && cpuQuiet) {
                drained = draining;
                break;
            }
            // The feeder polls: on its own core it never sleeps; otherwise a short sleep.
            if (useGpu && (o.reserveFeederCore || !useCpu)) std::this_thread::yield();
            else Sleep(1);
        }
        // Stopped or drained: no more work is handed out; the workers' chunks in hand and the GPU's queued
        // launches finish (their parked samples are dropped).
        if (!failure.empty() || cancelled || drained) cpuLane.cancel();
        cpuLane.finish();
        cpuLane.waitParked();  // the frame's grid and film outlive every worker's use
        if (!failure.empty()) {
            result.failed = true;
            result.error = failure;
            return result;
        }
        if ((cancelled || drained) && useGpu) {
            Completion c;
            while (gpu_->inFlight())
                if (!gpu_->poll(c)) std::this_thread::yield();
        }
        if (cancelled) {
            result.cancelled = true;
            result.seconds = at();
            return result;
        }
        result.drained = drained;
        if (useGpu) {
            gpu_->downloadFilm();
            while (!gpu_->filmReady()) std::this_thread::yield();
            result.timeline.filmLanded = at();
        }
        result.seconds = std::chrono::duration<double>(Clock::now() - start).count();

        result.refinedCpu = cpuLane.counters.refined.load();
        // Merge and audit: every pixel has exactly the samples planned for it, spp
        // in a uniform frame (ACC-07), and no provisional sample is left.
        const size_t pixels = size_t(plan_->width) * plan_->height;
        result.rgba.resize(pixels * 4);
        result.counts.resize(pixels);
        const bool aux = plan_->aovs && (plan_->aovs->beautyAux || plan_->aovs->alphaAux);
        if (adaptive || aux) result.odd.resize(pixels * 4);
        const auto& hs = film.sums();
        const auto& hc = film.counts();
        U64 planned = 0;
        // A camera that may make no ray for a sample (camera_core.h's CameraResult::None: a fisheye's corners, a bake
        // camera's texels no face covers) leaves a pixel fewer samples than planned, none at all where it never makes
        // one; pixels outside the sub viewport take none.
        const bool dropsRays = plan_->camera.type == surface::CameraFisheye || plan_->camera.type == surface::CameraBake;
        const int* view = plan_->viewport;
        for (size_t i = 0; i < pixels; ++i) {
            double provisional[4] = {0, 0, 0, 0};
            const U32 left = partial ? film.provisional(i, gpu_->provSums(), gpu_->provCounts(), provisional) : 0u;
            if (left && !drained) {
                result.failed = true;
                result.error = "Provisional samples left at pixel " + std::to_string(i) + ": " + std::to_string(left);
                return result;
            }
            const U32 gc = (useGpu ? gpu_->counts()[i] : 0) + left, count = hc[i] + gc;
            const int px = int(i % plan_->width), py = int(i / plan_->width);
            const bool inView = px >= view[0] && py >= view[1] && px < view[2] && py < view[3];
            const U32 expected = !inView ? 0u : !adaptive ? (plan_->pixelSamples.empty() ? plan_->spp : plan_->pixelSamples[i])
                                 : state.chain ? gpu_->chainPlanned()[i] : state.planned[i];
            result.counts[i] = count;
            planned += expected;
            // A drained frame has what it got: the mean of its samples, black where none landed.
            if (drained) {
                for (int c = 0; c < 4; ++c)
                    result.rgba[i * 4 + c] = count ? float((hs[i * 4 + c] + (useGpu ? gpu_->sums()[i * 4 + c] : 0.0) + provisional[c]) / count) : 0.0f;
                result.cpuSamples += hc[i];
                result.gpuSamples += gc;
                continue;
            }
            if (dropsRays ? count > expected : count != expected || (inView && !count)) {
                result.failed = true;
                result.error = "Sample audit failure at pixel " + std::to_string(i) + ": " + std::to_string(count);
                return result;
            }
            for (int c = 0; c < 4; ++c) {
                const double sum = hs[i * 4 + c] + (useGpu ? gpu_->sums()[i * 4 + c] : 0.0);
                result.rgba[i * 4 + c] = count ? float(sum / count) : 0.0f;
                if (adaptive || aux)
                    result.odd[i * 4 + c] = count ? float((film.odd()[i * 4 + c] + (useGpu ? gpu_->odd()[i * 4 + c] : 0.0)) / count) : 0.0f;
            }
            result.cpuSamples += hc[i];
            result.gpuSamples += gc;
        }
        resolveAovs(film, result, adaptive, useGpu ? gpu_ : nullptr);
        result.cpuSteals = cpuLane.counters.steals.load();
        result.cpuBusySeconds = cpuLane.counters.busyMicros.load() * 1e-6;
        result.cpuSlowestChunk = cpuLane.counters.slowestChunkMicros.load() * 1e-6;
        result.cpuFirstClaim = cpuLane.counters.firstClaimMicros.load() == ~0ull ? -1 : cpuLane.counters.firstClaimMicros.load() * 1e-6;
        result.cpuLastClaim = cpuLane.counters.lastClaimMicros.load() * 1e-6;
        result.cpuMoves = cpuLane.counters.moves.load();
        result.gpuItemsPerSecond = gpuRate;
        result.cpuSamplesPerSecond = result.seconds > 0 ? double(result.cpuSamples) / result.seconds : 0;
        // The lanes' rate ratio for the next grid's seeding.
        if (useGpu && useCpu && !drained && gpuRate > 0 && result.cpuSamplesPerSecond > 0) deviceRatio_ = gpuRate / result.cpuSamplesPerSecond;
        if (useGpu) {
            std::copy(gpu_->profile(), gpu_->profile() + ProfileStages, result.gpuProfile);
            result.fineLoaded = gpu_->fineLoaded();
            result.fineEvicted = gpu_->fineEvicted();
        }
        result.adaptivePasses = state.generated;
        result.treeUpdates = state.treeUpdates;
        if (state.chain) {
            const auto& s = gpu_->chainState();
            result.adaptivePasses = state.uniformEnd + (s.chainPasses ? s.chainPasses - 1 : 0);
            result.treeUpdates = s.treeUpdates;
            result.chainStalls = state.stalls;
            std::copy(gpu_->chainStages(), gpu_->chainStages() + 4, result.chainStages);
            result.chainRenders = gpu_->chainRenders();
        }
        result.treeSeconds = state.treeSeconds;
        result.passes = state.trace;
        result.errorSeconds = state.errorSeconds;
        result.meanSamples = double(planned) / double(pixels);
        if (!drained) ownership_.finished(grid);
        return result;
    }
    // A picture's pixels no sample has reached filled from the nearest that have (fillUncovered), for the session's
    // own pictures: realtime's, denoised.
    static void fillGaps(std::vector<float>& rgb, const std::vector<unsigned char>& covered, U32 w, U32 h) { fillUncovered(rgb, covered, w, h); }

private:
    // The frame so far, as its end would make it: the mean of each pixel's samples, the host's and the GPU's films
    // merged and their AOVs resolved. The samples in it.
    // crypto: Cryptomatte's fragments too (a checkpoint's), the host's and the GPU's, finalized.
    U64 snapshotResult(HostFilm& film, RenderResult& r, bool adaptive, const GpuLane* gpu, bool crypto = false) const {
        std::vector<double> sums, odd, aov, heat;
        std::vector<U32> counts;
        std::vector<float> depths;
        film.snapshot(sums, counts, odd, aov, depths, heat);
        const size_t pixels = size_t(plan_->width) * plan_->height;
        r.rgba.assign(pixels * 4, 0.0f);
        r.counts.resize(pixels);
        r.odd.assign(pixels * 4, 0.0f);
        U64 seen = 0;
        for (size_t i = 0; i < pixels; ++i) {
            const U32 count = counts[i] + (gpu ? gpu->counts()[i] : 0u);
            r.counts[i] = count;
            seen += count;
            if (!count) continue;
            for (int c = 0; c < 4; ++c) {
                r.rgba[i * 4 + c] = float((sums[i * 4 + c] + (gpu ? gpu->sums()[i * 4 + c] : 0.0)) / count);
                r.odd[i * 4 + c] = float(odd[i * 4 + c] / count);
            }
        }
        resolveAovValues(film.aovTables(), aov.data(), depths.data(), gpu ? gpu->aovs() : nullptr, gpu ? gpu->aovDepths() : nullptr,
                         r.counts, adaptive, r.aov);
        if (!heat.empty()) {
            r.heat.resize(pixels);
            for (size_t p = 0; p < pixels; ++p) r.heat[p] = float(heat[p] + (gpu && gpu->heat() ? gpu->heat()[p] : 0.0));
        }
        if (crypto && !film.crypto().empty()) {
            r.crypto = std::make_shared<aov::CryptoBuffer>();
            film.snapshotCrypto(*r.crypto);
            if (gpu) const_cast<GpuLane*>(gpu)->mergeCrypto(*r.crypto);  // the device's lists, read without being changed
            r.crypto->finalize(r.counts);
        }
        return seen;
    }
    // The AOV accumulators (the host's, and the GPU's merged in) resolved as MoonRay's snapshots resolve them.
    void resolveAovValues(const aov::Tables& t, const double* hostAov, const float* hostDepths, const float* gpuAovs,
                          const float* gpuDepths, const std::vector<U32>& counts, bool adaptive, std::vector<float>& out) const {
        const size_t pixels = size_t(plan_->width) * plan_->height;
        if (!t.channels) return;
        out.assign(pixels * t.channels, 0.0f);
        const unsigned consistent = adaptive ? std::max(2u, plan_->minSamples) : plan_->spp;
        std::vector<double> merged(t.channels);
        std::vector<float> depths(std::max<U32>(t.closestCount, 1));
        for (size_t p = 0; p < pixels; ++p) {
            const double* a = hostAov + p * t.channels;
            if (gpuAovs) {
                std::copy(a, a + t.channels, merged.begin());
                for (U32 k = 0; k < t.closestCount; ++k) depths[k] = hostDepths[p * t.closestCount + k];
                mergeAovPixel(t, merged.data(), depths.data(), gpuAovs + p * t.channels, gpuDepths ? gpuDepths + p * t.closestCount : depths.data());
                a = merged.data();
            }
            float* o = out.data() + p * t.channels;
            const double w = double(counts[p]);
            for (U32 i = 0; i < t.count; ++i) {
                const aov::Entry& e = t.entries[i];
                for (U32 c = 0; c < e.channels; ++c) {
                    const double v = a[e.offset + c];
                    const bool data = std::isfinite(v);
                    double r = v;
                    if (e.type == aov::TypeVisibility) r = v;
                    // MoonRay's frame buffers sum in floats: a sum past the largest float is infinite.
                    else if (data && (e.filter == aov::FilterAvg || e.filter == aov::FilterSum) && std::fabs(v) > 3.402823466e38)
                        r = v > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
                    else if (e.filter == aov::FilterAvg && w > 0 && data) r = v / w;
                    else if (e.filter == aov::FilterConsistent && data) r = v / double(consistent);
                    o[e.offset + c] = float(r);
                }
            }
        }
    }
    // The film's AOV accumulators as MoonRay's snapshots resolve them (snapshotAovBuffer): an average over the pixel's
    // samples (its weight), the force-consistent filter over the frame's consistent sample count (getNumConsistentSamples),
    // the rest as they are (an untouched min or max infinite, as MoonRay leaves it).
    void resolveAovs(HostFilm& film, RenderResult& result, bool adaptive, const GpuLane* gpu) const {
        const aov::Tables& t = film.aovTables();
        const size_t pixels = size_t(plan_->width) * plan_->height;
        resolveAovValues(t, film.aovs().data(), film.aovDepths().data(), gpu ? gpu->aovs() : nullptr, gpu ? gpu->aovDepths() : nullptr,
                         result.counts, adaptive, result.aov);
        if (!film.crypto().empty()) {
            if (gpu) const_cast<GpuLane*>(gpu)->mergeCrypto(film.crypto());
            result.crypto = std::make_shared<aov::CryptoBuffer>(std::move(film.crypto()));
            result.crypto->finalize(result.counts);
        }
        if (!film.heat().empty()) {
            result.heat.resize(pixels);
            const double* gpuHeat = gpu ? gpu->heat() : nullptr;
            for (size_t p = 0; p < pixels; ++p) result.heat[p] = float(film.heat()[p] + (gpuHeat ? gpuHeat[p] : 0.0));
        }
    }
    // An adaptive frame's passes (async_adaptive.h). Passes below the minimum sample
    // count render every pixel and go out together, as table slots free. Each
    // later pass goes out once the trees have seen every pass before it, as each of
    // MoonRay's tiles reads trees its region rebuilt on finishing the pass before.
    // With a GPU, the errors are worked out on the device behind its last launch
    // and only they cross to the host; without one, the host works them out from
    // its film. The region trees then rebuild at once on the pool.
    struct Adaptive {
        using Clock = std::chrono::steady_clock;
        const cpu::Plan& plan;
        U32 samplesPerChunk, tiles, tilesX;
        std::vector<adaptive::Pass> passes;
        adaptive::Regions regions;
        bool cellular;
        adaptive::Cellular cells;
        // Each generated pass's live tiles (cellular frames: the cells; trees: the tiles with samples).
        std::function<void(const std::vector<U32>&)> onGeneration;
        // The same pass in numbers, for the log (Watch::generationLog): its index, its samples, its live tiles and
        // the samples generated in all, so far.
        std::function<void(U32 generation, U64 samples, size_t live, double spent)> onGenerationLog;
        PassTables tables;
        std::vector<U32> planned;    // samples each pixel will have once generated passes finish
        std::vector<U64> expected;   // samples of each generated pass
        std::vector<std::uint64_t> inImage;  // the tile positions inside the image
        // generated: passes sent out; retired: leading passes completed (their table
        // slots are free); evaluated: passes the trees have seen.
        U32 generated = 0, retired = 0, evaluated = 0, treeUpdates = 0;
        bool asked = false;  // errors asked of the GPU for every pass generated
        U64 askedLaunch = 0;  // the GPU's launch count when they were
        bool stop = false;   // the trees have converged: no more adaptive passes
        double spent = 0;    // a frame by sampling generations: the samples of the passes generated
        double budgetSeconds = 0;  // and the time it has: uniform's, priced at the seed's rate (0: the seed not done)
        bool cpuSamples = false;  // the CPU lane renders samples beside the GPU: its samples join the GPU's errors
        Clock::time_point start, askedAt;
        double treeSeconds = 0, errorSeconds = 0;
        std::vector<PassTrace> trace;
        double at() const { return std::chrono::duration<double>(Clock::now() - start).count(); }
        std::vector<double> sums, odd;  // frames without a GPU: the host film's copy
        std::vector<U32> counts;
        std::vector<float> errors;

        Adaptive(const cpu::Plan& p, U32 spc, bool cellularRules = false)
            : plan(p), samplesPerChunk(spc), tiles(((p.width + 7) / 8) * ((p.height + 7) / 8)), tilesX((p.width + 7) / 8),
              passes(p.adaptive ? adaptive::schedule(p.maxSamples) : std::vector<adaptive::Pass>{}),
              regions(int(p.width), int(p.height), p.adaptive ? p.targetError : 1.f),
              cellular((cellularRules || p.sampleBudget) && p.adaptive),
              cells(p.adaptive ? int(p.width) : 1, p.adaptive ? int(p.height) : 1, p.adaptive ? p.targetError : 1.f),
              planned(p.adaptive ? size_t(p.width) * p.height : 0, 0u) {
            if (!p.adaptive) return;
            // Realtime: the first sample in two passes, a pixel in each 2x2 of every tile, then the other three, so that a
            // quarter of the picture's samples shows the whole of it (filled between, fillUncovered).
            if (sparseStart(p) && !passes.empty() && passes[0].start == 0 && passes[0].end == 1) passes.insert(passes.begin(), adaptive::Pass{0, 1});
            cpu::need(passes.size() <= MaxPasses, "Adaptive pass budget");
            inImage.assign(tiles, 0);
            for (U32 tile = 0; tile < tiles; ++tile)
                for (U32 q = 0; q < TilePixels; ++q)
                    if (tilePixel(tile, q, p.width, p.height, p.viewport) != ~0u) inImage[tile] |= 1ull << q;
            if (cellular) cells.deadForever(inImage);
            for (U32 s = 0; s < PassSlots; ++s) {
                tables.masks[s].assign(tiles, 0);
                tables.bases[s].assign(planned.size(), 0);
            }
        }
        U64 done(U32 pass, GpuLane* gpu) const {
            return tables.cpuDone[pass].load(std::memory_order_relaxed) + (gpu ? gpu->passDone()[pass] : 0);
        }
        bool complete(U32 pass, GpuLane* gpu) const { return done(pass, gpu) == expected[pass]; }

        bool uniformStage(U32 j) const { return passes[j].start < plan.minSamples; }
        // A frame by sampling generations ends once uniform's time is up (at most twice uniform's samples).
        bool budgetSpent() const {
            if (!plan.sampleBudget) return false;
            if (plan.budgetBySamples) return spent >= double(plan.sampleBudget);  // the kit's "exactly the samples"
            return spent >= 2.0 * double(plan.sampleBudget) || (budgetSeconds > 0 && at() >= budgetSeconds);
        }
        // Whether pass j's end came well after the one before it (at most half way through the time to it).
        bool firstEndedEarly(U32 j) const { return j > 0 && trace[j - 1].completed < 0.5 * trace[j].completed; }
        // The time a frame by sampling generations has: the seed samples every pixel alike, so its rate is uniform's,
        // and uniform's samples at that rate are uniform's time. Known once the seed's passes have all completed.
        void priceBudget() {
            if (!plan.sampleBudget || plan.budgetBySamples || budgetSeconds > 0) return;
            U32 seed = 0;
            while (seed < passes.size() && uniformStage(seed)) ++seed;
            if (!seed || retired < seed) return;
            double samples = 0;
            for (U32 j = 0; j < seed; ++j) samples += double(expected[j]);
            const double seconds = trace[seed - 1].completed;
            if (samples <= 0 || seconds <= 0) return;
            // Uniform would go on at the seed's steady rate: its passes after the first (whose launches, the first,
            // are slower), which go out together, over the time from the first's end to the seed's end; then the time
            // so far, and the rest of uniform's samples at that rate. Where the first ended about when the seed did (its
            // texture tiles still loading, a car's: 33 s of 34) that time says nothing, and priced uniform at a few
            // seconds more than the seed: all the seed's samples over all its time.
            double rate = samples / seconds;
            if (seed >= 2 && firstEndedEarly(seed - 1))
                rate = (samples - double(expected[0])) / (trace[seed - 1].completed - trace[0].completed);
            budgetSeconds = seconds + std::max(0.0, double(plan.sampleBudget) - samples) / rate;
        }
        // The samples the time left buys, at the rate the last generation ran (its samples over the time since the
        // one before it completed): dear samples, the dragon's, buy fewer than the walls'.
        double samplesLeft() const {
            if (plan.budgetBySamples) return std::max(0.0, double(plan.sampleBudget) - spent);
            double left = 2.0 * double(plan.sampleBudget) - spent;
            if (budgetSeconds > 0) {
                const U32 g = generated - 1;
                double rate = double(plan.sampleBudget) / budgetSeconds;  // uniform's
                // The seed's last pass ends with those before it where they went out together: its time says nothing of its rate.
                if (g > 0 && trace[g].completed > trace[g - 1].completed && (!uniformStage(g) || firstEndedEarly(g)))
                    rate = double(expected[g]) / (trace[g].completed - trace[g - 1].completed);
                left = std::min(left, (budgetSeconds - at()) * rate);
            }
            return std::max(0.0, left);
        }
        // Whether pass j may go out: a free table slot, and either the uniform stage
        // or trees that have seen every pass before it.
        bool ready(U32 j) const {
            // Realtime: the seed's passes one at a time, each over the whole picture before the next (gone out together,
            // a GPU worker finished its cell's four passes while the next cells waited).
            if (plan.realtime && uniformStage(j) && j > retired) return false;
            return j < passes.size() && j < retired + PassSlots && (uniformStage(j) || (!stop && !budgetSpent() && evaluated == j));
        }
        // One pass at a time, so the feeder launches each as soon as it exists.
        void generateReady(Grid& grid, GpuLane* gpu) {
            if (ready(generated)) generate(grid, gpu);
        }
        void generate(Grid& grid, GpuLane* gpu) {
            const U32 j = generated, slot = j % PassSlots, size = passes[j].end - passes[j].start;
            const bool uniform = uniformStage(j);
            auto& masks = tables.masks[slot];
            auto& bases = tables.bases[slot];
            if (!gpu || cpuSamples) bases = planned;  // the CPU lane's; the GPU makes its own
            U64 samples = 0;
            std::vector<std::vector<Chunk>> perCell(grid.cellCount());
            // A frame by sampling generations: a pass no larger than the time left buys, the tiles furthest above the
            // target first, so that the last one does not run on past uniform's time.
            std::vector<char> dropped;
            if (!uniform && cellular && plan.sampleBudget && (budgetSeconds > 0 || plan.budgetBySamples)) {
                std::vector<std::pair<float, U32>> live;
                double wanted = 0;
                for (U32 tile = 0; tile < tiles; ++tile)
                    if (inImage[tile] && cells.alive(tile)) {
                        live.push_back({cells.ratio(tile), tile});
                        wanted += double(__builtin_popcountll(inImage[tile])) * size;
                    }
                const double left = samplesLeft();
                if (wanted > left) {
                    std::sort(live.begin(), live.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
                    dropped.assign(tiles, 1);
                    double taken = 0;
                    for (const auto& [r, tile] : live) {
                        const double cost = double(__builtin_popcountll(inImage[tile])) * size;
                        if (taken + cost > left) break;
                        taken += cost;
                        dropped[tile] = 0;
                    }
                }
            }
            for (U32 tile = 0; tile < tiles; ++tile) {
                std::uint64_t active = inImage[tile];
                // Realtime's first two passes (above): a pixel in each 2x2, then the rest.
                if (sparseStart(plan) && j < 2) {
                    constexpr std::uint64_t sparse = 0x0055005500550055ull;  // even rows' even columns
                    active &= j == 0 ? sparse : ~sparse;
                }
                if (!uniform && passes[j].start < plan.maxSamples)
                    active &= cellular ? (cells.alive(tile) && (dropped.empty() || !dropped[tile]) ? ~std::uint64_t(0) : 0)
                                       : regions.sampleArea(int(tile % tilesX) * 8, int(tile / tilesX) * 8);
                else if (!uniform)
                    active = 0;
                masks[tile] = active;
                if (!active) continue;
                for (U32 q = 0; q < TilePixels; ++q)
                    if (active >> q & 1) {
                        planned[tilePixel(tile, q, plan.width, plan.height)] += size;
                        samples += size;
                    }
                auto& list = perCell[grid.cellOf(tile)];
                for (U32 off = 0; off < size; off += samplesPerChunk)
                    list.push_back({tile, (j << PassShift) | off, std::min(samplesPerChunk, size - off), grid.cellOf(tile)});
            }
            expected.push_back(samples);
            spent += double(samples);
            if (onGeneration || onGenerationLog) {
                std::vector<U32> live;
                for (U32 tile = 0; tile < tiles; ++tile)
                    if (masks[tile]) live.push_back(tile);
                if (onGenerationLog) onGenerationLog(generated, samples, live.size(), spent);
                if (onGeneration) onGeneration(live);
            }
            tables.cpuDone[j].store(0, std::memory_order_relaxed);
            trace.push_back({samples, at()});
            ++generated;
            // A pass with no samples leaves the film, so the trees and every later
            // pass's masks, as they are: the frame has converged.
            if (!samples) {
                stop = true;
                return;
            }
            // On the stream ahead of any launch that reads it; an adaptive pass goes
            // out only once every earlier one has completed.
            if (gpu) gpu->uploadPass(slot, masks, uniform ? passes[j].start : NoSlot);
            for (U32 c = 0; c < perCell.size(); ++c)
                if (!perCell[c].empty()) grid.append(c, perCell[c].data(), U32(perCell[c].size()));
        }

        // The GPU's pass chain (GpuLane::launchChain): the frame's steps, the uniform
        // stage as one, then each adaptive pass; the first step still to queue; and
        // whether the frame is done.
        bool chain = false, chainOut = false, chainDone = false;
        std::vector<GpuLane::ChainStep> steps;
        size_t resume = 0;
        U64 chainsOut = 0;
        U32 uniformEnd = 0, stalls = 0;

        void startChain(GpuLane& gpu, const std::vector<U32>& tileOrder) {
            chain = true;
            while (uniformEnd < passes.size() && uniformStage(uniformEnd)) ++uniformEnd;
            steps.push_back({0, uniformEnd < passes.size() ? passes[uniformEnd].start : passes.back().end, true});
            for (U32 j = uniformEnd; j < passes.size(); ++j) steps.push_back({j, passes[j].end - passes[j].start, false});
            gpu.beginChain(regions, inImage, tileOrder);
        }
        // Queues the chain, and once it has come back either ends the frame or, if
        // a render parked samples, waits for the feeder to rerun them and resumes
        // the chain after that step.
        void advanceChain(GpuLane& gpu) {
            if (chainDone) return;
            if (!chainOut) {
                if (!gpu.ready()) return;
                gpu.launchChain({steps.begin() + std::ptrdiff_t(resume), steps.end()});
                chainOut = true;
                ++chainsOut;
                return;
            }
            if (gpu.chainsCollected() < chainsOut) return;
            const DeviceState& s = gpu.chainState();
            if (!s.chainStall) {
                chainDone = true;
                return;
            }
            if (gpu.inFlight() || gpu.parkedOutstanding()) return;
            size_t k = resume;
            while (k < steps.size() && steps[k].pass != s.chainPass) ++k;
            cpu::need(k < steps.size(), "GPU pass chain stalled outside its steps");
            resume = k + 1;
            ++stalls;
            chainOut = false;
            if (resume == steps.size()) chainDone = true;
        }

        // Moves the frame on: once the next pass needs the trees, gets the errors
        // after every pass sent out and rebuilds the trees, then sends out what that
        // allows. Waits only for the trees, which nothing else needs.
        //
        // The GPU is asked as soon as the last chunk has gone out, so its error
        // launch queues behind the passes. They count only if no launch went out
        // after them (a parked sample's rerun) and every pass has completed;
        // otherwise they are asked for again.
        void advance(Grid& grid, HostFilm& film, GpuLane* gpu, Pool& pool) {
            if (chain) {
                advanceChain(*gpu);
                return;
            }
            while (retired < generated && complete(retired, gpu)) trace[retired++].completed = at();
            priceBudget();
            const bool needTrees = !stop && generated < passes.size() && !uniformStage(generated) && evaluated < generated;
            if (needTrees && gpu) {
                if (!asked && grid.empty() && !gpu->parkedOutstanding()) {
                    gpu->requestErrors();
                    asked = true;
                    askedLaunch = gpu->launchCount();
                    askedAt = Clock::now();
                    trace[generated - 1].asked = at();
                } else if (asked && gpu->errorsReady()) {
                    if (gpu->launchCount() != askedLaunch) {
                        asked = false;
                    } else if (retired == generated) {  // else its launches are still to be collected
                        errorSeconds += std::chrono::duration<double>(Clock::now() - askedAt).count();
                        trace[generated - 1].errors = at();
                        asked = false;
                        rebuild(cpuSamples ? merged(gpu->errors(), film, pool) : gpu->errors(), pool);
                    }
                }
            } else if (needTrees && retired == generated) {  // no GPU: the host film
                film.snapshot(sums, odd, counts);
                errors.resize(counts.size());
                const unsigned n = pool.size() + 1;
                pool.run(n, [&](unsigned part) {
                    for (size_t i = part; i < counts.size(); i += n)
                        errors[i] = pixelError(&sums[i * 4], &odd[i * 4], counts[i]);
                });
                rebuild(errors.data(), pool);
            }
            generateReady(grid, gpu);
        }
        // The GPU's errors with the CPU lane's samples in them. A pixel both lanes sampled has two of MoonRay's errors,
        // each over its own samples: the CPU's from the host film, the GPU's over the rest of the pixel's planned
        // samples. An error is the pixel's spread over the square root of its samples (the two half-images'
        // difference), so the two pool by their counts, sqrt(nG^2 eG^2 + nC^2 eC^2) / n; where one lane has too few
        // samples for an error, the other's stands for the pixel, scaled to all of its samples.
        const float* merged(const float* gpuErrors, HostFilm& film, Pool& pool) {
            film.snapshot(sums, odd, counts);
            errors.assign(gpuErrors, gpuErrors + counts.size());
            const unsigned parts = pool.size() + 1;
            pool.run(parts, [&](unsigned part) {
                for (size_t i = part; i < counts.size(); i += parts) {
                    const U32 nc = counts[i], n = std::max(planned[i], nc);
                    if (!nc || !n) continue;
                    const double ng = double(n - nc), c = double(nc);
                    const float ec = pixelError(&sums[i * 4], &odd[i * 4], nc), eg = ng > 0 ? errors[i] : ErrorNotReady;
                    if (ec < 0 && eg < 0) errors[i] = ErrorNotReady;
                    else if (eg < 0) errors[i] = float(double(ec) * std::sqrt(c / n));
                    else if (ec < 0) errors[i] = float(double(eg) * std::sqrt(ng / n));
                    else errors[i] = float(std::sqrt(ng * ng * double(eg) * eg + c * c * double(ec) * ec) / n);
                }
            });
            return errors.data();
        }
        void rebuild(const float* e, Pool& pool) {
            const auto began = Clock::now();
            if (cellular) {
                // A frame by sampling generations: the target the samples still to spend would bring every tile to.
                const adaptive::Budget budget{&planned, samplesLeft(), plan.maxSamples, plan.noiseTarget};
                stop = cells.update(e, plan.sampleBudget ? &budget : nullptr) == 0;
            } else {
                pool.run(unsigned(regions.count()), [&](unsigned i) { regions.update(i, e); });
                stop = regions.done();
            }
            treeSeconds += std::chrono::duration<double>(Clock::now() - began).count();
            trace[generated - 1].trees = at();
            evaluated = generated;
            ++treeUpdates;
        }
        // Every pass that will run has gone out and completed.
        bool finished(GpuLane* gpu) const {
            if (chain) return chainDone;
            for (U32 j = retired; j < generated; ++j)
                if (!complete(j, gpu)) return false;
            return generated == passes.size() || (!uniformStage(generated) && (stop || budgetSpent()));
        }
    };

    // Important parts first (MoonRay's, tools/port_important.py): each cell's score from the picture so far, the mean
    // difference of its sampled pixels' tones from their left and upper neighbours' (light / (1 + light): bright and
    // dark weighed alike); -1 where none of its pixels has a sample yet.
    std::vector<float> cellScores(const std::vector<float>& rgb, const std::vector<unsigned char>& covered, U32 cellSize) const {
        const U32 w = plan_->width, h = plan_->height, cellsX = (w + cellSize - 1) / cellSize, cellsY = (h + cellSize - 1) / cellSize;
        std::vector<double> sum(size_t(cellsX) * cellsY, 0.0);
        std::vector<U32> n(sum.size(), 0);
        auto tone = [&](size_t i) {
            const float l = 0.2126f * rgb[i * 3] + 0.7152f * rgb[i * 3 + 1] + 0.0722f * rgb[i * 3 + 2];
            return std::isfinite(l) && l > 0 ? l / (1 + l) : 0.0f;
        };
        for (U32 y = 0; y < h; ++y)
            for (U32 x = 0; x < w; ++x) {
                const size_t i = size_t(y) * w + x;
                if (!covered[i]) continue;
                const size_t c = size_t(y / cellSize) * cellsX + x / cellSize;
                const float t = tone(i);
                double d = 0;
                if (x > 0 && covered[i - 1]) d += std::fabs(t - tone(i - 1));
                if (y > 0 && covered[i - w]) d += std::fabs(t - tone(i - w));
                sum[c] += d;
                ++n[c];
            }
        std::vector<float> scores(sum.size(), -1.0f);
        for (size_t c = 0; c < sum.size(); ++c)
            if (n[c]) scores[c] = float(sum[c] / n[c]);
        return scores;
    }
    // A picture's pixels no sample has reached yet, filled from the nearest ones that have samples (the finest level
    // of a pyramid of averages that covers them), so the whole frame shows from its first samples on, as MoonRay's
    // coarse passes show it. The picture alone: the film is untouched.
    static void fillUncovered(std::vector<float>& rgb, const std::vector<unsigned char>& covered, U32 w, U32 h) {
        bool any = false;
        for (unsigned char c : covered)
            if (!c) {
                any = true;
                break;
            }
        if (!any) return;
        struct Level { U32 w, h; std::vector<float> sum; std::vector<float> n; };
        std::vector<Level> levels;
        levels.push_back({w, h, std::vector<float>(size_t(w) * h * 3), std::vector<float>(size_t(w) * h)});
        for (size_t i = 0; i < covered.size(); ++i)
            if (covered[i]) {
                for (int c = 0; c < 3; ++c) levels[0].sum[i * 3 + c] = rgb[i * 3 + c];
                levels[0].n[i] = 1;
            }
        while (levels.back().w > 1 || levels.back().h > 1) {
            const Level& f = levels.back();
            Level next{(f.w + 1) / 2, (f.h + 1) / 2, {}, {}};
            next.sum.assign(size_t(next.w) * next.h * 3, 0.0f);
            next.n.assign(size_t(next.w) * next.h, 0.0f);
            for (U32 y = 0; y < f.h; ++y)
                for (U32 x = 0; x < f.w; ++x) {
                    const size_t from = size_t(y) * f.w + x, to = size_t(y / 2) * next.w + x / 2;
                    for (int c = 0; c < 3; ++c) next.sum[to * 3 + c] += f.sum[from * 3 + c];
                    next.n[to] += f.n[from];
                }
            levels.push_back(std::move(next));
        }
        for (U32 y = 0; y < h; ++y)
            for (U32 x = 0; x < w; ++x) {
                const size_t i = size_t(y) * w + x;
                if (covered[i]) continue;
                for (size_t l = 1; l < levels.size(); ++l) {
                    const Level& L = levels[l];
                    const size_t at = size_t(y >> l) * L.w + (x >> l);
                    if (L.n[at] > 0) {
                        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = L.sum[at * 3 + c] / L.n[at];
                        break;
                    }
                }
            }
    }

    // The image's tiles cell by cell, each cell's row by row, as the work grid holds them.
    std::vector<U32> tileOrder(U32 cellSize) const {
        const U32 tilesX = (plan_->width + TileSize - 1) / TileSize, tilesY = (plan_->height + TileSize - 1) / TileSize;
        const U32 per = cellSize / TileSize;
        std::vector<U32> order;
        order.reserve(size_t(tilesX) * tilesY);
        for (U32 cy = 0; cy * per < tilesY; ++cy)
            for (U32 cx = 0; cx * per < tilesX; ++cx)
                for (U32 ty = cy * per; ty < std::min(tilesY, (cy + 1) * per); ++ty)
                    for (U32 tx = cx * per; tx < std::min(tilesX, (cx + 1) * per); ++tx) order.push_back(ty * tilesX + tx);
        return order;
    }

    // How many chunks the next GPU launch takes: about one window of the GPU's
    // measured work; near the end, the share that lets both devices finish at
    // the same moment, given what the GPU already has queued. gpuRate and
    // queuedSamples in samples; a chunk of the range at the frontier holds a
    // tile's pixels times its samples.
    U32 launchSize(const RenderOptions& o, const Grid& grid, double gpuRate, const CpuLane& cpuLane, U64 queuedSamples,
                   std::chrono::steady_clock::time_point start, double realtimeItems = 0) const {
        U64 remainingChunks = 0;
        for (U32 i = 0; i < grid.cellCount(); ++i) remainingChunks += grid.remaining(i);
        if (!remainingChunks) return 0;
        const auto& ranges = grid.ranges();
        const U32 front = grid.frontier(Device::Gpu);
        const U32 itemsPerChunk = TilePixels * (ranges.empty() || front >= ranges.size() ? o.grid.samplesPerChunk : ranges[front].count);
        const U64 cap = gpu_->maxChunksPerLaunch();
        // An adaptive pass is on the frame's critical path, but goes out in launches of about windowMs (the first
        // ones firstChunks) as a uniform frame's do: the progressive pictures are copied behind the queued launches,
        // so a pass sent whole showed nothing until it ended (a whole first pass of a large frame takes many seconds).
        // Never below two samples a launch thread: a launch with fewer leaves most of the GPU's threads idle, its time
        // the slowest path's, and its rate (which sizes the next ones) far under the GPU's.
        const U64 least = (U64(gpu_->launchThreads()) * 2 + itemsPerChunk - 1) / itemsPerChunk;
        // Realtime: launches of about 15 ms at the GPU's rate (this frame's, else as earlier frames found it), so that an
        // edit cancels the frame within a launch or two and pictures go out between them: the GPU less full, the
        // answer sooner. Between one sample a launch thread and 128K samples (or the threads, where more): a launch with
        // fewer leaves most of the GPU idle, and its rate, which sizes the next ones, is never the GPU's (a heavy scene's
        // launches stayed at 8K samples, the GPU at a third of its rate, 570 launches for a frame).
        if (plan_->realtime) {
            const double rate = gpuRate ? gpuRate : gpu_->rateHint();
            const double threads = double(gpu_->launchThreads());
            const double samples = std::clamp(rate > 0 ? rate * 0.015 : 32768.0, threads, std::max(131072.0, threads));
            const U64 chunks = std::max<U64>(1, U64(samples / (realtimeItems > 0 ? realtimeItems : double(itemsPerChunk))));
            return U32(std::min<U64>({chunks, cap, remainingChunks}));
        }
        if (plan_->adaptive) {
            const U64 want = gpuRate ? std::max<U64>(U64(gpuRate * o.windowMs / 1000.0 / itemsPerChunk), least) : U64(o.firstChunks);
            return U32(std::min<U64>({want, cap, remainingChunks}));
        }
        if (!gpuRate) return U32(std::min<U64>({cap, remainingChunks, std::max<U64>(o.firstChunks, least)}));
        U64 want = U64(gpuRate * o.windowMs / 1000.0 / itemsPerChunk);
        want = std::max<U64>(want, least);
        if (cpuLane.threads()) {
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double cpuRate = elapsed > 0.01 ? double(cpuLane.counters.samples.load()) / elapsed : 0;
            const double remainingItems = double(remainingChunks) * itemsPerChunk;
            // GPU share L so that (queued + L) / gpuRate == (remaining - L) / cpuRate.
            const double share = (remainingItems * gpuRate - double(queuedSamples) * cpuRate) / (gpuRate + cpuRate);
            if (share <= 0) return 0;
            want = std::min<U64>(want, U64(share / itemsPerChunk) + 1);
        }
        return U32(std::min<U64>({want, cap, remainingChunks}));
    }

    cpu::PlanView plan_;
    const CpuScene& cpuScene_;
    GpuLane* gpu_;
    Ownership ownership_;
    std::vector<float> suitability_;  // the indexer's probabilities of the grid's cells (device_suit.h), from its first frame
    double deviceRatio_ = 0;
    std::unique_ptr<CpuLane> cpu_;  // persistent workers (spec §4)
    DWORD_PTR cpuAffinity_ = 0;
    std::unique_ptr<Pool> pool_;  // the adaptive error trees
    // activity(): the frame's CPU lane and the GPU's queued launches' cells.
    mutable std::mutex activityMutex_;
    const CpuLane* activeLane_ = nullptr;
    std::deque<std::vector<U32>> gpuFlights_;
    std::vector<U32> live_;  // liveTiles()
    U32 cellSize_ = 64;
};

}  // namespace xpu2::interactive::async

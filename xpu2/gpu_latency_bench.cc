// GPU round-trip latency on this machine, mimicking MoonRay's
// OptixGPUAccelerator::intersect: upload parameters and rays, launch, download
// results into pinned memory, synchronize. A plain empty CUDA kernel stands in
// for optixLaunch, so results are a lower bound for MoonRay's call.
//
// usage: gpu_latency_bench <sched: auto|spin|yield|blocking> <out.csv>
#include <windows.h>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define FN(n) static decltype(&n) p_##n = nullptr;
FN(cuInit) FN(cuDeviceGet) FN(cuDevicePrimaryCtxSetFlags_v2) FN(cuDevicePrimaryCtxRetain) FN(cuCtxSetCurrent)
FN(cuStreamCreate) FN(cuMemAlloc_v2) FN(cuMemAllocHost_v2) FN(cuMemcpyHtoDAsync_v2) FN(cuMemcpyDtoHAsync_v2)
FN(cuModuleLoadData) FN(cuModuleGetFunction) FN(cuLaunchKernel) FN(cuStreamSynchronize) FN(cuStreamQuery)
FN(cuEventCreate) FN(cuEventRecord) FN(cuEventElapsedTime) FN(cuEventSynchronize) FN(cuMemHostAlloc) FN(cuMemHostGetDevicePointer_v2)
#undef FN

static void check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(int(r)));
}

static const char* kPtx =
    ".version 7.0\n.target sm_52\n.address_size 64\n"
    ".visible .entry empty(.param .u64 p0, .param .u32 n) { ret; }\n"
    // touch: each thread reads one float of its ray and writes one float of its result.
    ".visible .entry touch(.param .u64 pin, .param .u64 pout, .param .u32 n)\n"
    "{\n .reg .pred p; .reg .b32 r<6>; .reg .b64 rd<8>; .reg .f32 f<2>;\n"
    " ld.param.u64 rd1, [pin]; ld.param.u64 rd2, [pout]; ld.param.u32 r1, [n];\n"
    " mov.u32 r2, %ctaid.x; mov.u32 r3, %ntid.x; mov.u32 r4, %tid.x;\n"
    " mad.lo.s32 r5, r2, r3, r4; setp.ge.u32 p, r5, r1; @p bra DONE;\n"
    " cvta.to.global.u64 rd3, rd1; cvta.to.global.u64 rd4, rd2;\n"
    " mul.wide.u32 rd5, r5, 56; add.s64 rd6, rd3, rd5; ld.global.f32 f1, [rd6];\n"
    " mul.wide.u32 rd5, r5, 104; add.s64 rd7, rd4, rd5; st.global.f32 [rd7], f1;\n"
    "DONE: ret;\n}\n";

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Stats { double median, p10, p90, mean; };
static Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * v.size()))]; };
    return {at(0.5), at(0.1), at(0.9), sum / v.size()};
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <auto|spin|yield|blocking> <out.csv>\n", argv[0]); return 2; }
    const std::string sched = argv[1];
    try {
        HMODULE lib = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!lib) throw std::runtime_error("nvcuda.dll");
#define LOAD(n) p_##n = reinterpret_cast<decltype(p_##n)>(GetProcAddress(lib, #n)); if (!p_##n) throw std::runtime_error(#n);
        LOAD(cuInit) LOAD(cuDeviceGet) LOAD(cuDevicePrimaryCtxSetFlags_v2) LOAD(cuDevicePrimaryCtxRetain) LOAD(cuCtxSetCurrent)
        LOAD(cuStreamCreate) LOAD(cuMemAlloc_v2) LOAD(cuMemAllocHost_v2) LOAD(cuMemcpyHtoDAsync_v2) LOAD(cuMemcpyDtoHAsync_v2)
        LOAD(cuModuleLoadData) LOAD(cuModuleGetFunction) LOAD(cuLaunchKernel) LOAD(cuStreamSynchronize) LOAD(cuStreamQuery)
        LOAD(cuEventCreate) LOAD(cuEventRecord) LOAD(cuEventElapsedTime) LOAD(cuEventSynchronize) LOAD(cuMemHostAlloc) LOAD(cuMemHostGetDevicePointer_v2)
#undef LOAD
        check(p_cuInit(0), "cuInit");
        CUdevice dev; check(p_cuDeviceGet(&dev, 0), "cuDeviceGet");
        unsigned flags = sched == "spin" ? CU_CTX_SCHED_SPIN : sched == "yield" ? CU_CTX_SCHED_YIELD
                       : sched == "blocking" ? CU_CTX_SCHED_BLOCKING_SYNC : CU_CTX_SCHED_AUTO;
        check(p_cuDevicePrimaryCtxSetFlags_v2(dev, flags), "set flags");
        CUcontext ctx; check(p_cuDevicePrimaryCtxRetain(&ctx, dev), "retain");
        check(p_cuCtxSetCurrent(ctx), "current");
        CUstream stream; check(p_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "stream");
        CUmodule mod; check(p_cuModuleLoadData(&mod, kPtx), "module");
        CUfunction fn; check(p_cuModuleGetFunction(&fn, mod, "empty"), "function");
        CUfunction touch; check(p_cuModuleGetFunction(&touch, mod, "touch"), "touch");

        constexpr size_t kRay = 56, kIsect = 104, kParams = 64, kMax = 16384;
        CUdeviceptr dRays, dIsect, dParams;
        check(p_cuMemAlloc_v2(&dRays, kRay * kMax), "alloc"); check(p_cuMemAlloc_v2(&dIsect, kIsect * kMax), "alloc");
        check(p_cuMemAlloc_v2(&dParams, kParams), "alloc");
        void *hRaysPinned, *hIsectPinned, *hParamsPinned;
        check(p_cuMemAllocHost_v2(&hRaysPinned, kRay * kMax), "pinned"); check(p_cuMemAllocHost_v2(&hIsectPinned, kIsect * kMax), "pinned");
        check(p_cuMemAllocHost_v2(&hParamsPinned, kParams), "pinned");
        std::vector<char> hRaysPageable(kRay * kMax, 1);
        char hParamsPageable[kParams] = {};
        CUevent e0, e1; check(p_cuEventCreate(&e0, 0), "event"); check(p_cuEventCreate(&e1, 0), "event");

        FILE* out = std::fopen(argv[2], "a");
        if (!out) throw std::runtime_error("output");
        auto report = [&](const char* name, unsigned rays, const Stats& s, double gpuUs) {
            std::fprintf(out, "%s,%s,%u,%.2f,%.2f,%.2f,%.2f,%.2f\n", sched.c_str(), name, rays, s.median, s.p10, s.p90, s.mean, gpuUs);
            std::printf("%-8s %-34s rays=%5u median=%8.1f us  p10=%8.1f  p90=%8.1f  gpu=%7.1f us\n",
                        sched.c_str(), name, rays, s.median, s.p10, s.p90, gpuUs);
        };
        auto launch = [&](unsigned n) {
            unsigned count = n; void* args[] = {&dRays, &count};
            check(p_cuLaunchKernel(fn, std::max(1u, (n + 63) / 64), 1, 1, 64, 1, 1, 0, stream, args, nullptr), "launch");
        };
        // One MoonRay-shaped call. pinnedIn: inputs from pinned memory; spin: poll instead of synchronize.
        auto call = [&](unsigned n, bool pinnedIn, bool spin, bool timeGpu) {
            check(p_cuMemcpyHtoDAsync_v2(dParams, pinnedIn ? hParamsPinned : hParamsPageable, kParams, stream), "params");
            check(p_cuMemcpyHtoDAsync_v2(dRays, pinnedIn ? hRaysPinned : (void*)hRaysPageable.data(), kRay * n, stream), "rays");
            if (timeGpu) check(p_cuEventRecord(e0, stream), "e0");
            launch(n);
            if (timeGpu) check(p_cuEventRecord(e1, stream), "e1");
            check(p_cuMemcpyDtoHAsync_v2(hIsectPinned, dIsect, kIsect * n, stream), "isect");
            if (spin) { while (p_cuStreamQuery(stream) == CUDA_ERROR_NOT_READY) {} }
            else check(p_cuStreamSynchronize(stream), "sync");
        };
        auto run = [&](const char* name, unsigned n, int reps, int gapUs, auto&& body) {
            for (int i = 0; i < 20; ++i) body();              // warm up
            std::vector<double> t; t.reserve(reps);
            for (int i = 0; i < reps; ++i) {
                if (gapUs) std::this_thread::sleep_for(std::chrono::microseconds(gapUs));
                const auto a = Clock::now(); body(); t.push_back(us(a, Clock::now()));
            }
            // GPU-side kernel time for this shape, from events.
            call(n, true, false, true); float ms = 0; check(p_cuEventElapsedTime(&ms, e0, e1), "elapsed");
            report(name, n, summarize(t), ms * 1000.0);
        };
        const int reps = 400;
        if (!std::getenv("RAMP_ONLY")) {
        run("empty_launch_sync", 1, reps, 0, [&] { launch(1); check(p_cuStreamSynchronize(stream), "sync"); });
        run("empty_launch_spin", 1, reps, 0, [&] { launch(1); while (p_cuStreamQuery(stream) == CUDA_ERROR_NOT_READY) {} });
        for (unsigned n : {1u, 512u, 2048u, 8192u, 16384u}) {
            run(("moonray_pageable_sync"), n, reps, 0, [&] { call(n, false, false, false); });
            run(("moonray_pinned_sync"), n, reps, 0, [&] { call(n, true, false, false); });
            run(("moonray_pinned_spin"), n, reps, 0, [&] { call(n, true, true, false); });
        }
        // Sporadic calls: an idle gap before each call, as when CPU work separates calls.
        for (int gap : {200, 2000, 20000}) {
            std::string name = "pageable_sync_after_gap_" + std::to_string(gap) + "us";
            run(name.c_str(), 512, 150, gap, [&] { call(512, false, false, false); });
            std::string name2 = "pinned_spin_after_gap_" + std::to_string(gap) + "us";
            run(name2.c_str(), 512, 150, gap, [&] { call(512, true, true, false); });
        }
        }
        // Real data movement: explicit copies (pageable rays, as MoonRay) against zero-copy
        // host-mapped buffers that the kernel reads and writes directly (one launch + sync).
        {
            void *hIn, *hOut; CUdeviceptr mIn, mOut;
            check(p_cuMemHostAlloc(&hIn, kRay * kMax, CU_MEMHOSTALLOC_DEVICEMAP), "mapped in");
            check(p_cuMemHostAlloc(&hOut, kIsect * kMax, CU_MEMHOSTALLOC_DEVICEMAP), "mapped out");
            check(p_cuMemHostGetDevicePointer_v2(&mIn, hIn, 0), "dptr in");
            check(p_cuMemHostGetDevicePointer_v2(&mOut, hOut, 0), "dptr out");
            auto touchLaunch = [&](CUdeviceptr in, CUdeviceptr outp, unsigned n) {
                unsigned count = n; void* args[] = {&in, &outp, &count};
                check(p_cuLaunchKernel(touch, std::max(1u, (n + 63) / 64), 1, 1, 64, 1, 1, 0, stream, args, nullptr), "touch");
            };
            for (unsigned n : {1u, 512u, 2048u, 8192u}) {
                run("touch_copies_pageable_sync", n, reps, 0, [&] {
                    check(p_cuMemcpyHtoDAsync_v2(dParams, hParamsPageable, kParams, stream), "params");
                    check(p_cuMemcpyHtoDAsync_v2(dRays, hRaysPageable.data(), kRay * n, stream), "rays");
                    touchLaunch(dRays, dIsect, n);
                    check(p_cuMemcpyDtoHAsync_v2(hIsectPinned, dIsect, kIsect * n, stream), "isect");
                    check(p_cuStreamSynchronize(stream), "sync"); });
                run("touch_zero_copy_sync", n, reps, 0, [&] { touchLaunch(mIn, mOut, n); check(p_cuStreamSynchronize(stream), "sync"); });
                run("touch_zero_copy_spin", n, reps, 0, [&] { touchLaunch(mIn, mOut, n); while (p_cuStreamQuery(stream) == CUDA_ERROR_NOT_READY) {} });
            }
        }
        // Ramp-up: after an idle gap, latency of the k-th consecutive call.
        for (int gap : {2000, 20000}) {
            std::vector<std::vector<double>> byIndex(20);
            for (int trial = 0; trial < 60; ++trial) {
                std::this_thread::sleep_for(std::chrono::microseconds(gap));
                for (int k = 0; k < 20; ++k) {
                    const auto a = Clock::now(); call(512, false, false, false); byIndex[k].push_back(us(a, Clock::now()));
                }
            }
            for (int k : {0, 1, 2, 4, 9, 19}) {
                std::string name = "ramp_gap_" + std::to_string(gap) + "us_call_" + std::to_string(k + 1);
                report(name.c_str(), 512, summarize(byIndex[k]), 0.0);
            }
        }
        std::fclose(out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

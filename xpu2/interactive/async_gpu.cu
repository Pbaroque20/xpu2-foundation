// The asynchronous GPU lane's OptiX program (spec §10, design G2).
//
// Each launch is a fixed-size grid. Every thread claims one sample at a time
// with an atomic on its launch's own counter, first from the parked samples it
// was given, then from its range of chunk items, and exits when both are used
// up; the host sizes the ranges, never the grid (GPU-04). Neighbouring claims
// are neighbouring pixels of the same sample, so a warp traces coherent rays.
// Results accumulate in the device film in double precision (ACC-04, ACC-06).
// Traversal and triangle intersection use the hardware; the shared integrator
// reconstructs each hit with the same arithmetic as the CPU lane.
#include "async_gpu_device.h"
// One program renders uniform and adaptive frames: the launch's adaptive flag turns on the adaptive film (its odd
// samples, the passes' counts), the error pass and the pass chain. The path tracer is out of line (X2_CALL), so
// the adaptive parts cost a uniform frame no registers, and OptiX compiles one module instead of two of nearly the
// same code.

extern "C" __global__ void __miss__radiance() { optixSetPayload_0(0xffffffffu); }
extern "C" __global__ void __miss__shadow() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__triangle() {
    optixSetPayload_0(optixGetInstanceId());
    optixSetPayload_1(optixGetPrimitiveIndex());
}
// A curve span (MoonRay XPU's curve programs, curve_core.h) or a point (points_core.h): only the distance is reported;
// the hit is worked out again afterwards, as on the CPU.
extern "C" __global__ void __intersection__curve() {
    if (params.meshMotion && params.meshMotion[optixGetInstanceId()].pointRadius) {
        const MeshMotion& motion = params.meshMotion[optixGetInstanceId()];
        const U32 prim = optixGetPrimitiveIndex();
        const surface::V3 centre = motion.steps > 1 ? motionVertex(motion, prim, optixGetRayTime()) : params.geometry[optixGetInstanceId()].vertices[prim];
        const float3 from = optixGetObjectRayOrigin(), along = optixGetObjectRayDirection();
        float t;
        surface::V3 Ng;
        if (points::intersect(centre, motion.pointRadius[prim], surface::V3{from.x, from.y, from.z}, surface::V3{along.x, along.y, along.z}, optixGetRayTmin(),
                              optixGetRayTmax(), t, Ng))
            optixReportIntersection(t, 0);
        return;
    }
    const curves::Set& still = params.meshAttributes[optixGetInstanceId()].curve;
    curves::Set moved;
    if (still.stepCount > 1) {  // curves in motion: their control points at the ray's time
        moved = still;
        moved.time = optixGetRayTime();
    }
    const curves::Set& c = still.stepCount > 1 ? moved : still;
    const float3 o = optixGetObjectRayOrigin(), d = optixGetObjectRayDirection();
    curves::Hit h;
    if (curves::intersectSpan(c, optixGetPrimitiveIndex(), surface::V3{o.x, o.y, o.z}, surface::V3{d.x, d.y, d.z}, optixGetRayTmin(),
                              optixGetRayTmax(), h))
        optixReportIntersection(h.t, 0);
}


// The film's math filters (async_cpu.h mergeAov, Film.cc's atomicMathFilter): averages and sums add, min and max keep
// the least and greatest, the closest keeps the nearer sample's values under its pixel's lock; force-consistent
// sampling keeps nothing, as MoonRay's vector film does.
static __device__ void atomicMinFloat(float* a, float v) {
    int old = __float_as_int(*a);
    while (v < __int_as_float(old)) {
        const int seen = atomicCAS(reinterpret_cast<int*>(a), old, __float_as_int(v));
        if (seen == old) break;
        old = seen;
    }
}
static __device__ void atomicMaxFloat(float* a, float v) {
    int old = __float_as_int(*a);
    while (v > __int_as_float(old)) {
        const int seen = atomicCAS(reinterpret_cast<int*>(a), old, __float_as_int(v));
        if (seen == old) break;
        old = seen;
    }
}
// A depth's bits in the order of the depths (negative ones reversed below the positive), for the closest filter's words.
static __device__ U32 orderedDepth(float z) {
    const U32 b = __float_as_uint(z);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}
static __device__ void mergeAovs(const aov::Tables& t, U32 pixel, const float* values, const float* depths) {
    float* acc = params.aovs + size_t(pixel) * t.channels;
    U32 closestAt = 0;  // the entry's first word among the pixel's closest channels
    for (U32 i = 0; i < t.count; ++i) {
        const aov::Entry& e = t.entries[i];
        float* d = acc + e.offset;
        const float* v = values + e.offset;
        switch (e.filter) {
        case aov::FilterAvg:
        case aov::FilterSum:
            for (U32 c = 0; c < e.channels; ++c)
                if (v[c] != 0.0f) atomicAdd(d + c, v[c]);
            break;
        case aov::FilterMin:
            for (U32 c = 0; c < e.channels; ++c) atomicMinFloat(d + c, v[c]);
            break;
        case aov::FilterMax:
            for (U32 c = 0; c < e.channels; ++c) atomicMaxFloat(d + c, v[c]);
            break;
        case aov::FilterClosest: {
            // Without a lock: each channel keeps the least of its words, the nearest sample's depth above its value
            // (the least value's bits between samples at one depth); the host reads them apart (GpuLane::landClosest).
            U64L* k = params.aovClosest + size_t(pixel) * params.aovClosestChannels + closestAt;
            closestAt += e.channels;
            const float z = depths[e.closest];
            if (!(z < aov::infinity())) break;
            const U64L high = U64L(orderedDepth(z)) << 32;
            if (high > (*reinterpret_cast<volatile U64L*>(k) & 0xffffffff00000000ull)) break;
            for (U32 c = 0; c < e.channels; ++c) atomicMin(k + c, high | __float_as_uint(v[c]));
            break;
        }
        default: break;
        }
    }
}
// A sample's Cryptomatte events into its pixel's fragment lists (CryptoBuffer::add), in every category. Without locks:
// OptiX promises nothing to a thread waiting on another, and a pixel's lock hung the render. Nodes are never taken
// out, so a fragment once found stays found; a new one goes in front by a compare-and-swap on the list's head, and
// a thread that loses the swap looks only through the nodes put in front since, for its fragment. A node made for a
// swap that lost to the same fragment is kept for this thread's next new fragment; the last one is left unlinked.
static __device__ void mergeCrypto(U32 pixel, const aov::CryptoEvent* events, U32 count) {
    const U32 words = params.cryptoNodeWords;
    U32 spare = ~0u;
    for (U32 k = 0; k < count; ++k) {
        const aov::CryptoEvent& e = events[k];
        if (!e.used || !(e.increment || e.credit)) continue;
        for (U32 c = 0; c < params.cryptoCategories; ++c) {
            const float id = c == 0 ? e.id : c == 1 ? e.materialId : e.assetId;
            U32* head = params.cryptoHeads + size_t(pixel) * params.cryptoCategories + c;
            U32* node = nullptr;
            bool made = false;
            for (U32 from = *reinterpret_cast<volatile U32*>(head), stop = ~0u;;) {
                __threadfence();
                for (U32 n = from; n != stop && n != ~0u;) {
                    volatile U32* m = params.cryptoNodes + size_t(n) * words;
                    if (__uint_as_float(m[1]) == id && (!params.cryptoMultiPresence || m[4] == e.presenceDepth)) {
                        node = const_cast<U32*>(m);
                        break;
                    }
                    n = m[0];
                }
                if (node) break;
                if (spare == ~0u) {
                    spare = atomicAdd(params.cryptoNodeCount, 1u);
                    if (spare >= params.cryptoNodeCapacity) {
                        spare = ~0u;
                        atomicAdd(&params.state->errors, 1u);
                        break;
                    }
                }
                volatile U32* m = params.cryptoNodes + size_t(spare) * words;
                for (U32 w = 2; w < words; ++w) m[w] = 0u;
                m[0] = from;
                m[1] = __float_as_uint(id);
                m[3] = 1u;  // a new fragment counts one sample (Cryptomatte.cc's Fragment)
                m[4] = e.presenceDepth;
                __threadfence();
                const U32 was = atomicCAS(head, from, spare);
                if (was == from) {
                    node = const_cast<U32*>(m);
                    made = true;
                    spare = ~0u;
                    break;
                }
                stop = from;
                from = was;
            }
            if (!node) continue;
            if (!made && e.increment) atomicAdd(node + 3, 1u);
            atomicAdd(reinterpret_cast<float*>(node + 2), e.weight);
            if (params.cryptoExtras) {
                const float d[21] = {e.position.x, e.position.y, e.position.z, e.p0.x, e.p0.y, e.p0.z, e.normal.x, e.normal.y, e.normal.z,
                                     e.beauty[0], e.beauty[1], e.beauty[2], e.beauty[3], e.refP.x, e.refP.y, e.refP.z,
                                     e.refN.x, e.refN.y, e.refN.z, e.uv[0], e.uv[1]};
                for (U32 w = 0; w < 21; ++w)
                    if ((e.increment || (w >= 9 && w < 13)) && d[w] != 0.0f) atomicAdd(reinterpret_cast<float*>(node + 5 + w), d[w]);
            }
        }
    }
}
// mode 4: a pixel's accumulators readied for a frame (initAovPixel): +inf for min, -inf for max, 0 for the rest,
// and the closest filter's depths +inf.
static __device__ void initAovs() {
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.pixels) return;
    const aov::Tables& t = *params.aovTables;
    float* acc = params.aovs + size_t(i) * t.channels;
    for (U32 k = 0; k < t.count; ++k) {
        const aov::Entry& e = t.entries[k];
        const float v = e.filter == aov::FilterMin ? aov::infinity() : e.filter == aov::FilterMax ? -aov::infinity() : 0.0f;
        for (U32 c = 0; c < e.channels; ++c) acc[e.offset + c] = v;
    }
    for (U32 k = 0; k < params.aovClosestChannels; ++k) params.aovClosest[size_t(i) * params.aovClosestChannels + k] = ~0ull;  // none yet
    if (params.heat) params.heat[i] = 0.0;
    if (params.cryptoHeads) {
        for (U32 c = 0; c < params.cryptoCategories; ++c) params.cryptoHeads[size_t(i) * params.cryptoCategories + c] = ~0u;
        if (i == 0) *params.cryptoNodeCount = 0u;
    }
}

// Partial shading. A provisional sample's record for the host (async_render.h hands it to the CPU lane, or back to
// the GPU once its tile has landed), its value as the provisional film holds it.
static __device__ void refineRecord(U32 pixel, U32 index, U32 pass, U32 mark, const float* v) {
    const U64L at = atomicAdd(&params.state->refineTail, 1ull);
    params.refines[at % params.refineCapacity] = Refine{pixel, index, pass, mark == ~0u ? 0u : mark, {v[0], v[1], v[2], v[3]}};
}
// A sample that read a coarser level: its value into the provisional film (in place of its old one when it was
// itself a provisional sample rendered again), and its record. Nothing else of it reaches the film.
static __device__ void provisional(U32 pixel, U32 index, U32 pass, U32 mark, const Radiance& r, const Refine* old) {
    const float v[4] = {float(r.rgb[0]), float(r.rgb[1]), float(r.rgb[2]), float(r.alpha)};
    double* s = params.provSums + size_t(pixel) * 4;
    for (int c = 0; c < 4; ++c) atomicAdd(s + c, double(v[c]) - (old ? double(old->rgba[c]) : 0.0));
    if (!old) atomicAdd(params.provCounts + pixel, 1u);
    refineRecord(pixel, index, pass, mark, v);
}
// A provisional sample rendered again in full (or deferred to the CPU lane, which renders it in full): its value
// leaves the provisional film.
static __device__ void settle(U32 pixel, const Refine& old) {
    double* s = params.provSums + size_t(pixel) * 4;
    for (int c = 0; c < 4; ++c) atomicAdd(s + c, -double(old.rgba[c]));
    atomicSub(params.provCounts + pixel, 1u);
}

// One sample. old: a provisional sample rendered again in full (null: a sample of its own), which was not counted in
// its pass and is now.
static __device__ __noinline__ void run(GpuBackend& backend, U32 pixel, U32 index, U32 tries, U32 pass, bool chain, const Refine* old) {
    Radiance r;
    const aov::Tables& at = *params.aovTables;
    if (at.channels) {
        r.aov = reinterpret_cast<float*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride + params.aovOffset);
        r.aovDepth = r.aov + at.channels;
        r.aovBackup = r.aovDepth + at.closestCount;
    }
    if (params.cryptoHeads) {
        r.crypto = reinterpret_cast<aov::CryptoEvent*>(params.scratch + U64L(optixGetLaunchIndex().x) * params.scratchStride + params.cryptoOffset);
        r.cryptoCapacity = params.cryptoEvents;
    }
    const long long began = params.heat ? clock64() : 0;
    const long long sampleFrom = backend.tick();
    struct Whole {  // the GPU profile: the whole sample, however it ends
        GpuBackend& b;
        long long from;
        __device__ ~Whole() { b.tock(ProfileSample, from); }
    } whole{backend, sampleFrom};
    U32* mark = params.partial ? params.lowFlags + optixGetLaunchIndex().x : nullptr;
    if (mark) *mark = 0u;
    const SampleStatus status = integrate(*params.config, pixel, index, backend, r);
    if (status == SampleStatus::Done && mark && *mark) {
        provisional(pixel, index, pass, *mark, r, old);
        return;
    }
    if (old && status != SampleStatus::Miss) settle(pixel, *old);
    if (status == SampleStatus::Done) {
        if (at.channels) mergeAovs(at, pixel, r.aov, r.aovDepth);
        if (params.cryptoHeads) mergeCrypto(pixel, r.crypto, r.cryptoCount);
        if (params.heat) atomicAdd(params.heat + pixel, double(clock64() - began) * params.secondsPerTick);
        double* s = params.sums + size_t(pixel) * 4;
        atomicAdd(s + 0, r.rgb[0]);
        atomicAdd(s + 1, r.rgb[1]);
        atomicAdd(s + 2, r.rgb[2]);
        atomicAdd(s + 3, r.alpha);
        atomicAdd(params.counts + pixel, 1u);
        if (params.adaptive) {
            if (index & 1) {  // MoonRay's Beauty Odd: the odd-index samples
                double* o = params.odd + size_t(pixel) * 4;
                atomicAdd(o + 0, r.rgb[0]);
                atomicAdd(o + 1, r.rgb[1]);
                atomicAdd(o + 2, r.rgb[2]);
                atomicAdd(o + 3, r.alpha);
            }
            atomicAdd(&params.state->passDone[pass], 1u);
        }
    } else if (status == SampleStatus::Skip) {
        // No camera ray: taken, nothing added.
        if (params.adaptive) atomicAdd(&params.state->passDone[pass], 1u);
    } else if (status == SampleStatus::Miss && old) {
        // A provisional sample that found no coarser level either: it stays provisional, as it was.
        refineRecord(pixel, index, pass, 0u, old->rgba);
    } else if (status == SampleStatus::Miss) {
        // Park it; the host sized this launch so the queue cannot overflow.
        const U64L at = atomicAdd(&params.state->parkedTail, 1ull);
        params.parked[at % params.parkedCapacity] = {pixel, index, tries, pass};
        if (chain) params.state->chainStall = 1;  // the chain waits for the host to land its tile
    } else if (status == SampleStatus::Defer) {
        // For the CPU lane (async_render.h): nothing of the sample reaches the GPU's film.
        const U64L at = atomicAdd(&params.state->deferredTail, 1ull);
        params.deferred[at % params.deferredCapacity] = {pixel, index, 0u, pass};
    } else {
        atomicAdd(&params.state->errors, 1u);
    }
}

static __device__ void commit() {
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.updateCount) return;
    const TileUpdate u = params.updates[i];
    auto** entry = reinterpret_cast<const surface::V3**>(params.pageEntry[u.tile]);
    *entry = reinterpret_cast<const surface::V3*>(u.address);
    params.tileState[u.tile] = u.address ? 2u : 0u;
}

// The film's per-pixel adaptive errors, read by the host's error trees.
static __device__ void errors() {
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.pixels) return;
    params.errors[i] = pixelError(params.sums + size_t(i) * 4, params.odd + size_t(i) * 4, params.counts[i]);
}

// The samples a launch renders, from wherever they come, for one call of run(): OptiX inlines a program's whole call
// graph into its entry, so each call of the path tracer would be a copy of it, its traces with it. The provisional
// samples rendered again in full and the parked samples first (they started before anything else here), then the
// launch's chunk items; or a pass the chain made on the device (async_adaptive.cu, mode 3).
struct Feed {
    U32 phase = 4;  // 4 the provisional samples, 0 the parked samples, 1 the chunk items, 2 the chain's pass, 3 none left
    Refine refine;  // the provisional sample being rendered again
    __device__ bool next(U32& pixel, U32& index, U32& tries, U32& pass, bool& chain, const Refine*& old) {
        const auto& c = *params.config;
        old = nullptr;
        for (;;) {
            if (phase == 4) {
                const U32 k = params.refineInCount ? atomicAdd(&params.state->refineClaimed[params.slot], 1u) : 0u;
                if (k >= params.refineInCount) {
                    phase = 0;
                    continue;
                }
                refine = params.refineIn[k];
                pixel = refine.pixel;
                index = refine.index;
                tries = 0;
                pass = refine.pass;
                chain = false;
                old = &refine;
                return true;
            }
            if (phase == 0) {
                const U64L k = params.parkedBegin + atomicAdd(&params.state->unparked[params.slot], 1u);
                if (k >= params.parkedEnd) {
                    phase = 1;
                    continue;
                }
                const Parked p = params.parked[k % params.parkedCapacity];
                pixel = p.pixel;
                index = p.index;
                tries = p.tries + 1;
                pass = p.pass;
                chain = false;
                return true;
            }
            if (phase == 1) {
                const U64L item = params.itemBegin + atomicAdd(&params.state->claimed[params.slot], 1u);
                if (item >= params.itemEnd) {
                    phase = 3;
                    continue;
                }
                const U64L chunkIndex = item / params.itemsPerChunk;
                const U32 offset = U32(item - chunkIndex * params.itemsPerChunk);
                const Chunk chunk = params.ring[chunkIndex % params.ringCapacity];
                const U32 s = offset / TilePixels, position = offset % TilePixels;
                if (s >= chunk.count) continue;
                const U32 at = tilePixel(chunk.tile, position, c.width, c.height, c.viewport);
                if (at == ~0u) continue;
                U32 sampleIndex = chunk.first + s, samplePass = 0;
                if (params.adaptive) {
                    // Only the tile's active pixels, each from its own sample-index base.
                    const U32 slot = chunkPass(chunk) % PassSlots;
                    if (!(params.passMask[slot][chunk.tile] >> position & 1)) continue;
                    sampleIndex = params.passBase[slot][at] + chunkOffset(chunk) + s;
                    samplePass = chunkPass(chunk);
                }
                pixel = at;
                index = sampleIndex;
                tries = 0;
                pass = samplePass;
                chain = false;
                return true;
            }
            if (phase == 2) {
                // Threads claim items one at a time, as in every render, so slow samples do not hold up a thread's share.
                DeviceState* s = params.state;
                const U64L perTile = U64L(params.chainSize) * TilePixels, items = U64L(s->activeTiles) * perTile;
                const U64L item = atomicAdd(&s->chainClaimed, 1ull);
                if (item >= items) {
                    phase = 3;
                    continue;
                }
                const U32 tile = U32(item / perTile), rest = U32(item % perTile);
                const U32 sample = rest / TilePixels, position = rest % TilePixels;
                if (!(params.activeMasks[tile] >> position & 1)) continue;
                pixel = tilePixel(params.activeTiles[tile], position, c.width, c.height);
                index = params.chainBase[pixel] + sample;
                tries = 0;
                pass = params.chainPass;
                chain = true;
                return true;
            }
            return false;
        }
    }
};


// A diagnostic build's exceptions (XPU2_OPTIX_EXCEPTIONS, async_gpu.h): the first one's code and launch index kept
// for the host, which logs them; the launch counts an error, so the frame falls back.
extern "C" __global__ void __exception__report() {
    if (atomicCAS(&params.state->exceptionIndex, 0u, optixGetLaunchIndex().x + 1u) == 0u)
        params.state->exceptionCode = U32(optixGetExceptionCode());
    atomicAdd(&params.state->errors, 1u);
}

extern "C" __global__ void __raygen__async() {
    if (params.mode == 1) {
        commit();
        return;
    }
    if (params.mode == 4) {
        initAovs();
        return;
    }
    if (params.mode == 2) {
        errors();
        return;
    }
    Feed feed;
    if (params.mode == 3) {
        // A pass the chain made runs only if its own compaction ran: after a stall or once the frame has converged,
        // the chain's later renders find another pass there and do nothing.
        if (*reinterpret_cast<volatile U32*>(&params.state->chainPass) != params.chainPass) return;
        feed.phase = 2;
    }
    GpuBackend backend;
    U32 pixel, index, tries, pass;
    bool chain;
    const Refine* old;
    while (feed.next(pixel, index, tries, pass, chain, old)) run(backend, pixel, index, tries, pass, chain, old);
    if (params.profile)  // the thread's profile into the launch's, once
        for (U32 k = 0; k < ProfileStages; ++k)
            if (backend.prof[k]) atomicAdd(&params.state->profile[k], backend.prof[k]);
}

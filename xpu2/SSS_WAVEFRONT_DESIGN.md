# Random-walk subsurface on the GPU, bit-exact with XPU: design and plan

Status: design only. Nothing here is implemented, installed or committed.
Measurements: the kit-exported Modo scene
`test-results/cl-export/Technik_Tom_Moonray_bundled_baked.rdla`, 200x200,
16 samples, one thread, XPU, RTX 3090, Windows 10, MoonRay 0.4.2 runtime with
an isolated capture `librendering_rndr.dll` (`xpu2/rndr_sss_feasibility.py`,
`xpu2/sss_probe.h`, sampling 1 ray in 16).

## Requirement

XPU2 must be able to fall back to XPU. Every change sits behind a switch; with
it off MoonRay runs XPU's code exactly. With it on, every subsurface ray must be
answered exactly as Embree answers it today, and every shading result and sum
must be unchanged.

## What happens today

In XPU mode each subsurface lane leaves the bundled integrator for scalar C++
(`CPP_computeRadianceSubsurface`). Random-walk subsurface
(`computeRadiancePathTraceSubsurface`) runs, per lane, `mBssrdfSamples` walks at
camera depth (64 here) or one walk deeper. Each walk steps up to 256 times; every
step traces one ray through Embree, synchronously. On exit,
`computeRadianceSubsurfaceSample` lights the exit point and continues the path
depth-first in scalar mode (more Embree rays and CPU occlusion rays).

## Measurements

| Quantity | Value |
| --- | --- |
| Walk rays / other subsurface rays / subsurface occlusion rays | 9.22 M / 1.35 M / 3.32 M |
| Walks | 1.35 M, in 28,857 subsurface calls |
| Steps per walk | mostly 3-8 (1.07 M of 1.35 M walks); 9-16: 0.22 M; 17-32: 49 K; longer: 8.5 K |
| Walks per call | 1 walk: 13,199 calls; 257-512 walks: 2,036 calls (most rays) |
| Rays by wave size, if a call's walks stepped together | 257-512: 4.2 M; 129-256: 1.7 M; 65-128: 1.1 M; smaller: 2.2 M |
| Embree, walk rays | hit 523 ns, miss 691 ns; 85% of walk rays miss |
| Embree, other subsurface rays | hit 1,031 ns, miss 371 ns |
| One-triangle confirmation (Embree's hit triangle alone, same device) | exact 84,255 of 84,255 walk hits, 16,491 of 16,491 others |
| GPU (OptiX) agreement, walk rays | same triangle 14.6%, both miss 84.8%, GPU hit where Embree missed 0.48%, different triangle 0.11%, GPU miss where Embree hit 0% |
| OptiX t against Embree t (same triangle) | relative error < 1e-6 in 97%, max 9.3e-4; absolute max 7.2e-7 |
| GPU call latency (XPU's `GPUAccelerator::intersect`) | 1 ray 1.18 ms, 512 rays 0.40 ms, 2,048 rays 0.61 ms, 8,192 rays 1.38 ms |

The GPU latency is about 0.35 ms fixed per call plus about 0.125 us per ray.
MoonRay's own XPU ray queue batches 16,384 rays per thread for the same reason.
Against Embree's 0.6-0.7 us per ray, a GPU call breaks even at about 700 rays and
is 2-4 times cheaper at 2,000-8,000 rays.

### Shortened Embree traces are not safe

Re-tracing an Embree hit on [tnear, t + max(t*1e-3, 1e-5)] returned the same
answer for 99.76% of walk hits but missed in 205 cases and found another triangle
once. All failures are tiny hits (t 6e-5 to 5e-3) about 10.6 units from the world
origin: Embree's box tests are not conservative, and their rounding there exceeds
the margin, so the box holding the hit is culled. A margin derived from coordinate
magnitude might work but needs proof; the design below does not need shortening.

## Exact GPU-guided trace

For a batch of rays:

1. The GPU runs a conservative test: triangles enlarged beyond Embree's
   rounding bound, the interval [tnear - e, tfar + e], every mask treated as a
   possible match, no filters. Its hit set is then a superset of Embree's.
   RTX hardware triangles cannot be enlarged, so this is a custom intersection
   program over an AABB acceleration structure of the subsurface trace-set
   geometry only.
2. GPU miss: the ray is a miss; Embree would also miss. No CPU work. This is
   85% of walk rays and their more expensive half (691 ns each).
3. GPU hit: Embree traces the ray in full, exactly as today.

Either branch returns exactly Embree's answer, so the scheduler may also send
any batch straight to Embree (small batches, tail of long walks) without
affecting results. Even with hardware triangles, no sampled ray had a GPU miss
where Embree hit; the conservative program must make that a guarantee.

## The wavefront inside one subsurface call (order-preserving)

Walks are independent: each draws from its own sequences (pixel, subpixel,
sample index, subsurface sample id) with its own counters. So all walks of one
`CPP_computeRadianceSubsurface` call (up to 8 lanes x 64) can step together:

- Walk state: the four integrator samplers, throughput, origin, direction,
  scatter count, Dwivedi flags, and the lane it belongs to.
- Each step: collect the rays of all active walks, trace them as one batch
  (GPU-guided or Embree), continue each walk; a walk that exits stores its exit
  intersection and throughput.
- After all walks finish, run every exit's `computeRadianceSubsurfaceSample` in
  the original order (lane, then sample s), summing radiance and the subsurface
  AOV exactly as the sequential code does. RNG use, shading and sums are
  unchanged; only the time at which rays are traced moves.
- To check before trusting it: intersection ids (`tls->getNextId()`) would be
  handed out in a different order. Shaders use them as cache keys (StamMap);
  confirm they never change results, or hand ids out in the original order.

The catch is batch size: a call has at most 512 walks, and most rays sit in
waves of 257-512. At the measured 0.35 ms per call, 512-ray waves cost 0.79 us
per ray, more than Embree. At today's latency the in-call wavefront would be
slower.

## Two ways past the latency

A. Cut the per-call GPU latency. If a call cost about 50 us, break-even would
   be about 80 rays and 512-ray waves would win while keeping everything inside
   one call, fully order-preserving and bit-exact. Candidate causes to measure:
   the Windows display driver model (WDDM) submission cost on a GeForce card,
   GPU clocks dropping between sporadic calls, uploads from pageable memory
   (MoonRay copies rays from arena memory), the launch itself. Hardware-
   accelerated GPU scheduling is not enabled on this machine (no `HwSchMode`
   value); changing it is the user's decision, not part of this work.

B. Pool walks across calls and threads to reach XPU-sized batches (thousands of
   rays). Then `CPP_computeRadianceSubsurface` can no longer return its lanes'
   radiance immediately: walks are parked and their results delivered later.
   Every ray and shading result can stay exact, but the order in which
   samples are added into each pixel changes, so pixel sums can differ in the
   last bits from XPU. That breaks strict bit-for-bit equality with the switch on
   (the switch-off fallback stays exact). It is a decision for the user, and a
   much larger change to the bundled integrator.

## Step 0 result: where the GPU call time goes

`xpu2/gpu_latency_bench.cc` (CUDA driver API, `test-results/gpu-latency`) mimics
MoonRay's call: upload parameters and rays, launch, download results into pinned
memory, synchronize. A plain kernel stands in for `optixLaunch`. Medians, RTX
3090, Windows 10 (WDDM), back-to-back calls unless noted:

| Call | 1 ray | 512 rays | 2,048 rays | 8,192 rays |
| --- | ---: | ---: | ---: | ---: |
| Empty launch + synchronize | 30-42 us | | | |
| MoonRay pattern, pageable inputs | 76-82 us | 104-117 us | 189-197 us | 316-326 us |
| MoonRay pattern, pinned inputs | 115 us | 117 us | 146 us | 217-226 us |
| Zero-copy (kernel reads and writes host-mapped memory; one launch + sync) | 30 us | 32 us | 50 us | 110 us |

- Context wait mode (auto, spin, blocking) changes little.
- After an idle gap the first call is slow: 512 rays take 250-380 us after
  2 ms and 800-1,000 us after 20 ms; by the second call (plain CUDA) it is back
  to about 130 us.

Inside MoonRay (OptiX, `GPUAccelerator::intersect`, walk rays of the Modo scene):

| Call | 1 thread | 12 threads |
| --- | ---: | ---: |
| First 512 rays after an idle gap | 1.22 ms | 0.63 ms |
| Steady 1-ray calls | 207 us | 234 us |
| Steady 512-ray calls | 362 us | 276 us |
| 8,192 rays | 1.20 ms | 0.67 ms |

So MoonRay's call costs about 200 us fixed (OptiX launch plus its explicit copies)
plus about 150 us of traversal for 512 rays, which is one ray's traversal depth
through the whole scene (8,382 meshes as instances, with hit programs), not
throughput: 512 rays barely occupy the GPU.

Conclusions:

- The call mechanism can reach about 30-50 us per 512-2,048-ray batch with
  zero-copy host-mapped buffers and a single launch. The 50-100 us target is
  reachable for the call itself.
- It is not reachable through MoonRay's existing `GPUAccelerator::intersect`
  (about 360 us per 512 rays even back to back).
- The rest is traversal. Walk rays only need the subsurface trace-set geometry
  (Embree's trace-set filter rejects everything else), so a dedicated, shallow
  acceleration structure over that geometry, traversed by the conservative
  miss-proof kernel, should take far less than 150 us.
- With about 50-100 us per 512-ray wave (0.1-0.2 us per ray against Embree's
  0.65 us), route A (the in-call, order-preserving wavefront) becomes viable,
  and strict bit-exactness is kept.

## Prototype result: dedicated structure and conservative kernel

`xpu2/sss_gpu_missproof_bench.cc` (standalone; `test-results/sss-missproof`)
uses data dumped by the capture build: the 1,007 meshes walk rays hit
(176,448 triangles, the scene's subsurface trace-set geometry) and 1.15 M
sampled walk rays (every 8th) with Embree's answers. It builds a binned-SAH BVH
(4 triangles per leaf, 115,697 nodes, 0.3 s on the CPU), uploads it once, and
runs an any-hit kernel (compiled with NVRTC) through zero-copy host-mapped ray
and result buffers:

- boxes and the interval are enlarged by margins scaled to coordinate size;
- triangles: a float Moller-Trumbore test rejects clear misses and accepts clear
  hits; anything within a tolerance of an edge or the interval goes to a
  double-precision point-to-triangle distance test;
- non-finite rays (192 of the sample have NaN) and rays exceeding 4,096 node
  visits are reported as candidates without further work, so no ray can
  run unbounded (one NaN ray once ran 4.3 s, past Windows' 2 s GPU timeout,
  and the driver reset the GPU).

| Batch | Median call | p90 | Per ray | Missed hits | Extra candidates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 us | 52 us | | 0 | 0.5% of misses |
| 32 | 75 us | 93 us | 2.3 us | 0 | 0.6% |
| 512 | 81 us | 108 us | 0.16 us | 0 | 0.8% |
| 2,048 | 107 us | 141 us | 0.05 us | 0 | 0.8% |
| 8,192 | 198 us | 265 us | 0.02 us | 0 | 0.8% |

(The double-precision test alone ran 310 us per 512-ray call with 36% extra
candidates, from an over-loose interval-end check that caught the walk's own
starting surface.)

No sampled ray had a GPU miss where Embree hit. This is empirical, on one scene;
a production version needs a written error bound for each tolerance (the float
differences involved are exact or nearly so here, because nearby vertex and
origin coordinates subtract exactly).

Applying Embree's default trace-set rule (material ids) in the kernel was not
safe: 392 of 168,765 Embree hits (0.23%) landed on faces whose material, as I
reconstructed it, differs from the ray's. Leave the trace set to Embree; the
superset costs only the 0.8% extra candidates.

### What it means for the frame

A 512-ray wave costs about 81 us on the GPU, plus Embree for its hits (15% of
rays at 0.52 us) and its rare extra candidates, against about 340 us of Embree
work today: roughly 2.8 times faster walk tracing in full waves. Break-even is
near 110 rays per wave, so smaller waves stay on Embree.

From the wave-size histogram: 4.2 M walk rays sit in waves of 257-512, 1.75 M
in 129-256, and 3.3 M in smaller waves. Estimated saving at one thread: about
1.6 s from the largest waves and 0.3 s from the next band, about 1.9 s of the
39.5 s frame (about 5%), before the wavefront's own bookkeeping. The ceiling is
set by two things this prototype does not remove:

- hits still need a full Embree trace (safe shortening is unsolved), and
- in-call waves are at most 512 rays; larger pooled batches (0.02-0.05 us per
  ray) would require route B and its pixel-sum order change.

Next, if pursued: the shadow-mode exactness harness (plan step 1), then the
in-call wavefront with Embree only (step 2), and then this kernel behind a
batch-size threshold (steps 3-4), measured at 1 and 12 threads.

## Steps 1-2 result: exactness harness and Embree-only wavefront

`xpu2/rndr_sss_wavefront.py` with `xpu2/sss_wavefront.h` (isolated
`librendering_rndr.dll`, built from the native tree by `tools/xpu2_geom_build.py
--dll rndr`). `PathIntegrator::computeRadiancePathTraceSubsurface` keeps its
name; its original body and a wavefront body become two lambdas, chosen by
`XPU2_SSS_WAVEFRONT`:

- unset or `0`: MoonRay's original loop;
- `1`: the wavefront. Every walk of the lane is a state machine; each wave
  prepares one ray per active walk, traces them with Embree in walk order, and
  consumes the results. A walk that exits keeps pt, pOut, nOut, dNdx and dNdy;
  after all walks finish, exits are shaded in the original walk order;
- `shadow`: the original runs for real; the wavefront then runs from the same
  starting state (Subpixel, sequenceID, subsurface AOV, AOV array) into
  scratch copies, and every output is compared bit for bit. Subpixel is
  compared by value and by the deferred nodes' numeric content, since its
  pointers are arena addresses.

Results on the Modo scene (200x200, 16 samples, one thread):

- Shadow mode, XPU: 82,195 subsurface calls, 1,352,905 walks, 989,363 waves;
  no difference in radiance, subsurface AOV, AOVs, sequenceID, Subpixel values
  or deferred-node content in any call. (Intersection ids are handed out in a
  different order, and nothing observable depended on it.)
- Scalar mode (repeatable), 100x100, 4 samples: the final image is identical
  word for word across MoonRay 0.4.2's own DLL, the new DLL with the switch
  off, and the new DLL with the wavefront on (84,577 walks through it).

Timing, XPU, Embree only, three alternating pairs (MCRT):

| Threads | Switch off | Wavefront on | Medians |
| --- | --- | --- | --- |
| 1 | 43.37 / 41.94 / 42.06 s | 41.62 / 41.45 / 41.22 s | 42.06 -> 41.45 s (1.5% faster) |
| 12 | 10.27 / 8.64 / 8.81 s | 8.35 / 8.08 / 7.46 s | 8.81 -> 8.08 s (8% faster, noisier) |

Every pair was faster, and Init Intersection fell from about 2.1 s to 1.7 s at
one thread for identical work. The likely cause is locality: a lane's walks
start at the same point, so stepping them together reuses the same BVH nodes,
triangles and attribute data from cache (an interpretation, not measured).

### Merged lanes

`xpu2/rndr_sss_merged.py` (on top of the per-lane module) adds
`XPU2_SSS_WAVEFRONT=merged` and `merged-shadow`. When every active lane of a
`CPP_computeRadianceSubsurface` call is random-walk subsurface, the call runs
three phases through `computeRadiancePathTraceSubsurface`, chosen by a one-shot
thread-local phase (read and cleared on entry, so nested calls run normally):

- Prepare, per lane in lane order: the function's setup and walk creation,
  kept in a per-lane job (heap allocated, alive until Finish);
- Waves, once: all walks of all lanes step together; before each trace the
  thread's subsurface trace set is set to that ray's lane, since Embree's
  filter reads it;
- Finish, per lane in lane order: the shared Bsdf is advanced exactly as the
  original loop does (label ids, an appended null Bssrdf, the lane's volume
  subsurface), the AOV array is re-initialised, exits are shaded in walk
  order, then the original epilogue, AOV queueing and result writes.

Otherwise (a diffusion lane present) the call runs MoonRay's original loop.
`merged-shadow` runs the original loop for real, capturing each lane's
radiance, subsurface AOV and AOV array, then replays the merged path on scratch
copies (Subpixel, PathVertex, sequence ids, a separate Bsdf, an AOV buffer, no
queueing) and compares lane by lane.

Shadow result, Modo scene, XPU, one thread: 82,195 lane comparisons in 28,488
merged calls, no fallback, and no difference in radiance, subsurface AOV, AOVs,
sequence ids, Subpixel (values and deferred-node content) or PathVertex. The
merged path traced all 9,215,173 walk rays in 391,144 waves:

| Wave size | Waves | Rays |
| ---: | ---: | ---: |
| 1 | 179,721 | 179,721 |
| 2 | 49,954 | 99,908 |
| 3-4 | 39,764 | 134,738 |
| 5-8 | 40,142 | 262,295 |
| 9-16 | 17,914 | 214,612 |
| 17-32 | 15,948 | 372,658 |
| 33-64 | 17,036 | 841,369 |
| 65-128 | 11,482 | 1,114,878 |
| 129-256 | 9,130 | 1,754,119 |
| 257-512 | 10,053 | 4,240,875 |

Timing, XPU, Embree only, three alternating pairs (MCRT):

| Threads | Off | Merged | Medians |
| --- | --- | --- | --- |
| 1 | 41.90 / 42.34 / 43.64 s | 42.25 / 42.18 / 43.16 s | 42.34 -> 42.25 s (no change) |
| 12 | 9.94 / 8.63 / 7.62 s | 9.38 / 7.74 / 10.69 s | inconclusive (about 15% spread, one slow merged run) |

Init Intersection falls from 2.09 s to 1.70 s at one thread, as with the per-lane
wavefront, but the per-lane version's 1.5% frame gain is gone. A likely cause is
the working set: 512 walks of about 1 KB of state each (intersection, samplers,
ray) exceed the CPU's L2 cache, where 64 walks fit (an interpretation, not
measured). On the CPU alone merged lanes buy nothing; their purpose is the GPU
step, which needs the larger waves. The 12-thread comparison needs more
repeats before any conclusion.

The earlier per-lane wavefront reordered work inside one lane only (up to 64 walks per wave
at camera depth); merging all lanes of a call (up to 512) is the next
restructuring, followed by the GPU miss-proof kernel behind a wave-size
threshold.

## Steps 3-4 result: the GPU miss proof inside the merged wavefront

Status (2026-10-19): shelved by the user's decision. The code stays behind
`XPU2_SSS_GPU`, which is off unless set to `1`; with it unset the merged
wavefront traces every ray with Embree.

Rebased onto the XPU2 candidate (`tools/xpu2_rebase_build.py --base
test-results/xpu2-disabled-route-build`, which now also compiles extra
translation units). `xpu2/rndr_sss_gpu.py` (on top of the merged module),
`xpu2/sss_gpu.h`/`.cc` and `xpu2/sss_missproof.cu` (compiled to PTX by
`tools/xpu2_sss_ptx.py`):

- `XPU2_SSS_GPU=1`: in the Waves phase, a wave of at least
  `XPU2_SSS_GPU_MIN` rays (default 128) goes to the GPU first. Rays it proves
  cannot hit anything skip Embree and count as misses (the intersection-ray
  statistic is still counted). After a miss the walk reads only the ray's
  origin, direction and its own sampled distance, which Embree does not change
  on a miss, so nothing downstream differs. Everything else goes to Embree as
  before.
- The structure: on first use, every mesh of MoonRay's Embree accelerator is
  read once (its BVHUserData list, through an explicit-instantiation accessor,
  no header change). Quads go in with both diagonals. Each walk key (the ray's
  material id and subsurface trace set) gets its own BVH, from the faces
  Embree's `bssrdfTraceSetIntersectionFilter` can accept for that key (same
  expressions: faces without an assignment, faces in the trace set, or faces
  whose material id matches). Masks and the other filters only remove hits, so
  they stay with Embree. Instances, curves, spheres, volumes or motion blur
  turn the GPU path off for the scene, with the reason in the report.
- Each wave: one launch through per-thread zero-copy host-mapped buffers; each
  ray record names its key's structure.
- `XPU2_SSS_GPU_VERIFY=1` still traces the proven rays with Embree and counts
  hits; `XPU2_SSS_GPU_REPORT` writes a CSV.

Exactness, Modo scene, XPU, one thread, 200x200, 16 samples:

| Run | Lane comparisons | Differences | Rays to GPU | Proven misses | Proven misses Embree hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole-scene structure (1.5 M triangles), merged-shadow | 82,195 | 0 | 6,181,382 | 5,314,250 | |
| Whole-scene structure, verify | 82,195 | 0 | 6,191,489 | 5,322,449 | 0 |
| Per-key structure (149,184 triangles, 1 key), verify | 82,195 | 0 | 6,157,471 | 5,311,635 | 0 |
| Per-key structure, merged-shadow | 82,195 | 0 | 6,154,859 | 5,310,322 | |

(Test results: `test-results/sss-gpu/allgeom`, `test-results/sss-gpu/keys`.)
The material rule that looked unsafe in the prototype (0.23% of hits) is exact
here, evaluated in-process with the filter's own calls; the prototype's
reconstruction from dumped data was at fault.

Speed: the GPU path is slower. About 86% of the GPU's rays are proven misses,
but a call costs far more in the renderer than in the benchmark:

| | Calls | Kernel wall time, summed | Per call |
| --- | ---: | ---: | ---: |
| Benchmark, back to back, 512 rays | | | 81 us |
| Renderer, 1 thread (whole-scene structure) | 20,746 | 11.08 s | 534 us |
| Renderer, 1 thread (per-key structure) | 20,757 | 10.14 s | 489 us |
| Renderer, 12 threads (per-key structure) | 20,765 | 3.75 s | 180 us |

The reason: the GPU stays in its idle power state. During the 1-thread render
`nvidia-smi` shows P8 throughout, 210 MHz SM and 405 MHz memory against
2,100 / 9,751 MHz maximum; at 12 threads it is mostly P8, with short bursts
in P2. The proof's load (and XPU's own OptiX calls) is too light and sporadic
for the driver to raise clocks, and at a tenth of the clock the kernel takes
about ten times as long. Two attempts from inside the renderer did not help:

- an empty kernel every 0.5 ms on its own stream (`XPU2_SSS_GPU_KEEPALIVE_US`):
  no change (kernel 9.8-9.9 s, still P8);
- a kernel that keeps one warp busy 300 us of every 500 us
  (`XPU2_SSS_GPU_KEEPALIVE_SPIN_US`): clocks rise to P2, 1,965 MHz, but the
  proof waits behind it (652 us per call, MCRT 57.6 s) and the card draws
  152 W instead of 32 W. Not usable.

Timing, XPU, three alternating rounds on one runtime (MCRT):

| Threads | Off | Merged | Merged + GPU |
| --- | --- | --- | --- |
| 12 | 8.49 / 7.74 / 10.75 s | 8.03 / 7.72 / 8.27 s | 10.06 / 9.58 / 9.71 s |
| 1 | 43.13 / 42.07 / 42.30 s | 42.73 / 42.32 / 42.76 s | 52.43 / 51.25 / 51.81 s |

Medians: 12 threads 8.49 / 8.03 / 9.71 s; one thread 42.30 / 42.73 / 51.81 s
(`test-results/sss-gpu/timing`). At one thread the 9.1 s loss matches about
10 s of kernel waiting less the Embree misses it saves.

At 12 threads the GPU runs cost about 1.6 s more than merged, more than the
0.3 s of kernel waiting per thread. The rest is likely the one-time build
(about 0.4-0.5 s for reading the meshes and building the key's BVH, under a
global lock that stalls every thread on the first wave), the lock taken on
every wave, and sharing the GPU with XPU's own queue (interpretations, not
measured separately).

With the GPU's clocks raised by the user:

- "Prefer maximum performance" for moonray.exe (NVIDIA Control Panel, per
  program) had no visible effect: still P8, 210-420 MHz SM, kernel 9.2 s,
  MCRT 52.3 s at one thread.
- `nvidia-smi -lgc 1500,2100` (admin, set by the user, undone with `-rgc`)
  holds the SM clock at 1,500 MHz, but the memory clock stays at 810 MHz
  (P5) for most of the render (241 of 268 five-second samples; P2 at
  9,501 MHz in the rest).

Timing under `-lgc 1500,2100`, three alternating rounds (MCRT,
`test-results/sss-gpu/timing-lgc1500`):

| Threads | Off | Merged | Merged + GPU | Kernel, summed | Per call |
| --- | --- | --- | --- | ---: | ---: |
| 12 | 8.79 / 8.87 / 8.05 s | 9.04 / 9.25 / 8.91 s | 8.90 / 10.02 / 12.80 s | 3.1-3.7 s | 150-180 us |
| 1 | 43.27 / 42.56 / 42.66 s | 42.90 / 42.39 / 42.48 s | 47.21 / 46.73 / 46.40 s | 4.0-4.2 s | 195-200 us |

Medians: 12 threads 8.79 / 9.04 / 10.02 s; one thread 42.66 / 42.48 / 46.73 s.
The kernel time halved (10.1 s to 4.1 s at one thread), but at about 0.66 us per
GPU ray it is no cheaper than Embree's 0.65 us per walk ray, so the GPU path is
still 4.3 s (10%) slower at one thread and slower at 12. The benchmark's 81 us
per 512 rays was measured with the memory clock up; with memory held at
810 MHz the traversal stays memory-bound. Even at benchmark speed the prize
was estimated at about 1.9 s of a 42 s frame (about 5%) at one thread.

What would change the outcome:

1. Memory clock as well as SM clock (for example `nvidia-smi -lmc`, where the
   card supports it), so the kernel approaches the benchmark's 81 us per call.
   That is the user's system setting, not something the renderer should
   force, and artists' machines would not have it.
2. Build ahead of the frame: read the meshes and build each key's BVH when the
   scene is built (the scene-index resource, built just in time), not on the
   first wave under a lock.
3. Larger batches (route B, across calls): fewer, fuller launches, but pixel
   sums change in the last bits, which the parity rule rules out today.

## Plan

0. Latency study (measure only). Done; see above. A microbenchmark on this GPU: empty launch and
   synchronize, pinned against pageable copies, back-to-back against idle-gap
   calls, with CUDA events to split the 0.35 ms. Outcome: whether route A can
   reach about 50-100 us per call.
1. Exactness harness first. A "shadow mode" that, for each subsurface call, runs
   both the original sequential walks and the new path and compares every lane's
   radiance, AOV and exit data bit for bit, then uses the original. This tests
   exactness directly, independent of XPU's run-to-run variation.
2. In-call wavefront with Embree only (no GPU), behind a switch. Exact by the
   harness; also measures its own overhead.
3. Conservative GPU miss-proof program (OptiX custom intersection, enlarged
   triangles, subsurface trace-set geometry), checked by the harness and by
   counting any GPU miss where Embree hits (must be zero).
4. Turn on GPU-guided batches above a size threshold set from step 0's latency.
   Measure at 1 and 12 threads on the Modo scene and the example scenes.
5. Only if step 0 fails and the user accepts last-bit pixel-sum differences:
   design route B.

Expected prize if batches are cheap: walk-ray Embree time is about 6 s of the
39.5 s frame at one thread; GPU-proven misses would remove most of the 5.4 s
spent on misses, while hits (0.7 s), exit lighting and the scalar bounces stay
on the CPU. A realistic ceiling is on the order of 10% of this frame at one
thread, more with the subsurface occlusion rays later.

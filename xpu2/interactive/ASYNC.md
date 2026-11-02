# XPU2: the asynchronous renderer

XPU2 renders with the CPU and the GPU working on the same image at the same time, neither waiting for the other. This
is the architecture of the Interactive xPU Renderer Spec (21 October 2026), built on the prototype's scene, sampling,
material, texture and fallback code. The earlier synchronous pipeline (`deadline_pipeline.h`) stays as the parity
build: exact shared triangle arithmetic, fixed ownership and byte-repeatable images (spec ACC-08, CPU-11).

## How a frame runs

- **Work grid** (`async_grid.h`, spec §7). The image is cut into 64×64 cells, seeded as a CPU/GPU checkerboard. Each
  cell holds its unstarted work as 16-byte chunks: one 8×8 tile and a range of sample indices, ordered by sample range
  so a cell converges evenly. A chunk is claimed exactly once, by one atomic on a cursor in host memory.
- **CPU lane** (`async_cpu.h`, §11). Persistent, pinned workers claim their own chunks by the cellular rules (§8):
  finish the cell they are in, step to the same-owner neighbour with the most unclaimed work, steal from the hottest
  opposite-owner neighbour, then drift to the hottest cell anywhere. Embree traverses native triangles in robust mode.
- **GPU lane** (`async_gpu.h`, `async_gpu.cu`, §10). Fixed-size OptiX launches (65,536 threads) over hardware
  triangles. Every thread claims one sample at a time from its launch's range with one atomic; neighbouring claims
  are neighbouring pixels of the same sample, so warps trace coherent rays. Results accumulate in a device film in
  double precision. The program is built twice, for uniform and for adaptive frames: OptiX links a pipeline's programs
  into one kernel, and the adaptive paths cost a uniform frame a third of its GPU rate in registers. Geometry in motion
  has a key a time step and is traced at the sample's time (`optixGetRayTime`); the pipeline is compiled for motion in every
  scene, about 2 to 3% on a still one, where a second pipeline would double the compile. `XPU2_GPU_MOTION_OFF` defers motion
  and points to the CPU lane again, to compare.
- **Mailbox and broker** (`async_render.h`, §8-9). The feeder thread, pinned to its own core, is the GPU's broker: it
  claims chunks for the GPU (its own cells first, then the CPU's hottest) and hands them over through a ring in
  device memory, with ownership passing at publication. It keeps two launches queued, so the GPU never waits for
  the host; it polls events, so the host never waits for the GPU (ARC-01). Launches are sized from the GPU's rate,
  timed on the device (the donation forecast, CA-08); near the end of a frame the remaining work is split so both
  devices finish together (the lookahead).
- **Textures** (§13). The files are the system of record. Host RAM holds the coarser levels of every texture and each
  level-0 tile once it is first read, decoded on the spot, so the CPU never misses; under memory pressure the oldest
  of those tiles are let go (see Decided, below). VRAM holds each texture's mip levels from 1 up, for the whole session, and a bounded
  cache of fine-detail (level-0) tiles, loaded on demand and evicted least recently used. A GPU sample that needs a
  tile that is not resident parks in a bounded queue on the GPU, asks for the tile once, and runs again on the GPU,
  with the same random numbers, after the tile has landed. The host sizes launches so the queue cannot overflow.
- **Ownership** (CA-10, CA-12). Cells keep their owner from frame to frame and flip towards the device that rendered
  most of them, at most 8 a frame, with a 16-frame cooldown and a 64-frame freeze after thrashing.
- **Correctness.** Every sample has a device-independent identity (pixel, index, frame), so it does not matter which
  device computes it. The shared integrator (`async_integrator.h`) is the parity pipeline's estimator, operation for
  operation. A started sample is never dropped (ACC-03). At quiescence the two films merge and every pixel's sample
  count is audited (ACC-07).

## Adaptive sampling

The kit's frames sample by cellular generations, on the host, with the CPU taking part (`INDEXER.md`,
`RenderOptions::cellular`). What follows is the error-tree sampler they are checked against, which the benchmark and
`XPU2_SESSION_ADAPTIVE=trees` still run.

A scene with `sampling_mode` 2 renders with MoonRay's adaptive sampler (`async_adaptive.h`, adapted from MoonRay's
`rndr/adaptive`): the same error measure (the luma of the difference between the even- and odd-index half-images,
over the square root of the pixel's luma, plus the alpha difference), the same 4×4 overlapping regions, each with
its error tree (Dammertz et al.'s hierarchical stopping condition), and the same pass schedule: one sample, then 4,
6, 8, … more. Passes below `min_adaptive_samples` render every pixel; each later pass renders only the pixels its
trees still find unconverged, filled to their bounding rectangle in each 8×8 tile.

- **It fits the work grid.** A pass is chunks over the tiles that still have active pixels, each carrying its tile's
  mask, appended to the tiles' cells; unconverged cells are the ones under pressure. A pixel's sample indices stay
  contiguous from pass to pass, as MoonRay keeps them, so its identities do not depend on which pass or device ran it.
- **Fresh trees for every pass.** Each pass's masks come from trees that have seen every pass before it, as each of
  MoonRay's tiles reads trees its region rebuilt on finishing the pass before. That makes a frame a chain of about 30
  dependent passes, so it is bound by the chain's latency, not by sample throughput.
- **The chain runs on the GPU** (`async_adaptive.cu`, `GpuLane::launchChain`). The host queues a whole frame at once:
  the uniform stage as one render, then for every adaptive pass four CUDA kernels and an OptiX render, in stream order.
  The kernels work out every pixel's error from the device film; rebuild the region trees, one block a region, from a
  summed-area table, a level of nodes at a time, with split points found by bisection; give every tile its mask and
  every active pixel its sample-index base; and list the active tiles in the work grid's cell order. The render then
  runs those tiles' pixels. Once the trees converge, or a pass comes out empty, the rest of the chain does nothing.
- **The same trees on both sides.** The host and the GPU share the tree arithmetic (`async_adaptive_core.h`), with
  every product and sum the CUDA compiler could fuse into one multiply-add written out rounded, so from the same
  errors both build the same trees. Passes made on the host (`RenderOptions::deviceChain` false) give every pixel
  exactly the samples the chain does; the adaptive benchmark checks this.
- **Texture misses.** A chain render that parks a sample stalls the chain: its later steps do nothing, the host lands
  the tiles and reruns the parked samples by ordinary launches, then queues the chain again after that step. It
  happens on a session's first frame, which loads its fine-detail tiles.
- **Who samples.** With a GPU and the chain, the GPU takes every sample of an adaptive frame: a CPU chunk would sit on the chain.
  Without one, the CPU lane samples and the host works the errors out from its film and rebuilds the trees on a pool.

The same target need not give the same error as MoonRay's: the trees are MoonRay's, but the estimate reads XPU2's
own samples (Sobol camera and first-bounce dimensions, against MoonRay's PMJ02 and Poisson-disk tables), which split
into even and odd halves differently. Per sample, the two renderers' noise is the same.

## Results

See [ASYNC_RESULTS.md](ASYNC_RESULTS.md).

## Build and run

```powershell
python xpu2/interactive/build_async.py --source $toolchainRoot --runtime $referenceRoot --output test-results/my-async-build
python xpu2/interactive/run_async_benchmark.py --build test-results/my-async-build/build.json --output test-results/my-async-benchmark
python xpu2/interactive/run_async_adaptive_benchmark.py --build test-results/my-async-build/build.json --output test-results/my-async-adaptive
```

`async_grid_test.cc` checks the scheduler's invariants (VER-5): every chunk claimed exactly once under 12 workers and
the broker, every pixel given exactly its samples, the cellular rules, the checkerboard and the flip limits.

## First load and re-cuts

A heavy scene's first picture comes from a coarse first pass (`coarse_geometry.h`): tessellation at a quarter of the
resolution, dense meshes simplified, shown in the pictures and never in the film, while the full plan is made on a
thread of its own (`RefineJob`, `xpu2_session.cc`). It is used for a scene with more than about 2 M tessellated
triangles (`Scene::heavyTessellation`), in realtime for the session's first preparation alone. A camera moved meanwhile
is followed on the coarse geometry, and the full plan takes the camera as it then is; any other edit waits for the full
plan and keeps what it can. Dense meshes that are not tessellated gain nothing from it: simplifying them costs more than
preparing them.

An edit that changes a mesh's shape, such as the kit's realtime re-cut, prepares the scene again but keeps all that did
not change: meshes by their signature, instancers' shared meshes among them, and a baked volume's grid while its mesh,
maps, place and render space are the same.

| The car, 7.8 M triangles, 960×540 | Before | Now |
| --- | --- | --- |
| One mesh re-cut, edit to render | 6.3 s | 0.73 s |
| Making its meshes, first load | 6.9 s | 2.7 s |

| Heavy displaced scene, realtime | First picture | Finished |
| --- | --- | --- |
| Tessellation not cached, coarse pass off | 15.1 s | 18.3 s |
| Tessellation not cached, coarse pass on | 2.6 s | 20.6 s |
| Tessellation cached, coarse pass off | 5.7 s | 8.9 s |
| Tessellation cached, coarse pass on | 2.6 s | 11.8 s |

Each edit case gave the same image as the scene rendered whole by a fresh session. The car's numbers use its export with
one substitute texture, through the session harness; a re-cut in Modo also costs the kit's own cut.

## Limits a scene can meet

- **Rays a path has room for.** A path's pending rays have 32 places (`MaxPending`). A hit's bounces are thinned evenly to
  the room under a reserve of 16, and past the reserve a child is kept by chance, its share of light divided by that
  chance: the image the same on average, a little noisier there. Material Samples of any count render.
- **Subsurface Samples.** At most 4 x 4 exit samples a hit: a higher setting takes 16 and says so in the log. The image is
  the same on average and as noisy as 4 x 4, so that more samples at a hit have to come from Pixel Samples.
- **Pixel filters.** Box, the two B-splines, and the kit's Gaussian, Triangle and Blackman-Harris, warped as its MoonRay
  warps them; widths up to 64.
- **Texture overflow mode.** Where a scene's load runs out of memory on the machine or the card, or a texture budget, the
  frame is made again with its textures compressed: BC6H, a byte a texel at any size, encoded by Intel's ISPC Texture
  Compressor as tiles are read and decoded texel by texel on both devices (`bc6h_core.h`, generated from the encoder's own
  layouts by `make_bc6h_core.py`, and matched against OpenImageIO's decoder on all fourteen modes). A map with negative values
  stays as it is, BC6H being unsigned. If that is not enough, textures are limited to half the size, down to 256 pixels a side.
  On the car the card's textures went from 3.7 GB to 311 MB, the frame took 7% longer, and the picture differs by 0.0025 a pixel
  on a mean of 0.60. The preview says so
  (`@@MODO_XPU2 NOTICE`). It also starts where every decoded tile has been let go, RAM is still short and the card is
  short too, with a size left to go to. It holds until the scene is loaded anew. Finer levels are neither decoded nor sent; `XPU2_SESSION_CARD_FREE_MB` stands in
  for a full card in tests. A failure once a frame is rendering still goes to MoonRay XPU.
- **A texture tile that cannot be read.** The frame fails with the file, the tile and OpenImageIO's reason, and the scene goes
  to MoonRay XPU. Every thread joins before the failure goes on, and the CPU lane's workers stop before the frame's grid and film
  go: thrown past running threads it ended the process ("terminate called"). `XPU2_SESSION_FAIL_TILE=N` fails the Nth read, for
  tests. Why the one crash seen, the car's second Render on 31 October, would have failed a read is not known.
- **Letting tiles go.** Free RAM below a tenth of the machine's (2 GB at least), or the session's commit past 92% of its ceiling,
  lets the oldest decoded tiles go, but 1 GB of them stay, tiles too few to matter are left alone, and a letting go that did not
  help waits twice as long before the next, up to a minute. At 85%, every tile, every quarter of a second to two, a car's render
  stalled reading them again: its commit passes that with the graphics driver's alone, RAM to spare.
- **Moving lights on the card.** Each launch thread holds up to 16 of them at its sample's time, worked out as a path
  reaches each (`GpuBackend::movedLight`): about 440 MB of the card, for 65,536 threads, in a scene with a moving light. A path
  holds one light at a time, so any number of them can move. Large moving meshes against the card's budget have not been tested.
- **A card that fills up.** The fine-detail cache is sized from the card's free memory when the scene is prepared, and made
  again to fit where, as a frame starts, the card has come to hold much less (`GpuLane::fineCacheTooBig`).

## Decided, 30 October 2026

Each of these was on the list of what was left; each was measured or read against the renderer as it stands.

- **The adaptive chain as one CUDA graph: not built.** The kit's frames make their passes on the host, by sampling
  generations (`XPU2_SESSION_CHAIN` is off by default, `xpu2_session.cc`). The GPU chain runs only in the adaptive
  benchmark and under that preference, so a graph would speed up a path no artist's render takes.
- **Interactive frame mode (§15): shipped in the preview session,** without reprojection or frame deadlines.
  `xpu2_session.cc` publishes the merged film as it renders, stops a frame when the scene changes, and shows it through
  the kit. Reprojection is left out on purpose: it would show stretched pixels after every move, the kind of brief
  glitch the half-resolution start was removed for. The session has OptiX denoising; the 0.5 kit does not use it in
  realtime. Deadlines have not been needed: Realtime Sample % and the Realtime Adaptive Sampling Cap bound a frame.
- **Spill under host memory pressure (MEM-06 to MEM-09): done, as letting tiles go** (see Limits a scene can meet). Level-0 tiles
  are decoded when first read, and those let go are decoded again from their files if asked for again; the card's copies stay.
  `tex::spill` guards every host read, so a tile is freed only when no reader can hold it. `XPU2_SESSION_SPILL_MB` holds the tiles
  to a size, for tests: every tile let go, continuously, gave CPU-only images byte-identical and GPU-only images within the
  run-to-run difference. The coarser levels are not let go.
- **Mapped zero-copy rings (MBX-01): measured, not adopted.** The chunk ring in pinned host memory, mapped for the
  device, against stream-ordered copies, 12 threads: gallery-dense 392 against 388 ms, workshop-detail 609 against
  611 ms, the difference within the noise. Copies stay, as the simpler and the one that does not read across the bus.
- **SIMD batches on the CPU lane (CPU-03): measured for camera rays, not adopted.** Camera rays traced eight at a time
  through Embree's packets, the only coherent rays of a path, CPU only: 0.3% and 1.3% less time, and images no longer
  byte-identical to one-ray tracing. Vectorizing the rest would mean rewriting the integrator around queues of rays.

## Not done yet

- **Coverage.** OpenVdbMap, deep output and camera medium materials are not native; `FEATURE_PROGRESS.md` holds the
  list, and what is not native routes to MoonRay XPU.
- **GPU modules linked by the scene's families.** All nine modules link into every pipeline.
- **Letting geometry and coarser texture levels go under pressure.** Past its ceiling the session still goes to
  MoonRay XPU, the machine left alone.

# XPU2's scene indexer: classification, routing and incremental updates

The scene index (`../indexed_scene.h`, `../rdl_scene_index.h`, `../rdl_replica.h`) captures every RDL object and
attribute of a Modo session incrementally, and the router (`routing_index.h`) keeps a decision per entity, redone only
for what an edit touched. The classifier (`entity_classes.h`) gives each object its device class, shown in the
session's log. The plan builder (`cpu_scene.h`, `cpu::prepare`) stays the final check of every attribute, refusing a
scene at the first thing XPU2 does not render, and the GPU leaves a whole scene to the CPU for what its programs lack
and no stand-in can take (`async_gpu.h`, `cpuOnly`).

This note began as the plan to finish it. It keeps that plan and what each step came to; the list at the end says
where they stand.

## 1. Classification per entity

Each entity (geometry, instance, material, map, light, light filter, volume, displacement, camera) gets:

| Field | Meaning |
|---|---|
| support | `Fallback` (XPU2 does not render it: the scene goes to MoonRay XPU), `Cpu` (the CPU lane alone), `Both` |
| evidence | the conformance identity: the XPU2 feature and the comparison that verified it (for example `volume.vdb:vs1`) |
| families | the GPU feature families it needs (DwaBase, layers, hair, glitter, toon, subsurface, image maps, procedural maps, curves, points, motion, volumes, light filters, ...) |
| cost | a relative cost per hit on each device, from measured shading times per family |

The rules are the plan builder's admission, made per object so both use one source of truth: the class and attribute
checks, map bindings and their programs, and the GPU capability table (what the GPU programs implement, which is
what `cpuOnly` refuses today, family by family).

A material is classified by the shading families its BSDF tree and map graph use, so a scene with one glitter
material and many DwaBase ones is not judged by the glitter.

## 2. Routing by device

- **GPU programs by family.** The GPU program is built from the families the scene's `Both` entities use. With the
  map interpreter split into a function per map type and the families moved into separate modules, a scene compiles
  only what it uses, cached per combination.
- **Deferral to the CPU.** An entity classified `Cpu` is in the GPU scene as a stand-in: a bounding primitive (a
  VDB's box, a mesh left to the CPU by the card's budget) or a flag on a light or material. A GPU sample that meets
  one stops and is handed to the CPU lane, which renders that sample whole from its identity (pixel, index, frame).
  Every sample's random numbers come from its identity, so the CPU's result is the one the GPU would have made; no
  path state crosses devices, and the GPU's partial work for it is dropped (its film never sees it). The CPU lane's
  retry queue takes these samples.
- **Cells by suitability.** The work grid's cells start on the device suited to what they see: a cheap first-hit pass
  (the camera rays of one sample a tile) gives each tile the entities it sees first; a tile whose entities are `Cpu`
  or cheaper on the CPU starts CPU-owned. Ownership then follows the deferral rate: a cell whose GPU samples mostly
  defer flips to the CPU (the existing ownership flips, with deferrals counted as the CPU's).
- **Scene mode.** Any `Fallback` entity sends the scene to MoonRay XPU, with the entity and the reason, decided as
  the edit arrives instead of after a failed plan.

## 3. Incremental updates end to end

The router's change set (entities added, removed, changed, and those affected through dependencies) goes to the
renderer with the frame:

- unchanged meshes, tessellations, grids, images and light tables are reused (as today, by signature), and the
  renderer is told which ones changed instead of finding out;
- the CPU's Embree scene and the GPU's acceleration structures are refitted or rebuilt per changed mesh (instances:
  their transforms only), not rebuilt whole;
- material, light and filter records are rewritten in place where their layout allows;
- the RDLB file is written only for MoonRay's fallback, not on every edit.

## 5 in detail: the GPU program in modules

What the compile looks like now (build 74, OptiX 7.6, optimization level 3): two modules of about 10 MB of PTX each
(uniform and adaptive frames, nearly the same code), compiled in parallel, each one task on one thread for over half
an hour and about 13 GB. Marking the path tracer's big pieces `__noinline__` shrank NVRTC's work and the PTX's largest
function (integratePaths 30.8k to 12k instructions), but not OptiX's: it compiles a module's call graph as a whole.

1. **One program** for uniform and adaptive frames (the launch's adaptive flag): half the compile's CPU and memory,
   the same wall time (done, build 76).
2. **Test builds at a low optimization level** (`XPU2_OPTIX_LEVEL`, its own shader mark): GPU correctness tests
   without the hour.
3. **Families as callables in their own modules.** OptiX compiles each module on its own task, in parallel, and never
   inlines a callable into its caller. A direct callable cannot trace rays (OptiX 7.6), so the split follows what
   traces:
   - direct callables (shading, no rays): the map interpreter (a callable per map family), the material's
     parameters (resolveDwaTree, blendParameters, the hair and glitter blends), the BSDF's sample and eval per lobe
     family (base, hair, glitter, toon), the camera's ray (cameraSampleAt), the material AOVs;
   - continuation callables (they trace): subsurface (diffusion and random walk), volumes (their radiance and
     transmittance);
   - the raygen module keeps the path loop, light sampling and the traces.
   On the CPU the same calls are plain calls. A scene's pipeline links only the modules its families use (the
   indexer's families, step 1), each module compiled once and kept in OptiX's cache, so a new scene compiles at most
   the modules it adds.

   Done so far, in PTX instructions (whole, build 76 was 238k in one module):

   | Module | Callables | Build 78 | Build 81 | Next (written) |
   |---|---|---|---|---|
   | path tracer | (raygen, misses, hits) | 145k | 60k | 60k |
   | material | a hit's Bsdf, a material's presence | 85k | 85k | 12k |
   | maps | the map interpreter | 57k | 57k | 57k |
   | lights | shapes sampled, intersected, evaluated | - | 12k | 12k |
   | lobes | lobes evaluated, sampled, their albedo | - | 36k | 36k |
   | volumes | volume radiance, subsurface, shadow transmittance (continuation callables) | - | 50k | 50k |
   | DwaBase resolve | a tree's layers blended | - | - | 29k |
   | DwaBase lobes | a resolved material's lobes, glitter | - | - | 51k |

   OptiX at build 79 (four modules, the path tracer 134k): 17 of 19 compile tasks in 12 s, then the path tracer's
   module alone on one thread, past 30 minutes and 25 GB, as the whole program did before: the cost is one large
   module's, so the split continues until no module is large. Two things learned on the way: a module of callables is compiled
   relocatable (`-rdc`), or NVRTC drops its entry points; and with the materials gone NVRTC inlined 34 helpers it
   had kept as single calls (the module grew to 292k), so the big helpers are marked out of line (X2_CALL, now
   defined beside X2_HD in sample_stream.h).

## 6 in detail: where an edit's time goes now

An edit makes the plan again (meshes, grids, images and light tables reused by signature), then the CPU's Embree
scene whole, the GPU's every acceleration structure (GpuLane::prepare builds a GAS per mesh, reusing none) and a new
renderer. First steps: the GPU lane and the CPU scene keep each mesh's acceleration structure and uploaded buffers by
its MeshData (the plan builder hands an unchanged mesh's same shared object to the next plan), rebuilding only the
instance structures; the renderer keeps its ownership and lane rate across an edit of the same image size.

## 7. Scheduling: the whole picture first, important parts, rights of passage

What a preview showed before (build 86, the Modo scene of 264 classes at 1280x720, 16 samples): every chunk 4
samples, and a CPU worker finishing a cell before it moved on, so parts of the picture went to 16 samples while the
rest was black, and the whole frame was seen only once a quarter of its samples were in. The grid now (session 90):

- **Progressive order.** A uniform frame's sample ranges grow 1, 1, 2, then the chunk size, and no chunk is handed to
  a device past the least-sampled range of the cells it works without a right of passage (its frontier: its reserved
  cells and the open ones), so the faster device is never held back by the slower one's cells. The frontier moves
  when a range is handed out, not when it is done. The pictures fill the pixels no sample has reached from the finest
  level of a pyramid of averages that covers them (display only, as MoonRay's coarse passes).
- **Important parts first** (MoonRay's, `tools/port_important.py`): the pictures score each cell by the mean difference
  of its pixels' tones from their neighbours'; the cells above the mean stay their owner's (the indexer's choice,
  section 2) and may run a range ahead of the frontier; the rest are open to either device, the most important first.
  Scored at most once a second (without a watch, from pictures made for it alone).
- **Rights of passage.** A device enters a cell the other has reserved when that cannot hold the frame up: the CPU
  when one of its chunks ends well before the GPU would get through its reserved work, or when the GPU has not
  claimed for four launch windows (a latency gap); the GPU when the CPU's reserved work outlasts four of its windows,
  in a cell it would not mostly defer (the indexer's p of at least 0.2). Before the lanes' rates are measured, always.
- **Launches that fill the GPU.** A launch is sized from the GPU's measured rate (about 10 ms of work), the rate
  measured only on launches of 65,536 samples or more. A slow first launch (textures still landing, the Modo scene's
  heavy materials) made the next ones small, small ones were never measured, and the size stuck at a few thousand
  samples, a fraction of the launch's 65,536 threads: the Modo scene ran at 200,000 samples a second. Launches are now
  never below two samples a thread (session 92): that scene 74 s to 20.9 s on the GPU alone, 62 s to 19.3 s on both.
- **Cellular GPU workers.** The GPU's broker takes each launch as shares of 32 workers, each in a cell and moving by
  the same rules as a CPU worker (finish, own reserved cells, open cells, passage; neighbours first). The launch
  itself runs all at once; only where its work comes from is cellular. The outlines show them in their own colour.

## 8. Partial shading

A GPU sample that needs a fine-detail tile not yet resident is finished from the texture's next level up (levels 1
and up are always resident: the half-resolution picture and smaller, as MoonLightIPR halves textures), its tile
asked for as before. It is provisional: its value goes to the GPU's provisional film, not the film, and a record of
it (pixel, index, pass, missing tile, value) to the host. Either lane renders it again in full: the CPU lane at once
(it has every level in RAM), or the GPU once the tile has landed; its provisional value then leaves the provisional
film, and the full sample enters the film and the AOVs. The pictures show the provisional samples; the frame's end
has none left (audited), so the image stays MoonRay's. A provisional sample is not counted in its adaptive pass until
it is done in full, so the error trees never judge a low-fidelity pixel. A drain (Stop) hands out no new chunks but
renders the provisional samples in full first, then ends the frame. Never in the GPU's pass chain.

Compile time: with partial shading the maps module's OptiX compile went from 10 s to 7, then 34 and 40 minutes, its
PTX the same size each time, whether the coarser read was a retry loop, inline in each texel fetch, or a direct
callable. What those builds shared: OptiX calls in the map callable (optixGetLaunchIndex for the thread's mark, then
optixDirectCall). The textures now carry the thread's mark from where the raygen's thread is known, and the sink reads
a parent texel as a plain function (session 94).

## 9. Next: material-coherent shading on the GPU

The Modo scene above, at 640x360 on the GPU alone: 1.1 s with no bounce, 6.4 s with one, 12.6 s with four; its
subsurface makes no difference. Its 264 classes carry 489 map nodes, and a bounce ray meets any of them: a warp's
threads then run their different materials' maps one after another. MoonRay XPU queues hits by material and shades a
queue at a time (on the CPU); XPU2 is to do the same on the GPU: a bounce's hits written out, sorted by material, and
shaded in order, with the path's state in memory between the stages (a wavefront), exactly as the megakernel shades.

On hold (2026-10-25). Those timings came from the launches that did not fill the GPU (section 7). With full launches
(session 96, the same scene, 640x360, beauty only): depth 0 1.24 s, 1 3.43 s, 2 4.72 s, 4 5.01 s; a bounce costs 1.8x
the first hit, not 5x. And the scene has two materials (its layer gives each of its two meshes one layered subsurface
material; the 489 map nodes are inside them), so sorting by material would group threads into at most two sets. The
GPU profile (XPU2_PROFILE: rays, shadow rays, materials, presence, volume crossings, the rest, as shares of the
samples' clock ticks) is to say where a bounce's time goes before anything is built; a scene of many materials is to
be profiled too. The profile (session 97, 640x360, GPU only): shadow rays 68% of the samples' clock ticks at depth 0
and 55% at depth 4, rays 4% and 16%, materials 9% and 11%, the rest 20% and 17%. Sorting by material could reach a
tenth of the time at most: on this scene the lights' shadow rays are the work (two lights, several samples a hit).
Where a sample would stop to be sorted: after the trace of a bounce ray, before its volumes and
shading (the loop-carried state: the pending stack, top, radiance, alpha, the Cryptomatte marks and events, the AOV
values, the backend's texture-miss flag; the camera and time are made again from the sample's identity).

## 10. Shadow rays: a bounce stands for its BSDF sample's shadow ray

What can be cut without changing MoonRay's estimator: not the light samples, the BSDF samples, their weights or a
Russian roulette (each changes the picture's noise); shadow rays of a sample with no contribution are skipped already,
as MoonRay skips them. A BSDF sample that reaches a light and goes on as a bounce traces two rays the same way from the
same point: the shadow ray, then the bounce. The bounce's nearest hit answers the shadow ray's question (occluded
exactly when it is nearer than the shadow ray's end), so the light's contribution is held on the bounce (Pending::held)
and settled when it is traced, in both lanes. Not where the two could differ: presence shadows, ray termination
lights, shadow linking, a minimum shadow distance, a toon shadow origin, volumes, or a bounce type whose occluders are
not the shadow rays' (Config::shadowReuse, from the meshes' visibility; XPU2_SESSION_SHADOW_REUSE=0 turns it off).

The Modo scene (session 98, 640x360, GPU only, beauty): shadow rays a sample 12.21 to 6.01 (light samples' 5.28,
BSDF samples' 6.93 to 0.73), rays for hits 7.20 either way; 5.01 s to 4.76 s. The bounces got dearer: shadow rays went
from 55% of the samples' clock ticks to 38%, rays for hits from 16% to 28%: a shadow ray traced the same way just
before had brought the acceleration structure's nodes into the caches. Full size, both lanes, every output: 19.8 s to
18.7 s. The pictures with and without it agree to rounding (21 outputs within 0.0005, mean ratio 1.0000).

The GPU round on session 98 found one place it did not: a bounce that meets a surface of partial presence goes on
through it as a copy of itself, and the copy settled the held light a second time (sw-textures, presence maps: block
error 0.0033 to 0.0116, mean ratio 1.0007). The light is settled at the first surface the bounce meets, where its
shadow ray would have stopped (presence shadows off: a shadow ray sees no presence), and the ray going on through it
holds nothing (session 99).

## 11. Sampling generations: one sampler for uniform and adaptive

Asked 2026-10-25: in uniform renders the plain background got as many samples as the dragon, where they could have
gone; and one generational sampler in place of uniform and adaptive. Every XPU2 frame now runs by the cellular
generations (async_adaptive.h Cellular); the scene's sampling mode picks only when it ends.

- Uniform (cpu::Plan::sampleBudget, cpu_scene.h): every pixel is seeded with an eighth of its samples (16 at least);
  the seed's steady rate (its passes after the first, which go out together) prices uniform's samples as uniform's
  time. Where the first pass ended past half way to the seed's end (its texture tiles still loading), that time says nothing,
  and all the seed's samples over all its time price it: the car, at 16 a pixel, had stopped at 10.7 after 64.9 s, at 100%
  from 35 s; now 18.9 in 111 s at the same rate, past 16 because the seed's rate includes the first tiles' loading. Each generation
  after, the target is the error every tile would come down to if the samples that time left
  buys, at the last generation's rate (uniform's where the seed's passes ended together), were spent (a tile's error falls as the square root of its samples); tiles above
  it sample the next pass, the furthest above first if the pass would buy more than the time left. The frame ends when
  the time is up, at most twice uniform's samples and 16 times its samples a pixel. spp stays uniform's (ray
  differentials, progress). Not with a pixel sample map, a bake camera, or at 16 samples a pixel or fewer.
- Adaptive: as before, its target error, its min samples the seed and its max the most.
- Both, with a GPU: the CPU lane takes chunks too (open cells, 4 samples of a tile: milliseconds, so no generation waits
  on it), and its samples join the GPU's errors on the host (Adaptive::merged: the two lanes' errors pooled by their
  counts, sqrt(nG^2 eG^2 + nC^2 eC^2) / n).
- Test switches: XPU2_SESSION_UNIFORM_BUDGET=0 (exact uniform, for the rounds against MoonRay's uniform renders),
  XPU2_SESSION_ADAPTIVE_CPU=0 (the GPU takes every adaptive chunk), XPU2_SESSION_ADAPTIVE=trees (MoonRay's trees).
- The kit's render settings (the XPU2 Sampling group, scene_settings.py): Generation Samples (a pixel on average; 0:
  Pixel Samples squared), First Look (0: an eighth, 16 at least), Halting Rule (as long as uniform would take, or
  exactly the samples), Halt at Noise (Target Error's units; 0: none), Most Samples (0: 16 times the average), CPU
  Workers. A preview's scene carries them as UserData("/modo/xpu2/generations") {2, on, samples, spend exactly, noise,
  first, most, CPU, realtime} (rdla.py; MoonRay leaves it alone; never in a final render's), which prepare reads
  (cpu_scene.h). Form 2 says they rule: a scene's adaptive sampling is then MoonRay's alone (form 1 left an adaptive
  scene its own minimum, maximum and target error). The session's log says what a frame got ("XPU2: generations:
  first ..., most ...").

His dragon scene at 640x360, display error (8-bit RMS) against 1,024 samples a pixel:

| | time | picture | noisiest quarter | calmest half |
|---|---|---|---|---|
| uniform, 64 a pixel | 17.5 s | 3.96 | 7.20 | 0.20 |
| generations, 64 | 18.5 s | 3.45 | 6.20 | 0.46 |
| uniform, 256 | 68.6 s | 1.80 | 3.28 | 0.07 |
| generations, 256 | 72.5 s | 1.47 | 2.60 | 0.29 |

At equal samples (the first build) generations took 67% longer: the dragon's samples (subsurface, rough transmission)
cost several times the walls'; uniform's distribution run through generation passes costs only 5% more than uniform
(the waits between generations), hence the time budget. The CPU lane rendered 5.4% of the samples (uniform: 2.4%).
Against MoonRay's uniform references (gl-base, sw-textures, cut-front, pt-static) every block error is within the noise
floor and the mean ratios are as before. A scene whose samples get cheaper as it goes (sw-textures, its textures
loading) ends earlier than uniform would (11.2 s for 14.9 s).

## 12. Scene preparation

His dragon scene (a displaced PolyMesh, 8.4 million vertices after MoonRay's tessellation) took 14.3 s from the scene's
arrival to "scene ready", and every edit (a camera moved) as long again. The session's log now says where it goes
("XPU2: prepared: ...", "within the plan", "tessellation: ...", "GPU scene prepared"):

- 3.4 s were need()'s messages, made for every one of 33 million tessellated indices (o.getName()+...) though used only
  on failure: X2_NEED makes them on failure alone, in the twenty loops over a mesh's elements.
- Reading the tessellation (3.8 s with the checks) and the polygon mesh (1.2 s) one element after another: in parallel
  (parallelRange; the mesh in two passes, which halves of each face make a triangle, then each face's triangles where
  the ones before it end, the same order). xpu2_geom.dll's buffers after MoonRay's tessellation over TBB (0.6 s to
  0.13 s). The rendered picture is the same to the bit (session 105 against 115).
- MoonRay's own tessellation is 5.2 s of what is left (its displacement about half, over 12 threads at about two busy).
- A displaced mesh MoonRay tessellates was never kept across edits (!objectDisplaced): now kept while the mesh, each
  part's displacement and its maps through and through (deepSignature), which displacement is whose and polyvertices
  stay the same; the CPU's Embree scenes are taken over where no mesh changed (CpuScene::sameGeometry).

First preparation 14.3 s to 9.6 s; a camera moved 9.7 s to 0.21 s; a displacement edited (its mesh alone tessellated,
the CPU scene built again) 1.7 s. A frame after a camera move on kept geometry against a fresh one: block error 0.0006,
mean ratio 1.0000 (MoonRay tessellated in the first camera's render space).

## 13. Realtime

The kit's preview window has an RT switch beside the engine (MoonLightIPR, or MoonRay where XPU2 renders the
preview); Render renders once. With RT on the preview follows the scene at the realtime preview size, and the
generations' UserData carries realtime (its ninth value, cpu_scene.h):

- A camera moved alone (Scene::onlyCameraDirty) keeps the plan, the CPU scene and the GPU's scene: cpu::updateCamera
  makes the camera and render space again and the GPU gets its configuration (GpuLane::updateCamera): 0.1 ms, where
  preparing took 0.21 s. A scene whose view depends on the camera (adaptive subdivision, volumes, filters, a pixel
  sample map, motion) prepares as before.
- A realtime frame is the GPU's alone; its first pass covers every pixel (sparse first passes, a quarter of the pixels
  then the rest, flashed as scanlines before they settled and are off since 2026-10-30; XPU2_SESSION_RT_SPARSE=1), with launches sized to about 15 ms of the GPU's rate across frames (rateHint), and
  its pictures every 33 ms.
- Not used by the 0.5 kit, which under XPU2 offers no Denoised Beauty in realtime and writes no guides for it
  (`panel.py` `_offer_denoised`, `denoising.py`); the session keeps the capability, and the measurements below are
  its record. Pictures with Denoised Beauty chosen are denoised in the session (RealtimeDenoiser, on a thread of its own): OptiX's
  denoiser (GpuLane::Denoiser, its HDR model on a stream of its own at the highest priority) with the kit's guides, the
  albedo denoised on its own first (as the kit's "Denoiser cleans its guides first"; this dragon's albedo differs from
  sample to sample) and the normals in camera space. A picture some of whose pixels have no sample yet is denoised at
  half its size and shown at its own: filled from their neighbours, those pixels read to OptiX as detail, and it left
  such a picture all but as it was (a mean change of 0.0001 against 0.03 once every pixel had a sample). The frame's
  last picture is denoised too, and the kit keeps it as the frame's until its own denoiser makes one from the files.
  Open Image Denoise on the CPU was tried first: 0.24 s for a 640 by 360 picture at its fast quality on six cores, and
  as long again for each guide it cleans.

His dragon at 640 by 360: a camera move's first picture 125-160 ms after it arrives, raw; denoised 195-300 ms, OptiX
taking 65-70 ms while the path tracer has the GPU (5.5 ms on its own), the half-size pictures included. Diagnostics:
XPU2_SESSION_RT_GUIDES (clean, raw, normal, none), XPU2_SESSION_RT_SHOW (color, albedo, normal: the denoiser's input
shown), XPU2_SESSION_RT_DEBUG (each picture's intensity and change), XPU2_SESSION_RT_DENOISE=0, XPU2_SESSION_RT_PRIORITY=0.

Added 2026-10-27: **realtime's textures at most 1280 pixels a side.** A realtime plan reads each texture from its first
mip level no wider or taller than 1280 (`texture::Pyramid::view`, `XPU2_SESSION_RT_TEXTURE_SIDE`, 0 = whole); what the
plan works out from textures (displacement) still reads them whole, and other plans are unchanged. On the car at
960x540: fine-detail tiles sent up 137,404 to 8,368 (2.8 GB to 206 MB of the card), the card's peak 17.9 to 14.6 GB,
the session 11.3 to 10.0 GB. The frame took as long (about 59 s): realtime's GPU is underused (570 launches of about
15 ms, the CPU a third of the samples), which is what to work on next.

## 14. His Modo scenes (9 Spheres, Dose, Interior) and the Source Overrides

Recorded at the time: exported headless with the test kit then installed (modo_cl, test-results/xpu2-cl), each was
refused before drawing:

- A displacement on fur (9 Spheres): MoonRay displaces meshes alone (PolyMesh, OpenSubdivMesh); on curves and points it
  is now left out as there, not refused.
- A signed texture (−1 to 0.74, a displacement map): texels below zero are kept (host_texture.h, texture_core.h's
  texel), as MoonRay's sampler keeps them; non-finite ones are still refused.
- Named UV sets (Dose, Interior): ModoTextureMap's modes 12 and 13 read uv_name's primitive attribute and
  modo_primary_uv, moved by st's offset along the set's st derivatives (primPartials), or uv_affine where the geometry
  gives none (a subdivision surface, State::subdivision), as ModoTextureMap.cc does; a hit without them gives the
  neutral result. The keys follow the map's attributes (map_admission.h).
- Far coordinates (Interior: u −1068, a set tiled a thousand times): beyond ±1024 repeats they are wrapped into the
  first (clamped near the edge for a clamped texture) rather than refused; nearer ones as before.
- Interior's displaced triangle meshes (XPU2's own displacement, mesh_displacement.h): each vertex over the machine's
  threads (12.3 s to 3.7 s), and kept across edits as MoonRay-tessellated ones are (displacementKey "triangles:N").

Against MoonRay's vector renders at reduced size, 64 samples a pixel, uniform: Dose (400 x 500) mean ratio 1.0006,
9.1 s against 46 s; Interior (512 x 340) 1.0003, 51 s against 193 s; 9 Spheres (640 x 480) 0.9993, 183 s against 177 s
with ball8's displacement off in both. MoonRay renders 9 Spheres black: ball8's displacement (a subdivision sphere
displaced through a named UV set's AttributeMap) engulfs the camera; XPU2 draws the sphere in place. 9 Spheres is slow
in XPU2 because ball6's absorption volume is textured (FamilyVolumesMapped, not in GpuFamilies): 3.6 million of its
19.7 million samples go to the CPU.

Source Overrides: XPU2 follows the toon terminator and albedo, Cryptomatte volumes, fog alpha, lights seen by the camera in light outputs (`MOONRAY_MODO_CAMERA_LIGHT_LPE`), opaque alpha,
transmission roughness and USD specular; counts each Cryptomatte sample once by construction (no shared flag to race);
the scene-level ones (display filter inputs, toon lines apart, light motion) arrive in the scene. Low-latency rendering
(async::lowLatency, from the process's priority class the kit lowers, or XPU2_SESSION_LOW_LATENCY): no thread raised
above the process's priority (the GPU feeder and the trees' pool give way to Modo) and a CPU thread left free besides
the feeder's. Realtime's guide cleaning follows "Denoiser cleans its guides first" (MOONRAY_MODO_DENOISE_CLEAN_GUIDES).
Diagnostic: XPU2_SESSION_DEBUG_FAILURES names the CPU lane's first failed shadings and texture lookups.

## 15. Textured volumes on the GPU, the texture cache's budget, displacements' attributes

- **Textured volumes.** A mapped BaseVolume whose maps give its attenuation alone (9 Spheres' absorbing ball6) has its
  extinction baked into a dense grid; the GPU now holds those grids (LaunchParams::volumeGrids, the NanoVDB ones left
  null) and samples them as the host does (the nearest voxel, volume_grid.h), and volume::gpuRenders is the one rule
  for what the GPU renders and what it defers (VDB, mapped colour, emission or anisotropy still go to the CPU lane).
  FamilyVolumesMapped joins GpuFamilies. His live 9 Spheres preview (1280 x 960, 16 samples): unfinished after 400 s
  with 5.2 million samples deferred and the GPU idle from 90 s on, now 104 s with none deferred.
- **The fine-detail cache's budget** (GpuLane::fineBudget): what the scene's level-0 tiles need, up to the card's free
  memory less a fifth of the card (1.5 GB at least) for the frame and Modo's viewport, never below 256 MB, measured at
  each preparation (XPU2_SESSION_FINE_BUDGET_MB fixes it); was 512 MB. The allocation cap is the card's own memory
  where it exceeds 20 GB.
- **Displacements read the mesh's primitive attributes.** MoonRay fills a displacement's state with what its
  displacement's attribute table asks for; the proxies XPU2 displaces through had none, so a named UV set read in a
  displacement read nothing and 9 Spheres' ball8 moved evenly. xpu2_geom.dll now gives each displacement a table of
  the control mesh's attributes and hands their values to displaceWith (X2SubdDisplaceAttrFn), whose state holds them
  as the hit's, without st derivatives as MoonRay's displacement state (subdivision and PolyMesh both).
- **Test with the kit's Source Overrides.** Run without its environment, an image without alpha reads alpha 0 (not
  MOONRAY_MODO_OPAQUE_ALPHA's one): ball8's displacement divides by that alpha and engulfs the camera, and MoonRay and
  XPU2 both render 9 Spheres black. compare.py takes --env for both renderers. With the kit's overrides, at 640 x 480
  and 64 samples a pixel: 9 Spheres 64 s against MoonRay's 176 s (mean ratio 0.9997), Dose 9.5 s against 43 s (1.0006).
- **Pinned host memory follows the machine.** His Dose preview (1600 x 2000, 26 outputs, adaptive) fell back with
  "GPU staging/residency budget": the frame's pinned readbacks outgrew a fixed 2 GB, and MoonRay XPU, rendering in
  XPU2's place, left the can out (MoonRay XPU's own doing). The pinned cap is now a quarter of the machine's memory,
  between 2 and 16 GB (GlobalMemoryStatusEx), and the refusal names the memory and its use. Reproduced from his live
  preview scene (its text, with a binary file of identical fingerprints written headless), it renders whole.

## 16. The tessellation handler: geometry by device, levels and their cache

Added 2026-10-26 (sessions 143-151).

Memory had a failsafe (the session's job-object ceiling, a geometry budget from free RAM and the card) but no answer
for a scene whose geometry is honestly large. Textures are streamed by level; geometry cannot be, the same way: a
coarser level changes what a ray hits (silhouettes, contact, displaced detail), not only how sharp a lookup is, and a
ray finds only what its acceleration structure already holds. So the handler distributes whole meshes and keeps
MoonRay's own levels, and the image stays MoonRay's.

1. **Estimate before tessellating** (done, session 144). A displaced mesh MoonRay tessellates uniformly is sized from
   its faces and mesh resolution (about r x r times its triangles, 18 bytes a triangle with its share of vertices)
   before its tessellation runs: one the machine cannot hold is refused at once (a 4.5 M-triangle tyre at resolution 8:
   2.8 s, where the budget check after tessellating took 30 s).
2. **Route by device** (done, session 144). Two budgets: RAM's refuses the scene (geometryBudget), the card's routes it
   (gpuGeometryBudget, cpu::routeMeshes). Past the card's, the largest meshes' data (a shared instance's once) is left
   to the CPU lane alone (Mesh::cpuOnly): on the GPU a stand-in box, a sample that meets it rendered by the CPU from its
   identity. Forced on the displaced-tyre test scene (every tyre the CPU's): display error 1.16 against MoonRay, as
   before; the 1.24 GB scene the card refused now renders.
3. **MoonRay's adaptive tessellation.** MoonRay dices by the dicing camera where adaptive_error > 0 (OpenSubdivMesh,
   PolyMesh); XPU2 tessellates through MoonRay's code, so it has the same rule and the same image. The kit can map
   Modo's displacement rate to adaptive_error, so both renderers make far less geometry where the mesh is small on
   screen. Done (session 145, kit render setting Adaptive Displacement, pixels, 0 = uniform as before; an object's own
   Subdivision Screen Error wins; instances stay uniform, MoonRay tessellating a shared reference once): a displaced
   polygon mesh is camera-dependent in XPU2 as a subdivision surface was (a camera move tessellates it again). Four
   displaced tyres at resolution 8: 351 MB of geometry uniform, 7 MB at 2 pixels; XPU2 24.1 s to 14.9 s, MoonRay 13.1 s to
   7.8 s; the same display error against MoonRay either way.
4. **A cache of levels.** Tessellations kept on disk by mesh signature, displacement key and level (a mesh resolution,
   or an adaptive dicing camera's bucket), so a scene opened again, a session restarted or a camera returned to reuses
   them instead of tessellating again. The routing sees whichever level the scene uses. Done (sessions 145-146:
   geometryKey, loadGeometry, saveGeometry; %LOCALAPPDATA%/MoonRayForModo/xpu2-geometry, 20 GB, written on threads the
   session joins before it ends): the 32 M-triangle displaced tyre's second session read 1.46 GB in place of tessellating,
   meshes and tessellation 11.2 s to 0.83 s, the scene ready at 4.3 s in place of 16.2 s, the picture the same (oiiotool
   diff PASS). The key holds the tessellation DLL's size and time: a new build makes its own.
5. **Tighter stand-ins.** Done as boxes round runs of 128 of the mesh's own triangles (session 145), not an inflated
   coarse level: an offset shell is hard to make enclose a displaced surface everywhere (open edges, negative heights,
   subdivision bulging past its cage), and a ray slipping through would lose a hit; every triangle is inside its box.
   Forced routing on the displaced tyres: 56.9 s to 48.4 s, the same picture. Superseded plan:
   **Coarse levels as stand-ins.** A cached coarse level, pushed out along its normals by the displacement's largest
   height so that it encloses the fine surface, in place of the box: far fewer GPU samples handed to the CPU for a
   mesh left to it, the CPU still rendering each on the full mesh.

Checked: the card's 11 GB peak on the routed scene was not the deferred samples' queue (a fixed ring of 4 M entries in
pinned host memory); the lane's own device memory was 2.2 GB, the rest the driver's around launches and builds.
## 17. Scene preparation, second round: textures on demand, the canonical copy held once, geometry in memory

Added 2026-10-26 (sessions 152-156). On the car scene (Lamborghini, small overlay, 1 GB of geometry) a preview took
32 s from the scene's arrival to "scene ready". Most of it was the same gigabyte copied again and again, and every
texture decoded whole before the first sample. The image is unchanged by all of it: CPU-only and GPU-only renders
against the build before are 0.00 display error, and a live edit (200 texture opacities changed during a render)
gives the same two frames as the build before.

1. **Textures' finest level on demand** (`host_texture.h` `Pyramid::loadTile`, `texture_core.h` `Mip::load`). Levels 1
   and up are still decoded as the scene is prepared; level 0 starts with no pages, and a page is decoded, through a
   shared OpenImageIO ImageCache, the first time a lookup or the GPU's tile cache asks for it, with the same values
   the whole level gave. The GPU's tile cache decodes the tiles it is asked for side by side before it sends them up
   (`async_gpu.h`, `commitTiles`). Texture pyramids 18.4 s to 5.0 s; the session's RAM 12.3 to 10.1 GB; the first
   frame's render a little longer (2.8 s to 4.3 s) as tiles are decoded while it runs; the finished picture 54.2 s to
   38.9 s. `XPU2_SESSION_EAGER_TEXTURES=1` decodes every level at once, as before.
2. **The canonical copy held once** (`../rdl_scene_index.h`, `../indexed_scene.h`, `canonical_session.h`). Vectors whose
   values lie in memory as their codec writes them (whole numbers, floats, RDL's vectors and colours with no padding)
   are encoded and decoded as one block, the same bytes. The index keeps an attribute as captured, once
   (`SceneIndex::same`, `Buffer::ready`), with the capture's bytes moved in rather than copied (three copies of each
   byte before). The replica as an RDLB file, which nothing read, is no longer written (`XPU2_SESSION_CANONICAL_FILE=1`
   writes it). Capture 4.2 s to 0.8 s, index 3.5 s to 1.9 s, replica 4.2 s to 1.3 s, the file's 10 s gone; the session
   1.4 GB smaller.
3. **The geometry's binary scene in shared memory** (`xpu2_session.cc`, `readBinaryMemory`). The host may name the RDLB
   it hands over `modo-memory-rdlb:<length>:<name>`, a mapping it holds, which XPU2 reads as MoonRay reads the file
   (`BinaryReader::fromBytes`). The gigabyte no longer goes to the disk and back; XPU2's own load was only about 2 s,
   so what it saves is the host's write. The reference renderer is still handed the file.

The car's "scene ready" went from 32 s to 19.2 s. What is left: the plan's meshes and materials (about 7.5 s), its
objects (3 s) and texture pyramids (5 s).

## Order

As of 2026-10-30, where it could be checked against the code:

1. Classifier from the admission rules, with the GPU capability table; decisions shown in the session's log. Done
   (`entity_classes.h`).
2. Scene mode from the decisions; the plan builder keeps its checks as assertions. The mode is worked out and logged
   ("MoonRay XPU2 routing", `xpu2_session.cc`); the plan builder still refuses by itself.
3. Deferral to the CPU (GPU stand-ins, deferral status, the CPU lane's retry queue), replacing `cpuOnly`. Done for
   stand-ins; `cpuOnly` remains for light filters, shadow linking, lobe light sets and emissive volumes. Points, and
   geometry and lights in motion, are the GPU's since sessions 211-213 (`XPU2_GPU_MOTION_OFF` defers them again, to
   compare).
4. Cell seeding by suitability and ownership by deferral rate. Done (`device_suit.h`; deferred samples count as the
   CPU's in the ownership flips).
5. GPU program families as separate modules, compiled on demand. Nine modules, done (`build_async.py`); linking and
   compiling per scene by its families, not done.
6. The change set to the renderer: partial rebuilds of the scene on both devices. In part: a camera or a move
   without a rebuild where it can (`GpuLane::updateCamera`, `GpuLane::updatePlacements`), unchanged meshes kept.

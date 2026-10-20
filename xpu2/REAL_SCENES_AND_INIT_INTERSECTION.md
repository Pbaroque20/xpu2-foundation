# Real scenes, the 12-worker gate, and Init Intersection

Measured 2026-10-19 on the RTX 3090 / i7-4930K (6 cores, 12 threads). Nothing
here is installed, enabled by default or committed.

## Real MoonRay scenes: what the CPU spends its time on

The ten MoonRay example scenes (`downloads/moonray-example-scenes`) were
rendered with original XPU at 320x180, fixed 16 samples, one thread. The
`cpu_cost_full` build measured four of them (`tools/xpu2_real_scene_ceiling.py`,
`test-results/xpu2-real-scene-ceiling-1`):

| Scene | Frame | Material/maps | Selected light | Material cost per point |
| --- | ---: | ---: | ---: | ---: |
| salle_de_bain | 50.8 s | 10.5% | 3.3% | 0.35 us |
| the_wooden_staircase | 46.4 s | 16.4% | 4.0% | 0.60 us |
| contemporary_bathroom | 55.1 s | 12.7% | 3.6% | 0.44 us |
| country_kitchen | 68.4 s | 7.2% | 5.7% | 0.37 us |

Real materials cost 0.35-0.6 us per point, against 12.8 us on the eight-layer,
32-map stress fixture and about 1.5 us of bridge cost per lane. A GPU material
job through the current bridge would lose time on these scenes.

MoonRay's own breakdown over all ten scenes: Integration 17-42%, Init
Intersection 9-20%, shade handler plus shading plus warm texturing 12-25%,
presence shadow rays on the CPU 20-25% where enabled, GPU intersection 3-7%.
(The first, cold render of a scene also spends 24-35% reading textures.)

## Original XPU is not bit-repeatable on real scenes

Two original-XPU renders with identical settings at one thread differ in 50-75%
of pixels on the four textured scenes, as much as the candidate differs from
XPU. veach-mis (100.00% GPU intersection) repeated exactly with the origin
runtime but not with the 0.4.2 runtime; vectorized mode varied too. Scalar
mode at one thread repeated exactly, so it is the mode for word-for-word checks
of CPU-side changes.

## The frozen 12-worker gate cannot be passed by original XPU

`tools/xpu2_heldout_gate.py` (`test-results/xpu2-heldout-gate`) scored new
renders against the frozen five-reference envelopes, unchanged:

- 20 of 20 new original-XPU renders fail.
- 17 of 20 disabled-route candidate renders fail (3 pass).
- The disabled-route candidate is word-for-word identical to original XPU at
  one thread on all four fixtures.

The maximum over ten reference pairs is too tight; the signed-mean metric fails
most often. The 12-worker failures are not evidence against XPU2. Choosing a
replacement gate is the user's decision.

## Init Intersection

`xpu2/isect_profile.h` with `xpu2/geom_isect_profile.py` times sampled, exclusive
stages of `initIntersectionPhase1` and mesh `postIntersect`, built into an isolated
`librendering_geom.dll` by `tools/xpu2_geom_build.py` (the build tree is never
written). About 45 ns per stage is the timer itself. Net per hit on the kitchen,
all triangle meshes, about 700 ns in total:

| Part | Net per hit |
| --- | ---: |
| Instance attribute override | ~180 ns |
| Required-attribute interpolation | ~140 ns |
| Curvature (re-reads UVs and normals) | ~70-100 ns |
| Shadow epsilon, id, instance transforms, facing | ~70 ns |
| UVs | ~60 ns |
| Attribute table setup | ~40 ns |

Per-hit costs are nearly the same on small veach-mis, so this is instruction
work, not memory latency.

### Fix: no temporary key vector on instance hits

`Intersection::setInstanceAttributesOverride` builds a `std::vector` from an
initializer list on every call, one heap allocation per instance level per hit.
`xpu2/geom_instance_override.py` makes the same three fills, in the same order,
with a key list built once (only `geom/prim/Util.cc` changes).

- Scalar mode, one thread: word-for-word identical to the baseline.
- Kitchen, XPU, one thread, three alternating pairs: Init Intersection
  12.60 -> 9.86 s, MCRT 67.2 -> 63.8 s median (about 5% faster, every pair faster).
- Kitchen, XPU, 12 threads, five pairs: MCRT 11.29 -> 10.69 s median (about 5%),
  every pair faster.

It only helps instanced geometry. The example scenes put everything under
instancers; scenes exported from Modo mostly use plain meshes. Shipping it
needs a MoonRay Source Override switch, per the project's rule for MoonRay
source changes.

### Fix: curvature reuses the UVs postIntersect read

`xpu2/geom_isect_reuse.py` (on top of the instance fix) hands the hit's three
UVs from `postIntersect` to `computeIntersectCurvature` through a one-shot,
thread-local slot keyed by mesh, face and triangle half, for triangle and quad
meshes. Only `initIntersectionPhase1` calls the two, back to back; curvature
clears the slot and reads the UVs itself on any mismatch.

- Scalar mode, one thread: word-for-word identical to the baseline on the
  kitchen and on the Modo scene below.
- Kitchen, XPU, one thread, three runs each: Init Intersection 12.6 s baseline,
  10.0 s instance fix, 9.2 s both (27% less); MCRT 68.0 / 65.0 / 64.4 s median.
  The UV step alone is close to run-to-run noise in frame time.

## A plain-mesh scene exported from Modo

`test-results/cl-export/Technik_Tom_Moonray_bundled_baked.rdla` (kit export,
8,382 plain `RdlMeshGeometry`, no instancers, 860 Modo texture maps, 22 normal
maps), 200x200, 16 samples, one thread, XPU:

- Init Intersection is only 4.8% of MCRT. Integration 25%, subsurface
  integration 22%, CPU Embree intersection 18% (in XPU mode), shading 8%.
- Per hit (about 2.7 M hits, all triangles, no instance cost): required-attribute
  interpolation ~285 ns net leads, then curvature's re-reads ~100 ns, UVs ~87 ns,
  table setup ~72 ns.
- Both fixes: Init Intersection 1.88-2.01 s -> 1.85-2.00 s; frame change within
  noise. On scenes like this, Init Intersection is not where the time goes.

## Why the Modo scene traces 18% of its time on the CPU in XPU mode

Of 14.69 M intersection rays only 4.12 M went through the GPU bundle (99.88% of
those on the GPU); about 10.6 M were traced one by one with Embree. Of 10.56 M
occlusion rays, 3.3 M were not bundled. They come from subsurface scattering:

- In bundled and XPU mode, each subsurface lane goes to scalar C++
  (`CPP_computeRadianceSubsurface`, `PathIntegratorBundled.ispc:497`), which
  "drops into the depth-first integrator" (`PathIntegratorUtil.cc`).
- The diffusion BSSRDF traces up to `bssrdf samples` (64 here) projection
  probe rays per subsurface hit through `Scene::intersectRay`, Embree, one at a
  time (`PathIntegratorSubsurface.cc:785`): 1.35 M probes.
- Its light samples use CPU occlusion (`isRayOccluded`, lines 244 and 489),
  and its indirect bounces continue the whole path in scalar mode
  (`computeRadianceRecurse1`, line 475), whose rays also go to Embree:
  roughly 9 M of the CPU intersection rays.

With `max_subsurface_per_path 0` every ray is bundled, CPU Embree time is
about 0%, and MCRT falls from 39.5 s to 23.0 s (a different image, so this only
locates the cost). Subsurface integration (21%) plus CPU Embree intersection
(18%) and occlusion (3%) are the largest share of this Modo frame.

CPU Embree costs about 0.67 us per intersection ray here, against about
0.15 us of CPU time per ray handed to the GPU, so the GPU path would be cheaper
on the CPU side. Two routes, in order of size:

1. Batch the projection probes: the 64 probes of one subsurface hit are
   independent, so they could be traced as one batch (for all lanes at once)
   through the GPU accelerator instead of 64 Embree calls.
2. Queue the continuation: instead of recursing in scalar mode, put the
   subsurface exit point's bounce rays and shadow rays into the bundled ray and
   occlusion queues, as ordinary BSDF samples are, so XPU's GPU queues handle
   them. This is the larger share and the larger change.

### Feasibility: find on the GPU, confirm on the CPU

`xpu2/rndr_sss_feasibility.py` with `xpu2/sss_probe.h` (isolated
`librendering_rndr.dll`, read-only) tags the Modo scene's scalar
`Scene::intersectRay` calls by origin, samples 1 in 32, re-intersects Embree's
hit triangle alone in a one-triangle Embree scene on the same device, and sends
the same rays to XPU's GPU accelerator in batches of 256 (200x200, 16 samples,
one thread):

- This scene uses random-walk subsurface, not the diffusion probes: 9.22 M
  walk rays (14.7% hit), 1.35 M other subsurface rays, 3.32 M subsurface
  occlusion rays, no probes.
- Exact confirmation: 42,353 of 42,353 walk hits and 8,340 of 8,340 other hits
  re-intersected alone matched the full Embree result bit for bit (t, u, v, Ng).
- GPU agreement on walk rays: same triangle 42,031, both miss 244,206, GPU hit
  where Embree missed 1,405 (Embree's filters, such as the trace set or
  back-face culling), a different triangle 318, GPU miss where Embree hit 0;
  0.6% disagree. Other subsurface rays agreed completely. OptiX t/u/v equal
  Embree's bit for bit for only about 1.5% of same-triangle hits, so the
  confirmation step is required.
- Prize: Embree costs about 0.57 us per walk ray (about 5.2 s), 0.52 us per other
  ray (0.7 s), plus about 1.3 s of subsurface occlusion: roughly 8 s, 21% of the
  39.5 s frame at one thread.

What exact use would need:

1. A conservative GPU test (slightly enlarged triangles, no filters), so that
   a GPU miss proves an Embree miss. Hardware RTX triangles cannot do this; a
   software intersection program can. Misses are 85% of walk rays.
2. For GPU hits, Embree itself on the shortened ray [tnear, t_gpu + delta]:
   it returns exactly what the full trace returns, filters included, and also
   settles near-ties; on an unexpected miss, the full Embree trace.
   (Confirming only the GPU's triangle would miss the 0.11% different-triangle
   cases.)
3. Batches large enough to hide GPU latency. This is the blocker: a random walk
   is sequential, and `CPP_computeRadianceSubsurface` runs lane by lane, so at
   most about 8 rays are ready at once against roughly 4.5 us of Embree work.
   Many walks must be in flight together (a wavefront over many paths, parked
   and resumed), the same kind of restructuring as the XPU2 asynchronous stage.

Changing Embree to agree with OptiX is not an option: it would change original
XPU's own images, and the RTX hardware triangle arithmetic is not documented.

Either changes which intersector answers those rays (OptiX instead of Embree),
so images would no longer match current XPU bit for bit, though original XPU
already varies from run to run on real scenes. It is original XPU behaviour, so
the gain would apply to XPU itself, as a MoonRay Source Override.

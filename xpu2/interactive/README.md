# XPU2

*XPU2: Asynchronous Monte Carlo Path Tracer designed by Raphael Tobar, developed with AI assistance. Not affiliated with or endorsed by DreamWorks Animation; MoonRay and OpenMoonRay are DreamWorks Animation's.*

XPU2 is a Windows path tracer in which the CPU and an NVIDIA GPU render the same image at once, each claiming its own work and neither waiting for the other. **In MoonRay for Modo 0.5, a pre-production release, it is the default GPU renderer for MoonRay previews; final renders come from MoonRay, and a scene XPU2 does not render yet goes to MoonRay XPU.**

How it works, part by part: the [engineer manual](../manual/README.md).

## What works

- Both devices together, or either alone: a work grid of cells that move between them, one integrator compiled for both, every sample defined by its identity so it gives the same result on either. Geometry and lights in motion, and points, render on both. What the GPU programs lack is deferred to the CPU lane, and light filters, shadow linking, lobe light sets and volumes whose maps give their emission keep a scene on the CPU lane. Glowing volumes, VDB ones too, render on the GPU.
- MoonRay's path tracer and shader library, ported and checked against MoonRay: lights and light filters, the DwaBase materials and combinators, maps, subsurface, hair and curves, subdivision and displacement, volumes and VDB, cameras, motion blur, render outputs, light-path AOVs and Cryptomatte. Each feature's state and check are in [feature progress](FEATURE_PROGRESS.md).
- Sampling by generations, spending samples where the picture is still noisy.
- Textures in tiers: every level in host RAM, the coarse levels and a cache of fine tiles in VRAM, misses shaded from a coarser level and rendered again in full.
- A session speaking the kit's preview protocol: scene edits as they happen, realtime camera and object moves, cancel and drain, and whole-scene fallback to MoonRay XPU. Native and reference pixels are never blended.

## Limits

[Feature progress](FEATURE_PROGRESS.md) lists what is not ported or not yet checked, and [section 21 of the manual](../manual/README.md#21-not-done-yet) what is not done yet. Measurements so far come from one machine (an RTX 3090 and an i7-4930K).

## Measurements

[The asynchronous renderer's results](ASYNC_RESULTS.md) (22 and 23 October) and the [table on the Modo scenes](../FOUNDATION.md) are specific to their scenes, not a general promise.

As a record, the 22 October prototype this renderer grew from was compared with original XPU on two synthetic textured direct-lighting scenes: fresh warm render-phase medians fell from **4.095 to 2.763 seconds** and **6.882 to 4.605 seconds** at 12 threads. See [its validation and numerical evidence](VALIDATION.md).

## Build, run, and licensing

Read [BUILDING.md](BUILDING.md) before running. Original code is [MIT-licensed](LICENSE), copyright Raphael Tobar. Upstream-derived portions retain Apache-2.0/BSD terms and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Earlier experiments elsewhere under `xpu2/` are historical. Comments describing this directory's former private development do not restrict the licenses above. Local historical reports and generated assets are not all included.

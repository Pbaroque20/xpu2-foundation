# xpu2-foundation

![Cellular automata scheduling in XPU2](/xpu2/manual/xpu2-cellular-scheduling.gif)

*Cellular automata scheduling in XPU2; both CPU/GPU contribute to the image without waiting. Footage Provided by Pixelwerk.*

**XPU2** is an asynchronous CPU + GPU Monte Carlo path tracer. The CPU and GPU render collaborate on same image at once: each
claims its own work, neither waits for the other, and any sample can be rendered on either device with the same
result.

XPU2 is designed by Raphael Tobar, implemented with AI-assistance. Not affiliated with or endorsed by DreamWorks
Animation; MoonRay and OpenMoonRay are DreamWorks Animation's. XPU2 is released under the [MIT license](/LICENSE).

The mantra for XPU2 is: "Use your silicon!"

![XPU2 architecture](/xpu2/manual/xpu2_architecture.svg)

## Core ideas

- **Samples are defined by identity.** A sample is a pixel, a sample index and a frame seed, and its random numbers
  derive from those alone. Either device produces the same result, so work moves between them freely.
- **One integrator.** The path tracer is written once and compiled for both the CPU and the GPU.
- **Cellular work grid.** The image is divided into cells, each owned by a device. Workers on both devices move
  between cells by local rules, and every chunk of work is claimed exactly once.
- **No waiting.** A broker thread keeps two GPU launches queued, sized from the GPU's measured rate. The host polls
  the GPU and never blocks on it.
- **Work goes where it runs best.** An indexer classifies each scene object by the devices that can render it. Cells
  start on the better-suited device, and a GPU sample that hits a CPU-only feature is handed to the CPU.
- **Motion on both devices.** Moving meshes, subdivision surfaces, instances, curves, points, volume bounds and any
  number of moving lights render on the GPU as on the CPU, traced at each sample's time. Test scenes that used to send
  those samples to the CPU finished 2.6 to 8.5 times sooner; a scene with nothing moving costs the GPU about 2 to 3%.
  A deforming subdivision surface is tessellated once over its time steps, so adaptive tessellation and instanced
  deforming prototypes render too.
- **Volumes on both devices.** VDB clouds and fire, and mapped and glowing volumes, are marched on the GPU as on the
  CPU, the volumes' map programs run at each sample point. A march resumes from checkpoints instead of starting
  again, and a ray finds every volume it crosses in one traversal. On an RTX 3090 a VDB cloud went from 88 to 57
  seconds and a glowing mapped volume from 37 to 27.5, every picture bit-identical. Fire written as a single-value
  temperature or flames grid glows too.
- **Lean card memory.** The GPU's stack is sized from the program's real call chains and from what the scene contains,
  and each thread's scratch from what the scene uses. A launch is as wide as the card's cores can hold at once, and
  narrower only when the card is short of memory. On an RTX 3090 a simple scene went from 10.9 GB of the card to 4.5 GB
  at the same speed and with the same picture, so smaller cards can render. The parked-sample queue and the subsurface
  stack are made only for scenes that use them, and the texture cache grows as tiles are read.
- **Textures on demand.** Host RAM and VRAM keep each texture's coarse levels, up to 128 pixels a side; every larger
  level is read tile by tile as frames sample it, decoded on demand and let go under memory pressure. A GPU sample that
  misses a tile shades from a coarser level and is re-rendered in full before the frame ends, so pictures match loading
  everything. Only what frames sample counts against memory, so texture overflow is rare; where a scene still does not
  fit, each texture is fitted to what its meshes show on screen, then compressed (BC6H), then made smaller, and the
  preview says so. Texels are kept as half floats, grey maps in one channel and flat tiles as one value, so textures
  take about a third of the memory.
- **Sampling generations.** One sampler handles both uniform and adaptive sampling, spending samples where the error
  is highest. A frame runs for as long as uniform sampling would take, or to an exact sample count.
- **Audited results.** Each device keeps its own film. When a frame ends the films merge and every pixel's sample
  count is checked.
- **Every device in the machine.** Extra graphics cards follow the first, each with its own scene and film, taking
  cells from the same grid at their own rate. CPU workers span every Windows processor group. Realtime denoising goes
  to the first card that can run OptiX's denoiser, then to Open Image Denoise on the CPU. Each card and processor
  group can be switched off (`--list-devices` lists them).
- **NUMA-aware.** Workers are placed a share on each NUMA node, each card is fed and its texture tiles decoded on the
  node nearest it, and decoded tiles live in their node's memory. A node short of memory moves its oldest tiles to
  another node before letting any go: a hop across the sockets instead of a decode from disk. A simulation app tests
  cards, groups, nodes and denoiser hand-offs on one card.
- **Across machines: Rhizome.** P2P, decentralized network rendering: a swarm of peers, each a whole XPU2, with no
  render server. One peer takes the master integrator role for a job (any peer can, and the next takes over if it
  fails), work moves between neighbours by the same cellular rules, and every machine's RAM and card memory form one
  pool. A node fetches texture data from the pool only as its render reads it, and a cell goes back to the machine
  that rendered it the frame before, where that data already is. Peers find each other on a local network, and
  behind routers and VPNs by probing addresses. Artists get it in MoonRay for Modo 0.5.4, from one window in Modo;
  setting it up is a key, a firewall rule and Start Node on each machine
  ([steps](https://github.com/Pbaroque20/xpu2-foundation/tree/xpu2-network#setting-it-up)). Its architecture and
  state are on the [`xpu2-network`](https://github.com/Pbaroque20/xpu2-foundation/tree/xpu2-network) branch, also MIT.
  Confirmed by a user on 13 November 2026 between two computers: their main workstation (Ryzen 9 7950X, NVIDIA RTX 5090) and an AMD Ryzen Threadripper 1950X with an NVIDIA GTX 1080 Ti. That 1080 Ti is older than XPU2's GPU programs support, so the second machine renders its share on its processor (expected by design; not yet checked in that test's logs).

## Reference renderer: MoonRay

XPU2 currently implements [MoonRay](https://openmoonray.org/). MoonRay's path tracer and shader library (lights,
BSDFs, materials, maps, volumes, subsurface, hair, cameras, outputs) are ported to XPU2's integrator, so XPU2's images
converge to MoonRay's. Every port is checked against MoonRay renders, and scenes XPU2 can't render yet go to MoonRay
XPU. [Feature progress](/xpu2/interactive/FEATURE_PROGRESS.md) lists each feature and its check.

## Showcase: MoonRay for Modo

XPU2 renders previews in [MoonRay for Modo](https://www.github.com/pbaroque20/moonrayformodo/) on real production scenes, as part of the standard kit since 0.5, a
pre-production release; since 0.5.4 it also renders headless (`moonray_xpu2 render`) and across machines with Rhizome. Measured on an RTX 3090 and an i7-4930K (6 cores, 12 threads):

| Scene | XPU2 (CPU + GPU) | MoonRay | Image vs MoonRay |
|---|---|---|---|
| Dragon from Modo, no HDRI, 1280×720, 16 samples/pixel | 9.1 s | 46.2 s (XPU) | mean ratio 1.002 |
| Polygon test scene, 360×200, 1,024 samples/pixel | 12.2 s | 122.8 s (XPU) | block error 0.0006 |
| Dose, 640×480, 64 samples/pixel, kit Source Overrides on | 9.5 s | 43 s (vector) | mean ratio 1.0006 |
| 9 Spheres, 640×480, 64 samples/pixel, kit Source Overrides on | 64 s | 176 s (vector) | mean ratio 0.9997 |
| Interior, 512×340, 64 samples/pixel | 51 s | 193 s (vector) | mean ratio 1.0003 |
| Material fixtures, first 10 of the final benchmark, 64 samples/pixel | 86–92% less time | | all pass |

Test conditions are in the [kit history](/xpu2/interactive/KIT_ROADMAP.md) and the
[indexer notes](/xpu2/interactive/INDEXER.md) (sections 14 and 15).

## Limits

Windows only. The graphics card needs to be an NVIDIA RTX 20 series or newer; older cards, or none, render on the processor. Still on the processor: volumes whose maps read image textures, light filters, shadow linking and lobe light sets. Still handed to MoonRay XPU: a scene past its memory ceiling. Deep output covers surfaces, with the beauty and alpha; not yet volumes' segments, other deep outputs, or deep files from Rhizome. The whole list is in [section 21 of the manual](/xpu2/manual/README.md#21-not-done-yet).

## Textures in XPU2 and MoonLightIPR

The two GPU engines in MoonRay for Modo handle textures differently. XPU2 keeps textures exact and loads only the pieces a frame samples, compressing only when memory runs short. MoonLightIPR always compresses and keeps whole textures on the graphics card, which suits a fast preview.

| | XPU2 | MoonLightIPR |
|---|---|---|
| **File the engine reads** | `.tx` (OpenImageIO's tiled, mipmapped format), made by `maketx` | Targa for 8-bit images, PFM for float ones, made by `oiiotool`; the session then caches its own prepared `.mlt` |
| **Smaller detail levels (mips)** | Stored in the `.tx` | Made by the session when it prepares the texture, then cached |
| **Compression** | None normally: bytes, half floats, or one channel for grey maps. BC6H (1 byte a texel) only in texture overflow | Always: BC7 for 8-bit images, BC6H for floats (Intel's ISPC compressor), about 1 byte a texel |
| **Lossy?** | No, outside overflow | Yes, slightly (BC7 and BC6H) |
| **Loading** | On demand: the small levels (128 a side and under) stay on the card; every larger level is read tile by tile, on the processor and the card, as frames sample it; tiles let go when memory runs short. Renders finish only once their tiles have landed, so pictures match loading everything | Whole texture uploaded to the card at load |
| **When textures don't fit** | Rare, since only what frames sample counts. Texture overflow mode: fit each texture to what its meshes show on screen, then compress, then halve | Keeps the largest textures in the computer's memory, outside the card |
| **Realtime size cap** | 1024 a side on the card, normal maps included | 1024 a side |
| **Render size** | Full size | Full size (Render and Render Animation) |
| **MoonShine graph bakes** | Not baked: graphs run natively as MoonRay maps | Baked to images: 512 or 2048 in realtime, the image's own size up to 8192 for a final Render |
| **Shader Tree procedural layers** | Through the object, the native `ModoProceduralMap`; flat projections use the C++ 2D bake | The same C++ bake for flat projections (512 in realtime, 2048 for a Render); the native map through the object |
| **Layered environments** | Composed natively, 2048 × 1024 | Composed natively, 1024 × 512 in realtime, 2048 × 1024 for a Render |
| **Texture conversion** | Side by side, up to 8 converters | Side by side, up to 8 converters |
| **Animations** | Overflow fitting sized over the whole shot, so no popping | No overflow fitting, so nothing to pop |

## Start here

| Topic | Read |
|---|---|
| How XPU2 works, with the files behind each part | [Engineer manual](/xpu2/manual/README.md) |
| The core in depth | [Asynchronous renderer](/xpu2/interactive/ASYNC.md), [indexer and scheduling](/xpu2/interactive/INDEXER.md) |
| The MoonRay port | [Path tracer and shaders](/xpu2/interactive/PATH_TRACER.md), [feature progress](/xpu2/interactive/FEATURE_PROGRESS.md) |
| Running XPU2 in Modo | [Manual: XPU2 in MoonRay for Modo](/xpu2/manual/README.md#13-xpu2-in-moonray-for-modo), [tools](/xpu2/tools/README.md) |
| Rendering across machines from Modo (Rhizome) | [Setting it up](https://github.com/Pbaroque20/xpu2-foundation/tree/xpu2-network#setting-it-up), [MoonRay for Modo's installation guide](https://github.com/Pbaroque20/MoonRayForModo/blob/MoonLightIPR/docs/INSTALLATION.md#rendering-across-machines-with-rhizome) |
| Building, and a frame rendered without Modo (`moonray_xpu2 render -in scene.rdla -out beauty.exr`) | [Manual: building and running](/xpu2/manual/README.md#14-building-and-running) |

## Repository layout

| Layer | Location |
|---|---|
| Core | `xpu2/interactive/`: work grid, broker, CPU and GPU lanes, integrator framework, sampling, texture tiers, indexer |
| MoonRay port | `xpu2/interactive/`: plan builder for MoonRay's RDL scenes, and the shading library (`shade_*.h`, `light_*.h`, `camera_core.h`, `volume_*.h`, `aov_*.h`) |
| MoonRay for Modo host | `xpu2/interactive/xpu2_session.cc`, `xpu2/tools/` (headless Modo tools), and the MoonRay for Modo tree (`native-port/`, `kit/`, ...) |
| Documentation | `xpu2/manual/` |
| History | `xpu2/*.md`, `xpu2/*.cc`: the first attempt (the October 16 bridge experiments) |

The core and the MoonRay port currently share a folder. The manual's file map assigns each file to its layer.

## Branches

- `MoonLightIPR`, in MoonRay for Modo: where XPU2 is developed, with its host.
- `xpu2-async` (this branch): XPU2's core and its documents, brought up to date from `MoonLightIPR`. Its kit stops at the 0.5 release.
- [`xpu2-network`](https://github.com/Pbaroque20/xpu2-foundation/tree/xpu2-network): Rhizome, XPU2 across machines
  (P2P, decentralized network rendering), on top of this core.
- `xpu-plus`: XPU+, the earlier synchronous approach (19–20 October 2026). It speeds up MoonRay XPU's own pipeline with
  exact savings that keep images identical, rather than replacing it. Kept for its measurements and techniques.

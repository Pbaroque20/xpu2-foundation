# Experimental XPU2 prototype

*XPU2: Asynchronous Monte Carlo Path Tracer designed by Raphael Tobar. Not affiliated with or endorsed by DreamWorks Animation; MoonRay and OpenMoonRay are DreamWorks Animation's.*

This `codex/xpu2-foundation` branch publishes an **experimental standalone Windows CPU/GPU renderer**, developed by Raphael Tobar with AI assistance. It is not the production Modo plugin and does not replace original MoonRay XPU.

The prototype combines supported GPU traversal and shading with CPU rendering, resident scene data, progressive updates, resumable paths and pinned-reference fallback. In two synthetic textured direct-lighting scenes, fresh 12-thread comparisons measured about **33% shorter warm render times than XPU**, with the stated quality gates passing. That excludes startup/loading and is not a general performance promise.

**Limits:** static flat triangles; diffuse/emissive materials; supported image maps and a constant environment. Textured scenes are direct-only; untextured transport supports up to eight surfaces. General materials/lights, full Modo integration, dynamic device work sharing and production qualification remain unfinished.


## How XPU2 samples

XPU2 renders a preview in **generations**. Every pixel first gets a few samples, a first look. Then each generation
looks at where the picture is still noisy and sends the next samples there, and parts that are already clean stop.
Plain areas such as a wall or the sky finish quickly; hard ones such as glossy reflections, soft shadows and glass get
more. Only where the samples go changes, not how each one is worked out, so the picture settles to MoonRay's.

The settings are in **Render Settings**, in their own group, XPU2 Sampling:

| Setting | What it does | Default |
|---|---|---|
| Generation Samples | What the render is worth, in samples a pixel on average | 16 |
| Halting Rule | **Time**: stop after about the time a plain render of that many samples takes, spent where the noise is. **Exact**: every pixel gets exactly that many | Time |
| Halt at Noise | A part of the picture stops once it is this clean (lower is cleaner, and slower) | 10 |
| First Look | Samples every pixel gets before the generations choose | 4 |
| Most Samples | The most one pixel may get | 16 times the average |
| CPU Workers | The processor works beside the graphics card | On |
| Realtime Samples | While RT follows the scene, the share of Generation Samples it takes: lower settles sooner, noisier | 100% |

These settings always decide how XPU2 samples a preview. MoonRay's Sampling Mode and its adaptive settings are for
MoonRay renders: while XPU2 renders the preview they are greyed and show what XPU2 takes, and final renders use them as
always. A bake camera, or a camera with a pixel sample map, is sampled evenly with Pixel Samples, as MoonRay does.

Which devices XPU2 uses is in System > Preferences > MoonRay > XPU2 > Devices: the processor and the graphics card
together (the default), the graphics card only, or the processor only, which needs no shaders compiled.

The line under the picture says how it samples: **XPU2 Gen. 16 ps avg (4-256), noise 10** is 16 samples a pixel on
average, 4 to 256 in any one pixel, each part stopping at noise 10. The MoonRay Terminal shows each generation as it
goes, with how much of the picture is still sampling.

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

## More

- [Documentation guide: current prototype versus historical first attempt](xpu2/README.md)
- [Prototype, features and limitations](xpu2/interactive/README.md)
- [Build requirements and run instructions](xpu2/interactive/BUILDING.md)
- [Measurements and validation evidence](xpu2/interactive/VALIDATION.md)
- [MIT license for original prototype code](xpu2/interactive/LICENSE) and [upstream license notices](xpu2/interactive/THIRD_PARTY_NOTICES.md)

For the released **MoonRay for Modo plugin**, use [codex/native-avx-modo](https://github.com/Pbaroque20/MoonRayForModo/tree/codex/native-avx-modo). Plugin development continues on [MoonLightIPR](https://github.com/Pbaroque20/MoonRayForModo/tree/MoonLightIPR). Earlier XPU2 experiments elsewhere in this branch are historical; the standalone prototype is under `xpu2/interactive/`.

# Shading batch size on Modo scenes, warm texture lookups, the rest of Init Intersection

Measured 2026-10-20 on the all-savings runtime (`REBASE_0_4_2_RESULTS.md`), release
0.4.2 base, RTX 3090 / i7-4930K, XPU mode, `-res 4`, 4 samples, one thread unless
noted. Nothing here is installed, enabled by default or committed.

## 1. Shading batches on Modo scenes

`shadeBundleHandler` cuts each material's sorted points into runs that share a
layer assignment, and shades and integrates each run as its own workload. The sort
key only distinguishes assignment ids up to 127; a Modo export gives every mesh
part its own assignment (thousands), so points of many assignments interleave and
the runs are a few points long. Most of each 8-lane block is then empty.

Average run length (`XPU2_SHADE_FIX_REPORT` counters):

| Scene | Points | Runs | Points per run |
| --- | ---: | ---: | ---: |
| Modo scene (Technik) | 4,526,667 | 977,294 | 4.6 |
| Modo scene, no fuzz | 3,683,680 | 746,642 | 4.9 |
| country_kitchen | 13,181,560 | 419,304 | 31.4 |
| contemporary_bathroom | 15,850,189 | 502,239 | 31.6 |

Assignments that use the same light set share the same light list, light-filter
list and light accelerator objects (`Scene::mIdToLightSetMap` points into one entry
per rdl2 LightSet). `XPU2_SHADE_MERGE_LIGHTSETS=1` (`xpu2/rndr_shade_fixes.py`, off
by default) continues a run across points whose three light objects are the same.

| Scene | Runs, merged | Points per run | MCRT off -> on (single runs) |
| --- | ---: | ---: | --- |
| Modo scene (Technik) | 142,607 | 31.7 | 146.0 -> 122.9 s (15.8% faster) |
| Modo scene, no fuzz | 115,941 | 31.8 | 118.8 -> 107.2 s (9.8% faster) |
| country_kitchen | 417,425 | 31.6 | 63.1 -> 62.6 s |
| contemporary_bathroom | 500,971 | 31.6 | 52.1 -> 51.8 s |

Alternating timing on the all-savings runtime with merging off and on (MCRT,
`test-results/batching/timing.txt`; GPU in its idle state throughout):

| Scene | Threads | Merge off | Merge on | Medians |
| --- | ---: | --- | --- | --- |
| Modo scene (Technik) | 1 | 139.19 / 138.49 / 138.39 s | 119.14 / 119.87 / 119.78 s | 138.49 -> 119.78 s, 13.5% faster (every run) |
| Modo scene (Technik) | 12 | 27.27 / 30.14 / 28.77 / 25.94 / 24.98 s | 20.41 / 21.27 / 21.36 / 23.01 / 28.79 s | 27.27 -> 21.36 s, about 22% faster (4 of 5 pairs) |
| Modo scene, no fuzz | 12 | 21.57 / 19.32 / 23.45 / 19.05 / 22.68 s | 23.43 / 22.60 / 21.07 / 26.70 / 21.64 s | 21.57 -> 22.60 s, inconclusive (2 of 5 pairs) |

At one thread the gain is clear on the Modo scene; at 12 threads these scenes vary
by up to 40% between identical runs, so one 12-thread result is a likely gain and
the other shows nothing either way.

### Not bit-exact

An 8-lane block computes one active light set and uniform sample counts for its
members (for example the light-sample Russian roulette sample numbers), so which
points share a block changes the random numbers a point uses. With merging the
image is a different, equally valid sample of the same picture. On the Modo scene
(same settings, `test-results/batching`):

| Pair | RMS difference | Mean abs. | Signed mean | Words differing |
| --- | ---: | ---: | ---: | ---: |
| XPU against XPU, identical runs | 0.00024-0.00026 | 0.00001 | <= 0.000001 | 8-11% |
| all savings (merge off) against XPU | 0.00026 | 0.00001 | 0.000001 | 11% |
| merge on against merge off | 0.00048 | 0.00004 | 0.000002 | 14% |

So merging moves the noise about twice as much as XPU's own run-to-run variation
on this scene, without a bias. Exact XPU parity (the fallback rule) rules it out
as a default.

### Decision and plan (2026-10-20): an opt-in XPU2-C1 option

The user decided: batch merging stays off by default and becomes an opt-in
option of XPU2-C1, described as statistically equivalent to XPU, not bit-exact.
Plan for when XPU2-C1 goes into the kit:

1. A separate checkbox in the XPU2-C1 preferences (for example "Merge shading
   batches across objects (faster on Modo scenes; images match XPU statistically,
   not bit for bit)"), off by default, passed to MoonRay as
   `XPU2_SHADE_MERGE_LIGHTSETS=1`. It is independent of the exact savings, which
   keep their own Source Overrides switches, on by default.
2. With the option off, XPU2-C1 keeps its bit-exact fallback to XPU; with it on,
   the fallback is statistical only, and the kit should say so where the option
   is shown.
3. Before release: an unbiasedness check over more scenes and sample counts
   (per-pixel means of many seeds with merging on and off agree within their
   noise), and 12-thread timing on several Modo scenes with more rounds, since
   the 12-thread results above vary up to 40% between runs.

## 4. Texture lookups

The example scenes' "Texturing (OIIO)" shares in the earlier survey (24-35%) came
from the first render after the files were out of the operating system's cache:

| Scene | Survey (cold) | Warm, two renders |
| --- | --- | --- |
| salle_de_bain | 35% of MCRT; 33.3 s OIIO file I/O, 7.0 s opening 3 files | 1.05 s, 2.2%; file I/O 0.1 s |
| the_wooden_staircase | 24%; 11.8 s file I/O | 2.9 s, 6.4%; file I/O 0.1-0.2 s |

The files are ordinary 64x64-tiled, zip-compressed mip-mapped TIFFs; OIIO's own
tool reads and rewrites the largest (4096x2048) in 0.8 s. Warm lookups are a few
percent and live inside OIIO's filtering; no exact saving there is worth its risk.

The Modo scene's textures go through the kit's `ModoTextureMap`, not MoonRay's
OIIO texture system ("No images opened"); its whole shading line is 9.4% of
MCRT, mostly the layered and skin materials themselves.

What would help the cold case is overlapping the file reads with scene loading
(about 40 s of text parsing on the Modo scene): reading the scene's texture files
in the background while MoonRay parses. That is exact (it only warms the OS file
cache) and belongs in the kit; not built here.

## 5. The rest of Init Intersection

After the instance-attribute and UV-reuse fixes, Init Intersection is 4.3% of MCRT
on the Modo scene (6.3 s) and 14.6% on country_kitchen (9.2 s). What remains is
MoonRay's generic interpolation of each attribute a material requests
(`Intersection::setRequiredAttributes`, `shading::Interpolator`): plain array reads
per attribute and vertex, about 140 ns per hit on the kitchen and 285 ns on the
Modo scene, plus curvature, epsilon and table setup. No further exact saving of
note was found; at most about 1-2% of a frame is left, so none was built.

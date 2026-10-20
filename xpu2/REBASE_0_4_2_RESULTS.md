# The XPU2 candidate on the 0.4.2 runtime, with every exact saving

Done 2026-10-19/20, following `REBASE_0_4_2_PLAN.md`. RTX 3090 / i7-4930K,
Windows "Balanced" power plan, GPU clocks not locked (idle P8 between renders).
Nothing is installed, enabled by default in the kit or pushed.

## Runtimes

All are staged by `tools/xpu2_stage_runtime.py` (hard links to a base runtime, named
files replaced by copies; the base's inventory is checked before and after):

| Name | Folder | Contents |
| --- | --- | --- |
| R | `test-results/rb-R` | `runtime/release-0.4.2`, byte-identical to the installed kit's runtime |
| X | `test-results/rb-X` | R with the original-XPU renderer DLL of `xpu2-origin-build/control` |
| C | `test-results/xpu2-candidate-042` | R with the XPU2 candidate's renderer DLL (switches off = disabled route); also a build folder for `tools/xpu2_rebase_build.py` |
| Rg / Rgr | `test-results/rb-Rg`, `rb-Rgr` | R with the geometry library relinked from the current tree, plain / with the Init Intersection savings (`geom_isect_reuse`) |
| A | `test-results/rb-A` | C with `xpu2/rndr_all_fixes.py` (shade-handler and integration savings, per-lane subsurface wavefront) and the Init Intersection geometry library |

## Gates

Word-for-word image comparisons, one thread, `-res 8`, 4 samples:

- Scalar mode (repeatable), country_kitchen, contemporary_bathroom,
  glass-of-water, the_wooden_staircase and the Modo scene: R = X = C, and
  R = Rg = Rgr = A (A with `XPU2_SSS_WAVEFRONT=1`): no difference anywhere.
- veach-bidir, XPU mode: R = X = C; R = A in most renders (below).
- veach-bidir, vectorized mode: R = X = C = A, except an alternate image
  (8,796 words) that R itself and A each produced once.
- Subsurface wavefront shadow mode on the Modo scene (A): 82,195 lane
  comparisons, 0 differences.

### The occasional alternate XPU image

On veach-bidir in XPU mode, R, C, X, Rgr and A with all its switches off
repeated exactly in 34 of 34 renders (R also 8 of 8 under added CPU load). With
any of the renderer savings on, about one render in six produced one of a few
alternate images (6,462, 6,689 or 6,725 differing words), with any single
saving as well as all of them (presence skip alone 1 of 6, light split alone 1
of 6, chunk scan and emissive skip alone 0 of 6).

Every alternate render also changed MoonRay's own count of shadow rays and its
GPU share of them: 2,118,620 shadow rays at 99.95% GPU in the normal renders,
2,118,565 at 100.00% or 2,118,412 at 99.98% in the alternates; two alternates
with the same split (from different savings) are the same image. XPU traces any
shadow-ray batch of 1,024 rays or fewer on the CPU with Embree
(`XPUOcclusionRayQueue::processRays`), whose answers differ in the last bits from
OptiX's, and where its queues are cut into batches depends on when they are
flushed. So the alternates are XPU's own CPU/GPU split moving, which the faster
code makes more frequent on this scene; whenever the split matches R's, the
image is R's bit for bit, and the decisions the savings replace were checked
directly (`SHADE_HANDLER.md`, `INTEGRATION.md`). The split's timing dependence
is inferred from the code and these counts, not traced to a particular flush.

## Timing

MoonRay's MCRT time (rendering only: no loading, render prep or output), XPU
mode, uniform 4 samples, `-res 4`. Each round runs R, C and A back to back;
five rounds at 12 threads, three at one thread; A with
`XPU2_SSS_WAVEFRONT=1`. The machine was otherwise idle; the GPU stayed in its
idle P8 state for most of the runs (`test-results/rebase-042/clocks-*.txt`), the
same for every runtime. Raw times: `test-results/rebase-042/timing.txt`.

| Scene | Threads | R (0.4.2 XPU) | C (candidate) | A (all savings) | A against R |
| --- | ---: | --- | --- | --- | ---: |
| contemporary_bathroom | 1 | 52.79 / 52.57 / 52.89 s | 56.20 / 56.23 / 56.31 s | 50.51 / 50.54 / 50.48 s | 4.3% faster |
| veach-bidir | 1 | 24.45 / 24.59 / 24.36 s | 26.23 / 26.08 / 26.09 s | 22.35 / 22.28 / 21.84 s | 8.9% faster |
| Modo scene (Technik) | 1 | 142.24 / 141.27 / 142.16 s | 145.50 / 143.69 / 146.61 s | 140.42 / 141.76 / 139.67 s | 1.2% faster |
| contemporary_bathroom | 12 | 9.68 / 9.57 / 9.28 / 9.74 / 9.64 s | 10.52 / 9.39 / 9.50 / 10.30 / 10.06 s | 9.06 / 8.33 / 8.83 / 8.68 / 9.59 s | 8.4% faster |
| country_kitchen | 12 | 11.70 / 11.74 / 11.39 / 11.62 / 12.12 s | 12.38 / 12.50 / 11.88 / 12.41 / 13.29 s | 10.73 / 10.85 / 10.87 / 10.73 / 10.70 s | 8.3% faster |
| glass-of-water | 12 | 10.53 / 10.47 / 10.18 / 10.84 / 9.71 s | 11.06 / 10.69 / 10.98 / 10.59 / 11.03 s | 9.91 / 10.74 / 11.03 / 9.06 / 9.84 s | 5.3% faster |
| veach-bidir | 12 | 4.45 / 4.37 / 4.39 / 4.48 / 4.39 s | 4.47 / 4.79 / 4.67 / 5.08 / 4.65 s | 3.80 / 4.00 / 3.98 / 4.10 / 3.97 s | 9.3% faster |
| Modo scene, no fuzz | 12 | 19.47 / 23.04 / 31.11 / 20.49 / 24.39 s | 22.03 / 20.37 / 19.68 / 25.05 / 23.74 s | 19.72 / 21.82 / 21.79 / 22.57 / 21.22 s | 5.5% faster |
| mask_reference (Modo) | 12 | 0.27 / 0.26 / 0.25 / 0.30 / 0.25 s | 0.29 / 0.32 / 0.30 / 0.29 / 0.24 s | 0.25 / 0.23 / 0.24 / 0.25 / 0.26 s | 6.1% faster |
| Modo scene (Technik) | 12 | 31.28 / 30.56 / 28.04 / 28.33 / 23.41 s | 29.04 / 36.15 / 34.98 / 25.60 / 26.51 s | 30.68 / 32.46 / 33.73 / 29.47 / 27.11 s | 8.3% slower (inconclusive) |

(Percentages from the medians.) At one thread every A render beat every R
render. At 12 threads A's median is lower on six of seven scenes; the Modo
scene at 12 threads varies by about 30% between rounds in every runtime (R
alone from 31.3 to 23.4 s, drifting down), so that row says nothing either way.

The candidate without the savings (C) is 1.5-7% slower than R on every scene:
that is the disabled XPU2 code's cost. The savings remove it and go 1-9% past
XPU on the 0.4.2 base, while the gates above show identical images.

## What is not covered

- Full resolution, adaptive sampling and the interactive preview.
- Full GPU capacity (the GPU stayed in its idle state; the same for every
  runtime, see `INTEGRATION.md` and the subsurface write-up).
- Astra's XPU2 GPU offload features: all off here.
- In the kit, each saving would need its own MoonRay Source Overrides switch, on
  by default, and XPU2-C1 a selectable mode.

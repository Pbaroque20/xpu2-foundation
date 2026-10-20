# Shade handler: where its own time goes, and two exact savings

Measured 2026-10-19 on the RTX 3090 / i7-4930K (6 cores, 12 threads), XPU mode,
on the XPU2 candidate (`test-results/xpu2-disabled-route-build`). The GPU's SM
clock was held at 1,500 MHz by the user (`nvidia-smi -lgc 1500,2100`) for the
timing runs. Nothing here is installed, enabled by default in the kit or
committed.

## The cost

MoonRay's "Shade handler (excl. isect+shading)" line, over the ten example
scenes (one thread, 320x180, 16 samples): 6.8-15.3% of MCRT. That is
`pbr::shadeBundleHandler`'s own work, without Init Intersection, shading,
texturing, AOVs or ray derivatives, which have their own lines.

## Stage profile

`xpu2/shade_profile.h` with `xpu2/rndr_shade_profile.py`: one call in 8 is
timed stage by stage (`XPU2_SHADE_PROFILE`, `XPU2_SHADE_PROFILE_RATE`). Stages
that can run other handlers through queue flushes (radiance queue, presence
rays, integration) are inclusive and left out below. Each stage also carries
about 25-30 ns of timer per point. Built onto the candidate with
`tools/xpu2_rebase_build.py`, which now takes `ShadeBundleHandler.cc` from the
candidate's own patched copy (it carries the XPU2 GPU-owned branch, which is
unused with XPU2 off and left unmarked).

Exclusive stages, ns per shaded point, one thread (bathroom / kitchen /
glass-of-water / veach-bidir / Modo scene):

| Stage | ns per point |
| --- | --- |
| presence (material presence + stochastic presence test) | 225 / 222 / 233 / 217 / 428 |
| sort key (UV Morton code, mip selector, udim) | 108 / 109 / 108 / 108 / 107 |
| ray states AOS to SOA | 91 / 94 / 93 / 98 / 166 |
| resort (ray-state pointers per chunk) | 67 / 66 / 65 / 64 / 83 |
| chunk setup (light lists, run of equal light set) | 35 / 45 / 73 / 121 / 43 |
| intersections AOS to SOA | 39 / 41 / 43 / 45 / 92 |
| free ray states | 34 / 35 / 34 / 33 / 107 |
| sort | 13 / 12 / 11 / 11 / 11 |

Two of these are wasted work:

- Presence. For a path whose throughput is low, MoonRay decides presence
  stochastically: it draws a 1D integrator sample and keeps the surface when
  sample < presence. Almost every surface has presence 1, where the outcome is
  always "keep", but the sample (a sequence hash, a table lookup or a hash
  fill of 8 values, and a Cranley-Patterson rotation) is still drawn.
- Chunk setup. Each chunk of up to 64 points scans forward for the run of
  points with the same light-set assignment, to the end of the call, although
  only the first 64 are used. On a call of 3,500 points that is quadratic.

## The savings (`xpu2/rndr_shade_fixes.py`)

- `XPU2_SHADE_PRESENCE_SKIP` (on unless `0`): at presence >= 1 the result is
  1, at presence <= 0 it is 0, without the draw. Exact because every 1D
  integrator sample is in [0, 1): the precomputed table
  (`bc_1D_integrator_points_4096_sequences_of_1024.bin`, 4,194,304 values)
  holds values in [3.5e-7, 0.99999994]; the Cranley-Patterson rotation maps
  any input below 1 to [0, 1); the R1 fallback (`x - floor(x)` of a
  non-negative value) and the random engine past depth 3 (`bitsToFloat`, at
  most 1 - 2^-23) stay below 1. NaN and values strictly between 0 and 1 take
  the original path. Nothing else reads the sample.
- `XPU2_SHADE_CHUNK_SCAN` (on unless `0`): the scan stops after the chunk size;
  the chunk it yields is the same.

Both are applied to MoonRay's chunk loop and to the candidate's GPU-owned copy
of it. If they go into the kit, each needs its own MoonRay Source Overrides
switch, on by default.

### Exactness

`XPU2_SHADE_FIX_CHECK=1` still computes the original answer at every shortcut
and counts disagreements (`test-results/shade-fixes/check-*.csv`), one thread:

| Scene | Presence shortcuts | Still drawn | Disagreements | Chunks shortened | Disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| country_kitchen | 10,928,357 | 194,026 | 0 | 404,431 of 419,274 | 0 |
| contemporary_bathroom | 14,664,169 | 0 | 0 | 486,909 of 502,333 | 0 |
| glass-of-water | 15,167,962 | 0 | 0 | 530,275 of 537,315 | 0 |
| veach-bidir | 6,229,775 | 0 | 0 | 227,456 of 229,663 | 0 |
| Modo scene | 4,044,542 | 0 | 0 | 939,581 of 986,008 | 0 |

A whole-image comparison cannot test this: scalar mode does not use this
handler, and XPU and vectorized renders are not bit-repeatable on these scenes
(see `REAL_SCENES_AND_INIT_INTERSECTION.md`). Since every decision the
shortcuts replace came out identical, the rest of the frame is unchanged.

### Profile with the savings

Presence fell to 100-124 ns per point (306 on the Modo scene), chunk setup to
8 ns (54 on the Modo scene). MoonRay's own shade-handler line, profiled
builds, one thread: bathroom 8.09 -> 5.89 s, kitchen 6.93 -> 5.27 s,
glass-of-water 9.56 -> 6.80 s, veach-bidir 4.54 -> 2.91 s, Modo scene
4.38 -> 3.92 s.

### Timing

Same runtime, switches off (`=0`) and on, alternating, three rounds, MCRT
(`test-results/shade-fixes/timing.txt`; scenes at `-res 4`, 4 samples):

| Scene | Threads | Off | On | Medians |
| --- | ---: | --- | --- | --- |
| contemporary_bathroom | 12 | 9.54 / 9.82 / 9.89 s | 9.25 / 10.51 / 9.09 s | 9.82 -> 9.25 s (5.8%) |
| glass-of-water | 12 | 10.62 / 12.25 / 11.80 s | 10.51 / 10.27 / 10.55 s | 11.80 -> 10.51 s (10.9%) |
| veach-bidir | 12 | 4.36 / 4.67 / 4.82 s | 4.30 / 4.38 / 4.98 s | 4.67 -> 4.38 s (6.2%) |
| country_kitchen | 12 | 13.51 / 12.64 / 12.37 s | 12.44 / 12.38 / 12.60 s | 12.64 -> 12.44 s (1.6%) |
| Modo scene | 12 | 27.48 / 32.15 / 34.26 s | 23.84 / 29.07 / 31.77 s | 32.15 -> 29.07 s (9.6%, drifting) |
| contemporary_bathroom | 1 | 61.02 / 59.65 / 60.19 s | 59.86 / 57.72 / 58.02 s | 60.19 -> 58.02 s (3.6%) |
| veach-bidir | 1 | 28.45 / 30.12 / 29.37 s | 27.35 / 27.32 / 27.37 s | 29.37 -> 27.35 s (6.9%) |

18 of the 21 alternating pairs favour the savings (12 of 15 at 12 threads,
all 6 at one thread). The 12-thread runs are noisy (the Modo scene drifts upward across rounds
in both settings), so the 12-thread percentages are indicative.

## What is left

The other stages are mostly memory traffic on large records (ray-state and
intersection transposes, resort, frees). The sort key's mip selector (an
out-of-line `log2f`) could be replaced by an exponent read with a fallback
near powers of two, but it would need an exhaustive float test to prove the
same sort keys, for about 1% at most.

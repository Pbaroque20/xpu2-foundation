# Bundled integration: where its own time goes, and two exact savings

Measured 2026-10-19 on the RTX 3090 / i7-4930K (6 cores, 12 threads), XPU mode,
on the XPU2 candidate (`test-results/xpu2-disabled-route-build`), one thread
unless noted; GPU SM clock held at 1,500 MHz by the user. Nothing here is
installed, enabled by default in the kit or committed.

## The cost

MoonRay's "Integration" line is 17-42% of MCRT over the example scenes, the
largest CPU line. In XPU mode it is almost all `PathIntegrator_integrateBundled`
(ISPC): per 8-lane block, lights hit by camera rays, volumes, cutouts,
emission, the active light set, subsurface, emissive volume regions, the
radiance queue, then `computeRadianceBsdfMultiSampler` (BSDF samples, light
samples, shadow rays, bounce rays).

## Stage profile

`xpu2/integ_profile.cc` with `xpu2/rndr_integ_profile.py` puts `extern "C"`
marks into the candidate's own copies of `PathIntegratorBundled.ispc`,
`PathIntegratorMultiSampler.ispc` and `PathIntegratorUtil.ispc`
(`tools/xpu2_rebase_build.py` now compiles ISPC with the build tree's flags).
One call in 4 is timed at any nesting depth; a queue flush can run whole
handlers and nested integration calls, so a nested call's time is taken out of
its parent's stage, and stages that flush (`*_incl`) are inclusive.

The exclusive stages add up to MoonRay's own line, which checks the method
(estimated from the sample / MoonRay's Integration line): bathroom 21.8 / 20.0 s,
kitchen 23.1 / 21.5 s, glass-of-water 22.2 / 20.9 s, veach-bidir 9.7 / 8.8 s
(the difference is the marks' own cost). Per integrated lane
(bathroom / kitchen / glass-of-water / veach-bidir / Modo scene,
`test-results/integ-fixes/profile-deep-summary.txt`):

| Stage | ns per lane |
| --- | --- |
| BSDF sampling (sample generation and lobe sampling) | 191 / 179 / 191 / 128 / 824 |
| Bounce rays (path vertex, ray, differentials, light-set lists) | 200 / 195 / 174 / 128 / 811 |
| Shadow-ray preparation for light samples | 153 / 210 / 74 / 228 / 758 |
| Light sample: BSDF evaluation and MIS | 115 / 268 / 119 / 121 / 1,132 |
| BSDF sample direction: light intersection | 110 / 127 / 113 / 81 / 1,792 |
| Light sample: sample and evaluate the light | 67 / 150 / 169 / 98 / 1,624 |
| Entering `drawLightSamples` (call and stack frame) | 53 / 155 / 60 / 78 / 406 |
| Emissive volume regions | 59 / 60 / 55 / 55 / 100 |
| Light-sample Russian roulette | 31 / 72 / 13 / 49 / 175 |
| XPU2 light-mode setup in `drawLightSamples` | 13 / 29 / 13 / 17 / 89 |

(The Modo scene integrates about 5 lanes per call, against 31-32 elsewhere,
so its per-lane figures are higher.)

Most of this is the integrator's real work. Two parts are not:

- Emissive volume regions: for every block with lobes, the integrator calls
  into C++, which for each lane stops and starts two profiling accumulators
  and returns black when the scene has no emissive volumes (none of these
  scenes has any). The ISPC side adds that black into a local that starts
  black.
- Entering `drawLightSamples`: the candidate's copy carries the XPU2
  light-group paths, with about 100 KB of stack arrays, allocated and probed
  on every call (once per light per block) even when that mode is off.
  Upstream's function has no such frame. This is part of the candidate's
  disabled-route cost against original XPU.

## The savings (`xpu2/rndr_integ_fixes.py`, `xpu2/integ_fixes.cc`)

- `XPU2_INTEG_EMISSIVE_SKIP` (on unless `0`): the call is skipped when the
  scene has no emissive regions; the local stays black, so the radiance is the
  same.
- `XPU2_INTEG_LIGHT_SPLIT` (on unless `0`): `drawLightSamples` decides the
  XPU2 light mode first, as the candidate does, and calls either the full
  candidate function (mode on, or switch off; marked `noinline`) or a copy
  without the mode's code (mode off). With the mode off the full function runs
  exactly the copy's statements. ISPC here does no floating-point
  reassociation or contraction (no fast-math flag; AVX1 has no FMA), so the
  results are the same.

`xpu2/rndr_cpu_fixes.py` combines these with the shade-handler savings
(`SHADE_HANDLER.md`).

### Exactness

veach-bidir repeats bit for bit at one thread in vectorized and XPU mode,
which makes it a whole-image gate (the textured scenes do not repeat; see
`REAL_SCENES_AND_INIT_INTERSECTION.md`). At `-res 8`, 4 samples:

- integration savings on against off: identical in vectorized and XPU mode;
- all four savings (`rndr_cpu_fixes`) on against off: identical in vectorized
  mode; in XPU mode two of three "on" renders are identical to "off" and to
  original MoonRay 0.4.2 XPU, and the third differs in about 6,400 words.
  Repeat renders of "off" matched each other three times, but XPU itself is
  not always repeatable on this scene (that third render also differs from
  the other two "on" renders, which are identical to each other), so that
  render is XPU's own variation, not the savings;
- the candidate with the savings off is identical to original 0.4.2 XPU.

### Timing

The reference that matches the candidate is `test-results/xpu2-origin-build/control`:
the same runtime with the original XPU renderer DLL (the candidate's geom and
shading DLLs predate release 0.4.2, so release 0.4.2 is not the same base).
Three alternating rounds, MCRT medians, XPU mode, scenes at `-res 4`, 4
samples, all four savings on or off (`test-results/integ-fixes/timing-control-off-on.txt`):

| Scene | Threads | Original XPU (control) | Candidate, savings off | Candidate, savings on | On against XPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| contemporary_bathroom | 1 | 62.18 s | 66.84 s | 61.07 s | 1.8% faster |
| veach-bidir | 1 | 28.68 s | 30.87 s | 27.17 s | 5.3% faster |
| contemporary_bathroom | 12 | 9.32 s | 9.78 s | 9.34 s | even |
| glass-of-water | 12 | 9.84 s | 10.11 s | 10.14 s | 3% slower |
| veach-bidir | 12 | 4.41 s | 4.73 s | 4.37 s | 1% faster |
| country_kitchen | 12 | 11.12 s | 12.07 s | 11.38 s | 2% slower |

At one thread the candidate without the savings is 7-8% slower than original
XPU (the disabled XPU2 code, mostly the light-sampling stack frame), and the
savings recover that and go past XPU. At 12 threads the result is parity
within the noise (single runs vary 5-15%).

An earlier run in the same layout against release 0.4.2 (with the GPU's SM
clock locked at 1,500 MHz; `timing-xpu-off-on.txt`) gave, medians, original /
off / on: bathroom one thread 54.72 / 58.22 / 53.82 s, veach-bidir one thread
25.42 / 27.40 / 24.39 s; 12 threads bathroom 8.97 / 9.09 / 8.59 s,
glass-of-water 10.27 / 10.10 / 9.77 s, veach-bidir 4.16 / 4.46 / 4.04 s,
country_kitchen 10.93 / 11.60 / 10.87 s; the Modo scene (27.6 / 31.0 / 31.3 s)
and country_kitchen at one thread drifted during the run and are not usable.
The two runs were taken under different machine conditions, so their
absolute times are not comparable with each other.

Not in these builds: the Init Intersection savings (`librendering_geom.dll`,
about 5% at one thread on the kitchen) are built from the current build tree,
which is at 0.4.2, while the candidate's geom DLL is older, so they cannot be
dropped into the candidate's runtime without changing its MoonRay version; and
the per-lane subsurface wavefront (`XPU2_SSS_WAVEFRONT=1`, off by default),
which only affects random-walk subsurface scenes.

# Plan: the XPU2 candidate on the 0.4.2 runtime, with every exact saving

Written 2026-10-19; carried out 2026-10-19/20, see `REBASE_0_4_2_RESULTS.md`.

## What the candidate is built on today

- Its renderer DLL (`test-results/xpu2-disabled-route-build/runtime/librendering_rndr.dll`)
  is 199 objects from a snapshot of `build/native-renderer-avx` (16 October) plus 10
  patched objects (PathIntegratorBundled, PathIntegratorMultiSampler and
  PathIntegratorUtil ISPC; PathIntegratorUtil.cc, PbrTLState, ShadeBundleHandler,
  RenderContext, RenderDriver, RenderFrame, RenderFramePasses).
- 208 of the 209 snapshot objects are still identical to the build tree's; the one
  that changed, RenderContext, is replaced by the candidate's own copy, which is the
  current upstream file plus three XPU2 hunks. So the renderer code is current.
- The rest of its runtime is not: `librendering_geom.dll` and
  `librendering_shading.dll` come from the 0.3.50/0.3.51 runtimes, and 23 files
  differ from release 0.4.2 (geom, shading, rndr, dwabase, glitter,
  UsdPreviewSurface, the primvar and projection maps, noise, interpolation,
  application, VdbGeometry, `moonray.exe`, kit tools). Its reference runtime,
  `xpu2-origin-build/control`, is the same mixture with the original renderer DLL.
- The installed kit runtime is byte-identical to `runtime/release-0.4.2`. The build
  tree's `bin` has moved on since (its rndr, geom, scene_rdl2 and `moonray.exe`
  differ from the release).

So the timings so far compare like with like (candidate against `control`), but
not against what ships, and the Init Intersection savings (built from the current
tree's geom objects) cannot go into the old runtime without also changing its
geometry library version.

## Steps

1. Reference runtime R: a hard-linked copy of `runtime/release-0.4.2` (the kit's
   runtime). Record its inventory; check nothing writes into it.
2. Original-XPU twin X: R with the renderer DLL relinked from the same 209
   snapshot objects without the XPU2 patches (the `control` DLL, or a fresh
   baseline relink). Gate X against R: veach-bidir bit for bit (vectorized and
   XPU, one thread); scalar mode on the four textured scenes and the Modo scene.
   If X and R differ, the build tree's renderer has drifted from the release and
   X, not R, is the reference for "same code, with or without XPU2".
3. Candidate C: R with the candidate's renderer DLL. Gate C (XPU2 off) against X
   the same way, plus the frozen held-out gate fixtures (`tools/xpu2_heldout_gate.py`)
   and the XPU2 acceptance checks Astra used (`xpu2/XPU2_ACCEPTANCE.md`).
4. A new candidate build folder for `tools/xpu2_rebase_build.py` (`link.rsp`, the
   patched sources and `report.json` from the current candidate, runtime = C), so
   the patch modules build onto it unchanged.
5. Renderer savings onto C: `rndr_cpu_fixes` (shade handler and integration), then
   the per-lane subsurface wavefront (`rndr_sss_wavefront`, off unless
   `XPU2_SSS_WAVEFRONT=1`). Same gates; merged-shadow and shadow mode for the
   subsurface path on the Modo scene.
6. Geometry savings: `tools/xpu2_geom_build.py --dll geom --variant geom_isect_reuse`
   from the current tree, compared first as a baseline relink against R's geom DLL
   (they differ in bytes today; confirm the renders do not). Gate in scalar mode
   (repeatable) on the textured scenes and the Modo scene, word for word.
7. All-in runtime A = C + step 5 + step 6. Gates as above.
8. Timing: R (or X), C with savings off, A; alternating, five rounds at 12 threads
   and three at one, on the five scenes used so far plus two more Modo scenes;
   GPU and CPU clocks sampled during each run, power plan recorded. Optionally a
   second pass at full capacity, if you set Windows' High performance plan and lock
   the GPU clocks (your settings).
9. Write-up and commit on the XPU2 branch; nothing pushed.

## Decisions and risks

- If step 2 shows the build tree's renderer has drifted from the release, the fair
  reference is X; the kit would need a renderer rebuild anyway to carry any of
  this, so that comparison is the one that matters for shipping.
- Each saving that goes into the kit later needs its own MoonRay Source Overrides
  switch, on by default (they are environment switches now).
- Astra's candidate chain (`xpu2/native_build.py`, the untracked
  `xpu2/native_disabled_route.py`) is not rerun; only the runtime around the
  renderer DLL changes. If the gates in step 3 fail, rerunning that chain against
  the current tree is the fallback, and Astra should know the candidate's runtime
  base was older than 0.4.2.
- Estimated machine time: about 3-4 hours of renders, mostly step 8.

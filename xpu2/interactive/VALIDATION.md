# Validation scope — 22 October 2026

> **A record of 22 October 2026.** It describes the prototype as published that day; "current source" below meant that day's. For XPU2 as it is now, see the [README](README.md) and [feature progress](FEATURE_PROGRESS.md).

Publication is justified as an experimental prototype by finite measured improvements, not by production qualification. Gates were fixed before timing. Numerical reports are in [evidence/](evidence/); only workstation path prefixes were redacted. [The manifest](evidence/manifest.json) records original and redacted report hashes. Raw images and binaries are not included.

## Fresh original-XPU comparison after the ray-hit correction

RTX 3090, i7-4930K, 64 samples/pixel. One warmup and three measured runs per scene/thread count. Actual original XPU execution was checked. Original XPU runs used fresh processes with warmed caches; native runs retain scene resources. MCRT time is compared with native render-window wall time, excluding preparation and output.

- gallery-dense, 1 worker: XPU 26.04219 s; XPU2 14.41698 s; 44.64% shorter.
- gallery-dense, 12 workers: XPU 4.09532 s; XPU2 2.76256 s; 32.54% shorter. Native measured range 2.70136–4.29223 s, including the slower outlier.
- workshop-detail, 1 worker: XPU 44.45100 s; XPU2 26.52834 s; 40.32% shorter.
- workshop-detail, 12 workers: XPU 6.88182 s; XPU2 4.60464 s; 33.09% shorter.

[Full comparison](evidence/interactive-ray-hit-benchmark-20261022.json). These are two synthetic flat-triangle, diffuse/emissive, textured direct-lighting scenes with a constant environment. They are not representative of all MoonRay or Modo features.

The prior residency candidate failed the strict CPU/GPU ownership gate on a few pixels. [That failure is retained](evidence/interactive-residency-ownership-20261022.json). The shared ray-hit correction subsequently passed unchanged gates: maximum CPU/GPU differences 2.98e-8 and 4.47e-8; held-out seeds and 1024-sample checks passed. [Corrected ownership evidence](evidence/interactive-ray-hit-images-shared-20261022.json). Original XPU itself varied by about .00015 across 12-worker images in the larger fixtures; no universal repeatability guarantee is implied.

## Current source: shader compilation and regression checks

The shared immutable sampler table reduces PTX from 1,027,565 to 789,976 bytes. Across three independently empty private OptiX caches per build, median module compilation fell 14.460816 -> 10.3110377 s (28.7%). Native preparation fell 15.6192171 -> 11.5492625 s. OS and other driver caches were not purged. This is not full GUI or Modo startup latency.

Warm render medians against the immediately preceding XPU2 build, one warmup plus three measurements:

- Gallery, 1 worker: 13.3350564 -> 12.3375016 s.
- Gallery, 12 workers: 2.5246237 -> 2.2106625 s.
- Workshop, 1 worker: 21.9766398 -> 21.7685261 s.
- Workshop, 12 workers: 3.9935639 -> 3.932024 s.

All four <=5% regression gates passed, and the >=20% cold-compilation reduction gate passed. Small speed differences may be noise. These latest numbers are **not** a new paired original-XPU comparison. All 32 timed images and four single-device ownership images matched the saved native float32 RGBA exactly, preserving the preceding image-quality result.

[Timing/quality report](evidence/interactive-shader-qualification-20261022.json). [Recovery report](evidence/interactive-shader-recovery-20261022.json) covers 16,777,216 exact sampler words/conversions, six untextured spatial fixtures up to eight surfaces, planar progressive rendering, missing textures, bounded cache refill and CPU migration, cancellation, invalid continuation rejection, final controller operation and exactly owned original-XPU fallback pixels. This targeted sampler check is not a new billion-tuple qualification.

The published renderer sources differ from the validated local sources only in text encoding/line endings. Historical Python scripts replace a machine-specific toolchain path with `XPU2_TOOLCHAIN_ROOT`; documentation and packaging are revised for publication. [Source inventory](source-publication.json) records those differences. No arithmetic, filtering, ray-hit, material, scheduler or render-default changes were made for publication.

## Open gates

General material/light/geometry/AOV coverage, secondary textured derivatives, dynamic worker donation, page-selective texture residency, broad hardware/scene validation, clean-machine installation and reproducible reference rebuilding remain open. Native output remains experimental and globally unqualified. Unsupported work uses the pinned full-scene fallback; original XPU is not replaced or modified.

## Publication-copy checks

The isolated public-source checkout passed 54 unit tests and the native build. Its GPU PTX matches the validated local build byte for byte. Source-inventory and upstream-source/license hashes were verified. A fresh controller run passed progressive output, texture-cache reuse, CPU migration, injected-failure reference fallback and shutdown. See [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json). These checks do not change the qualification limits above.

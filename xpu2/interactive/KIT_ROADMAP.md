# XPU2 in MoonRay for Modo: history to 24 October 2026

A record. On 24 October this page was the working roadmap for making XPU2 Modo's preview renderer, on a separate test
kit built beside the standard one. That has since been done: XPU2 ships in the standard kit in 0.5, a pre-production
release, as the default GPU renderer for previews. For how it works now see the
[engineer manual](../manual/README.md) (section 13); for each feature's state,
[feature progress](FEATURE_PROGRESS.md).

## Where it stood that day

- **Working:** the kit's preview protocol, the fallback to MoonRay XPU (also when XPU2 crashed), the shader cache, quads
  and n-gons, mesh parts, the work grid's cells as outlines.
- **What stopped real scenes:** the dragon scene was refused before drawing anything, its preview being MoonRay's. HDRI
  environments, denoiser guides, texture-bound presence, motion blur, depth of field, subdivision, VDB volumes,
  light-path AOVs and Cryptomatte were not ported then. `FEATURE_PROGRESS.md` records what each came to.
- **Decided, not started:** sampling by cellular generations as the default, now done (`INDEXER.md` section 11).

## Measured that day

| Scene | XPU2 (CPU + GPU) | MoonRay XPU | Picture |
|-------|------------------|-------------|---------|
| Polygon test scene, 360x200, 1024 samples | 12.2 s | 122.8 s | Block error 0.0006 against MoonRay |
| Dragon from Modo, no HDRI, 1280x720, 16 samples | 9.1 s | 46.2 s | Mean ratio 1.002 |
| Material fixtures (first 10 of the final benchmark, 64 samples) | 86-92% less time | | All pass |

On an RTX 3090 and an i7-4930K. The final material benchmark stopped at 10 of its 59 cases, hence "first 10".

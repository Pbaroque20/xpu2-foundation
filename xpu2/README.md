# XPU2: prototype and research

*XPU2: Asynchronous Monte Carlo Path Tracer designed by Raphael Tobar. Not affiliated with or endorsed by DreamWorks Animation; MoonRay and OpenMoonRay are DreamWorks Animation's.*

> **XPU2 ships in MoonRay for Modo 0.5**, a pre-production release, as the default GPU renderer for previews. Start with
> the [engineer manual](manual/README.md). The sections dated below describe what it grew from.

**Start with [XPU2 as it is now](interactive/README.md).** This folder holds XPU2 in `interactive/` and two earlier efforts: the October 16 tests began the **first attempt**, which is now historical, and the standalone prototype of 22 October is what the present renderer grew from.

## 22 October: the standalone CPU/GPU prototype

The prototype combined supported CPU/GPU rendering, resident scene data, progressive output, resumable paths and pinned-reference fallback. As published that day it was Windows source, not an installed Modo feature or a production replacement for XPU.

1. [XPU2 now: features and limits](interactive/README.md)
2. [Build requirements and running the preview](interactive/BUILDING.md)
3. [October 22 measurements and validation limits](interactive/VALIDATION.md)
4. [Numerical evidence and report hashes](interactive/evidence/manifest.json)
5. [Publication build and recovery checks](interactive/PUBLICATION_CHECKS.json)
6. [Original-code MIT license](interactive/LICENSE) and [upstream notices](interactive/THIRD_PARTY_NOTICES.md)

The comparison published that day measured about 33% shorter warm render time at 12 threads on two restricted synthetic textured direct-lighting scenes. Startup/loading are excluded; this is not a general scene or end-to-end speed claim. Native rendering used the newer spec's statistical-equivalence gates; exact checks of particular kernels or fallback pixels do not imply universal bit-exact parity with XPU.

## First attempt: October 16 tests and subsequent bridge experiments

The first attempt explored offloading individual operations from MoonRay's native shading path. Its repeatability tests, exact-parity gates, queue experiments and overhead measurements are historical findings. They must not be read as the status or measured performance of XPU2 as it is now.

- [Historical overview of the first attempt](HISTORICAL_OVERVIEW.md): preserved original narrative and initial October 16 measurements.
- [Research index by topic](RESEARCH_INDEX.md): all first-attempt reports, including later follow-ups.
- [First-attempt repeatability investigation](REPEATABILITY.md).
- [First-attempt material-cost ceiling](NATIVE_MATERIAL_CEILING.md): the earlier bridge's cost analysis, not the present renderer's benchmark.

## Where the code lives

- [`interactive/`](interactive/): XPU2 as it ships in 0.5: the renderer, its session, tests, build tools and published evidence, with the 22 October prototype's sources beside them.
- Files directly in `xpu2/`: first-attempt bridge code, protocol models, probes and historical reports. Some shared dependencies remain in use; historical classification does not mean every source file is obsolete.
- [`reference/`](reference/): first-attempt pinned reference data. `interactive/` uses its own [reference lock](interactive/reference-lock.json).

Reports retain their original paths so existing GitHub links continue to work. Historical acceptance criteria and unfinished plans describe their own experiment; consult the [manual](manual/README.md) for today's scope.

# Building and running XPU2

XPU2 is built from source with the existing **native Windows** toolchain; this is not a clean-machine installer. No CUDA/OptiX SDK, Windows renderer, compiled PTX, user scenes, or DLLs are bundled here. The older code elsewhere on this branch is not a substitute for the required build dependencies.

## Prerequisites

The builder reads an existing MoonRay Windows development tree supplied with `--source`, containing:

- `toolchain/msys64/ucrt64/bin/g++.exe` and its matching libraries/headers.
- `build/native-avx/build.ninja`, the compiled scene_rdl2 dependencies and link libraries referenced there.
- Embree 4.4.1, OpenImageIO 3.1.17.0 and the corresponding runtime DLLs.
- `toolchain/xpu/optix-dev/include`.
- `toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include` and `toolchain/xpu/cuda_nvrtc-windows-x86_64-12.8.93-archive/bin`.
- A compatible NVIDIA GPU/driver. Measurements used an RTX 3090. No wider hardware qualification is claimed.
- Python 3; the optional GUI uses PySide6. Unit checks below use the standard library.

The `--runtime` folder must match `reference-lock.json`, which pins 1,135 files from the tested Windows reference. The build fails closed if they differ. The lock and hashes are included, but those binaries are not. Do not weaken the verification or overwrite the lock to make a different runtime appear validated. An independently rebuilt reference requires a new, separately documented pin and all image/performance gates rerun. Reference build provenance is binary-pinned; a clean source-to-binary rebuild remains unverified.

## Build the renderer and its session

From the repository root, substitute your actual development-tree and pinned-runtime locations. Every output directory must be new. The steps after these (compiling the GPU programs, packaging) are in [the manual, section 14](../manual/README.md#14-building-and-running).

```powershell
$toolchainRoot = 'D:/MoonRayForModo-dev'
$referenceRoot = 'D:/references/moonray-042'
python xpu2/interactive/build_async.py --source $toolchainRoot --runtime $referenceRoot --output test-results/async-build-N
python xpu2/interactive/build_session.py --source $toolchainRoot --runtime $referenceRoot --build test-results/async-build-N --output test-results/xpu2-session-N
```

## The 22 October prototype and its preview

The standalone prototype these tools build is kept as a record; it is not what the kit runs.

```powershell
python xpu2/interactive/build_prototype.py --source $toolchainRoot --runtime $referenceRoot --output test-results/my-xpu2-build
python xpu2/interactive/prepare_prototype_scene.py --scene 'D:/scenes/example.rdla' --build test-results/my-xpu2-build/build.json --runtime $referenceRoot --output test-results/my-scene
python xpu2/interactive/prototype_preview.py --build test-results/my-xpu2-build/build.json --runtime $referenceRoot --output test-results/my-preview --manifest test-results/my-scene/frozen/manifest.json --fine-textures --publish-ms 33
```

In the prototype's preview, select the admitted domain and opt into its native preview. Without that opt-in, rendering uses reference XPU. The GUI conversion is sRGB on black with clipped highlights; comparisons use separate raw linear float32 values, never display pixels.

## Checks and reproduction

```powershell
python -m unittest discover -s xpu2/interactive -p 'test_*.py'
python xpu2/interactive/build_startup_probe.py --source $toolchainRoot --runtime $referenceRoot --output test-results/my-timing-build
python xpu2/interactive/prepare_large_sampling.py --runtime $referenceRoot --build test-results/my-xpu2-build/build.json --output test-results/my-large-scenes
```

`prepare_large_sampling.py` generates the synthetic geometry and textures; it does not require personal scene assets. Qualification runners accept explicit build manifests and fresh output directories. Run their `--help` for inputs. Some historical experiments require the earlier generated result tree and dated filenames; they are not a one-command benchmark from a fresh clone. Set `XPU2_TOOLCHAIN_ROOT` for historical scripts that previously embedded a local development path. No historical result directory is silently fabricated or treated as a passing test.

The published evidence has workstation paths replaced with placeholders. Those report files are evidence, not executable build manifests. Raw EXR/RGBA files, compiler caches and intermediate results are omitted; their recorded hashes remain. Use newly generated manifests to run fresh checks. SDK/runtime acquisition and a portable clean-machine build are still future work.

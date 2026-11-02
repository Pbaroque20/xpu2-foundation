"""Build the asynchronous renderer's worker (async_probe.exe) and its OptiX program.

build_async.py --source <MoonRayForModo development tree> --runtime <pinned reference> --output <new folder>

The OptiX program is compiled for speed (contraction allowed); the parity
pipeline's program keeps its exact flags (gpu_build.PARITY_FLAGS). Never installs
or publishes anything."""
import argparse
import json
import shutil
from pathlib import Path

from gpu_build import compile_ptx
from native_build import bc6h_kernels, build_units
from reference_runtime import sha, verify

HERE = Path(__file__).resolve().parent
HEADERS = ('../../native-port/maps/ModoProcedural/modo_patterns.h', 'async_gpu_abi.h', 'points_core.h', 'async_adaptive_core.h', 'async_integrator.h', 'light_core.h', 'light_image.h', 'light_tree.h', 'gpu_jobs.h', 'resume_core.h', 'triangle_hit.h',
           'texture_core.h', 'surface_core.h', 'sample_stream.h', 'path_stage.h', 'shade_math.h', 'shade_tables.h',
           'shade_fresnel.h', 'shade_color.h', 'shade_lobes.h', 'shade_lobes_eval.h', 'shade_hair.h', 'shade_flakes.h', 'shade_glitter.h', 'shade_glitter_tables.h', 'shade_state.h', 'shade_builder.h',
           'shade_colorcorrect.h', 'shade_dwabase.h', 'shade_dwalayer.h', 'shade_maps.h', 'shade_noise.h', 'shade_bssrdf.h', 'shade_mapnodes.h', 'shade_mapnodes_more.h', 'curve_core.h', 'curve_embree.h', 'volume_core.h', 'shade_material.h',
           'dwa_admission.h', 'map_admission.h', 'aov_core.h', 'aov_material.h', 'aov_state.h', 'aov_crypto.h', 'camera_core.h', 'light_filter_core.h', 'volume_grid.h', 'volume_emission.h',
           'async_gpu_device.h', 'bc6h_core.h')
FAST_FLAGS = ['--std=c++17', '--gpu-architecture=compute_75', '--fmad=true', '--prec-div=true',
              '--prec-sqrt=true', '--ftz=false', '-DXPU2_PATHS_ONLY']


def build_all(source, runtime, out):
    source, runtime, out = (Path(p).resolve() for p in (source, runtime, out))
    out.mkdir(parents=True, exist_ok=False)
    verify(runtime, json.loads((HERE / 'reference-lock.json').read_text()))
    files = list(HERE.glob('*.h')) + [HERE / n for n in ('async_probe.cc', 'async_gpu.cu', 'async_material.cu', 'async_maps.cu', 'async_lights.cu', 'async_lobes.cu', 'async_volumes.cu', 'async_dwa.cu', 'async_dwalobes.cu', 'async_extras.cu',
                                                         'async_adaptive.cu', 'build_async.py',
                                                         'gpu_build.py', 'native_build.py')]
    files = list(files) + [HERE / HEADERS[0]]  # the patterns, kept with MoonRay's map in native-port
    held = {str(p): sha(p) for p in files}
    report = {'status': 'incomplete', 'source_hashes': held}
    try:
        # One program for uniform and adaptive frames (async_gpu.cu), its materials in a module of their own
        # (async_material.cu): NVRTC compiles the two at once (each in its own folder, for its log).
        (out / 'adaptive').mkdir()
        (out / 'material').mkdir()
        (out / 'maps').mkdir()
        (out / 'lights').mkdir()
        (out / 'lobes').mkdir()
        (out / 'volumes').mkdir()
        (out / 'dwa').mkdir()
        (out / 'dwalobes').mkdir()
        (out / 'extras').mkdir()
        # The modules the path tracer's callables reach (the volume module calls the others as the path tracer does).
        calls = ['-DXPU2_MATERIAL_MODULE=1', '-DXPU2_LIGHT_MODULE=1', '-DXPU2_LOBE_MODULE=1', '-DXPU2_EXTRAS_MODULE=1']
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(9) as pool:
            main = pool.submit(compile_ptx, source, out, 'async_gpu.cu', HEADERS, FAST_FLAGS + calls + ['-DXPU2_VOLUME_MODULE=1'])
            # Relocatable: their callables are entry points no code in the module calls (NVRTC would drop them).
            material = pool.submit(compile_ptx, source, out / 'material', 'async_material.cu', HEADERS,
                                   FAST_FLAGS + ['--relocatable-device-code=true', '-DXPU2_MAP_MODULE=1', '-DXPU2_DWA_MODULE=1'])
            dwa = pool.submit(compile_ptx, source, out / 'dwa', 'async_dwa.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            dwaLobes = pool.submit(compile_ptx, source, out / 'dwalobes', 'async_dwalobes.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            extras = pool.submit(compile_ptx, source, out / 'extras', 'async_extras.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            maps = pool.submit(compile_ptx, source, out / 'maps', 'async_maps.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            lights = pool.submit(compile_ptx, source, out / 'lights', 'async_lights.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            lobes = pool.submit(compile_ptx, source, out / 'lobes', 'async_lobes.cu', HEADERS, FAST_FLAGS + ['--relocatable-device-code=true'])
            volumes = pool.submit(compile_ptx, source, out / 'volumes', 'async_volumes.cu', HEADERS,
                                  FAST_FLAGS + calls + ['--relocatable-device-code=true'])
            ptx, report['cuda_build'] = main.result()
            materialPtx, report['cuda_build_material'] = material.result()
            mapsPtx, report['cuda_build_maps'] = maps.result()
            lightsPtx, report['cuda_build_lights'] = lights.result()
            lobesPtx, report['cuda_build_lobes'] = lobes.result()
            volumesPtx, report['cuda_build_volumes'] = volumes.result()
            dwaPtx, report['cuda_build_dwa'] = dwa.result()
            dwaLobesPtx, report['cuda_build_dwalobes'] = dwaLobes.result()
            extrasPtx, report['cuda_build_extras'] = extras.result()
        # The adaptive pass chain's CUDA kernels, beside the adaptive program.
        chain, report['cuda_build_chain'] = compile_ptx(source, out / 'adaptive', 'async_adaptive.cu', HEADERS, FAST_FLAGS)
        # The probe shares the session's render outputs (aov_plan.h, aov_display.h): MoonRay's light path
        # expressions and display filter buffers, as build_session.py.
        include = [source / 'toolchain/xpu/optix-dev/include',
                   source / 'toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include',
                   source / 'upstream/openmoonray/moonray/moonray/lib/rendering/lpe',
                   source / 'upstream/openmoonray/moonray/moonray/lib/rendering/displayfilter',
                   source / 'build/native-renderer-avx/moonray/lib/rendering/displayfilter']
        # With meshoptimizer's simplifier (the coarse first pass, coarse_geometry.h), a unit of its own as the session's.
        kernels, kernel_flags = bc6h_kernels(source, out / 'bc6h')
        exe, report['native_build'] = build_units(source, [HERE / 'async_probe.cc', HERE / 'xpu2_meshopt.cc', HERE / 'xpu2_bc6h.cc'], out / 'native',
                                                  [runtime / 'libembree4.dll', runtime / 'librendering_lpe.dll', runtime / 'librendering_displayfilter.dll',
                                                   *kernels, '-lcfgmgr32', '-lwinmm'],
                                                  ['-I' + str(v) for v in include] + ['-I' + str(source / 'upstream/meshoptimizer/src')] + kernel_flags)
        # MoonRay's sampler tables (a random walk's sequences) beside the GPU program.
        sampler = source / 'upstream/openmoonray/moonray/moonray/lib/rendering/pbr/sampler'
        for name in ('bc_1D_integrator_points_4096_sequences_of_1024.bin', 'ppd_2D_integrator_points_4096_sequences_of_1024.bin'):
            shutil.copyfile(sampler / name, out / name)
        if any(sha(Path(k)) != v for k, v in held.items()):
            raise RuntimeError('Build source drift')
        report.update(status='passed', exe=str(exe), ptx=str(ptx), ptx_material=str(materialPtx), ptx_maps=str(mapsPtx), ptx_lights=str(lightsPtx), ptx_lobes=str(lobesPtx), ptx_volumes=str(volumesPtx), ptx_dwa=str(dwaPtx), ptx_dwalobes=str(dwaLobesPtx), ptx_extras=str(extrasPtx),
                      ptx_chain=str(chain),
                      binary_hashes={'exe': sha(exe), 'ptx': sha(ptx), 'ptx_material': sha(materialPtx), 'ptx_maps': sha(mapsPtx),
                                     'ptx_lights': sha(lightsPtx), 'ptx_lobes': sha(lobesPtx), 'ptx_volumes': sha(volumesPtx), 'ptx_dwa': sha(dwaPtx), 'ptx_dwalobes': sha(dwaLobesPtx), 'ptx_extras': sha(extrasPtx), 'ptx_chain': sha(chain)})
    except Exception as error:
        report.update(status='failed', error=str(error))
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'runtime', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args()
    r = build_all(a.source, a.runtime, a.output)
    print(json.dumps({k: v for k, v in r.items() if k in ('status', 'error', 'exe', 'ptx')}, indent=2), flush=True)
    raise SystemExit(r['status'] != 'passed')

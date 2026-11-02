"""Build XPU2's own runtime for the private MoonRay for Modo test kit: its preview session (xpu2_session.cc) as
moonray_xpu2.exe, the GPU programs and sampler tables of an asynchronous build, and the pinned reference runtime's
libraries and scene class proxies it loads, all in one folder (the kit's runtime keeps its own, different builds of
the same libraries).

build_session.py --source <MoonRayForModo development tree> --runtime <pinned reference> --build <async build>
                 --output <new folder>

The GPU programs are the build's own, used only while every source they are made from is as it was then: OptiX's
disk cache keeps a program by its text, so an unchanged program needs no new compile. Never installs or publishes
anything."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_async import HEADERS
from native_build import bc6h_kernels, build, build_units
from reference_runtime import sha, verify

HERE = Path(__file__).resolve().parent
PROGRAM_SOURCES = HEADERS + ('async_gpu.cu', 'async_material.cu', 'async_maps.cu', 'async_lights.cu', 'async_lobes.cu', 'async_volumes.cu', 'async_dwa.cu', 'async_dwalobes.cu', 'async_extras.cu', 'async_adaptive.cu')


def imports(objdump, path):
    out = subprocess.run([str(objdump), '-p', str(path)], capture_output=True, text=True, errors='replace',
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return re.findall(r'DLL Name:\s*(\S+)', out)


def closure(objdump, runtime, roots):
    """Every library of the runtime folder the roots load, directly or through each other."""
    present = {p.name.lower(): p for p in runtime.iterdir() if p.is_file()}
    seen, todo = {}, list(roots)
    while todo:
        path = todo.pop()
        for name in imports(objdump, path):
            hit = present.get(name.lower())
            if hit and hit.name.lower() not in seen:
                seen[hit.name.lower()] = hit
                todo.append(hit)
    return sorted(seen.values())


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]


def build_session(source, runtime, async_build, out, cpu_test=False):
    source, runtime, async_build, out = (Path(p).resolve() for p in (source, runtime, async_build, out))
    out.mkdir(parents=True, exist_ok=False)
    verify(runtime, json.loads((HERE / 'reference-lock.json').read_text()))
    built = json.loads((async_build / 'build.json').read_text(encoding='utf-8'))
    if built.get('status') != 'passed':
        raise RuntimeError('The asynchronous build did not pass')
    drift = [n for n in PROGRAM_SOURCES if built['source_hashes'].get(str(HERE / n)) != sha(HERE / n)]
    # A CPU test (XPU2_SESSION_MODE=0) never loads the GPU programs: their sources may have moved on.
    if drift and not cpu_test:
        raise RuntimeError('The GPU programs are made from sources changed since that build: ' + ', '.join(drift))
    files = list(HERE.glob('*.h')) + [HERE / n for n in ('xpu2_session.cc', 'xpu2_scene.cc', 'xpu2_cpu.cc', 'xpu2_meshopt.cc', 'xpu2_bc6h.cc', 'build_session.py', 'native_build.py',
                                                         'xpu2_geom.cc', 'build_geom.py')]
    files += sorted((HERE.parents[1] / 'native-port').glob('modo_*.h'))
    held = {str(p): sha(p) for p in files}
    report = {'status': 'incomplete', 'source_hashes': held, 'async_build': str(async_build),
              'cpu_test_only': bool(cpu_test and drift), 'program_drift': drift}
    try:
        # MoonRay's light path expression state machine (librendering_lpe) for the light AOVs (aov_plan.h).
        # MoonRay's display filter input buffers (librendering_displayfilter) for the display filters (aov_display.h).
        include = [source / 'toolchain/xpu/optix-dev/include',
                   source / 'toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include',
                   source / 'upstream/openmoonray/moonray/moonray/lib/rendering/lpe',
                   source / 'upstream/openmoonray/moonray/moonray/lib/rendering/displayfilter',
                   source / 'build/native-renderer-avx/moonray/lib/rendering/displayfilter']
        # MoonRay's own subdivision surface tessellation (xpu2_geom.cc), built with MoonRay's flags on its own thread
        # beside the session's units, or taken from the cache when its sources and build script are unchanged.
        import threading
        from build_geom import build_geom
        key = sha_text('|'.join(sha(HERE / n) for n in ('xpu2_geom.cc', 'xpu2_geom.h', 'build_geom.py')) + '|' + str(source))
        cached = out.parent / 'geom-cache' / key
        geom_result = {}

        def make_geom():
            if (cached / 'xpu2_geom.dll').is_file():
                shutil.copytree(cached, out / 'geom')
                geom_result.update(json.loads((out / 'geom' / 'build.json').read_text(encoding='utf-8')), cached=str(cached))
                return
            geom_result.update(build_geom(source, out / 'geom'))
            if geom_result.get('status') == 'passed':
                shutil.copytree(out / 'geom', cached, dirs_exist_ok=True)
        geom_thread = threading.Thread(target=make_geom)
        geom_thread.start()
        try:
            kernels, kernel_flags = bc6h_kernels(source, out / 'bc6h')
            exe, report['native_build'] = build_units(source, [HERE / 'xpu2_session.cc', HERE / 'xpu2_scene.cc', HERE / 'xpu2_cpu.cc', HERE / 'xpu2_meshopt.cc', HERE / 'xpu2_bc6h.cc'], out / 'native',
                                                      [runtime / 'libembree4.dll', runtime / 'librendering_lpe.dll',
                                                       runtime / 'librendering_displayfilter.dll', *kernels, '-lcfgmgr32', '-lwinmm'],
                                                      ['-I' + str(v) for v in include] + ['-I' + str(source / 'upstream/meshoptimizer/src'), '-DXPU2_SPLIT_UNITS'] + kernel_flags)
        finally:
            geom_thread.join()
        geom = geom_result
        report['geom_build'] = geom
        if geom['status'] != 'passed':
            raise RuntimeError('xpu2_geom.dll failed to build: ' + geom.get('error', '')[-500:])
        rt = out / 'runtime-xpu2'
        (rt / 'adaptive').mkdir(parents=True)
        (rt / 'material').mkdir(parents=True)
        (rt / 'maps').mkdir(parents=True)
        (rt / 'lights').mkdir(parents=True)
        (rt / 'lobes').mkdir(parents=True)
        (rt / 'volumes').mkdir(parents=True)
        (rt / 'dwa').mkdir(parents=True)
        (rt / 'dwalobes').mkdir(parents=True)
        (rt / 'extras').mkdir(parents=True)
        shutil.copyfile(exe, rt / 'moonray_xpu2.exe')
        shutil.copyfile(out / 'geom' / 'xpu2_geom.dll', rt / 'xpu2_geom.dll')
        for name in ('async_gpu.ptx', 'material/async_material.ptx', 'maps/async_maps.ptx', 'lights/async_lights.ptx', 'lobes/async_lobes.ptx', 'volumes/async_volumes.ptx', 'dwa/async_dwa.ptx', 'dwalobes/async_dwalobes.ptx', 'extras/async_extras.ptx', 'adaptive/async_adaptive.ptx',
                     'bc_1D_integrator_points_4096_sequences_of_1024.bin',
                     'ppd_2D_integrator_points_4096_sequences_of_1024.bin'):
            shutil.copyfile(async_build / name, rt / name)
        objdump = source / 'toolchain/msys64/ucrt64/bin/objdump.exe'
        # The scene classes are read through their proxies (proxy mode): every one, and what they load.
        proxies = sorted(runtime.glob('*.so.proxy'))
        for p in proxies:
            shutil.copyfile(p, rt / p.name)
        # The display filters run as MoonRay's own, from their DSOs (aov_display.h).
        filters = sorted(runtime.glob('*DisplayFilter.so'))
        for p in filters:
            shutil.copyfile(p, rt / p.name)
        # But for Toon's, as the kit's MoonRay has it: the reference is older than its Output Mode and whole lines, which
        # the kit's scenes ask of it. As the reference's it gives every Output Mode the shading with its lines.
        for name in ('ToonDisplayFilter.so', 'ToonDisplayFilter.so.proxy'):
            shutil.copyfile(source / 'build/native-renderer-avx/bin' / name, rt / name)
        libraries = closure(objdump, runtime, [exe] + proxies + filters + [out / 'geom' / 'xpu2_geom.dll'])
        for p in libraries:
            shutil.copyfile(p, rt / p.name)
        for name in ('MOONRAY_LICENSE.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'RANDOM123_LICENSE.txt', 'ISPC_MATH_LICENSE.txt'):
            if (HERE / name).is_file():
                shutil.copyfile(HERE / name, rt / name)
        if any(sha(Path(k)) != v for k, v in held.items()):
            raise RuntimeError('Build source drift')
        manifest = {'executable_sha256': sha(rt / 'moonray_xpu2.exe'),
                    'programs': {n: sha(rt / n) for n in ('async_gpu.ptx', 'material/async_material.ptx', 'maps/async_maps.ptx',
                                                          'lights/async_lights.ptx', 'lobes/async_lobes.ptx', 'volumes/async_volumes.ptx',
                                                          'dwa/async_dwa.ptx', 'dwalobes/async_dwalobes.ptx', 'extras/async_extras.ptx',
                                                          'adaptive/async_adaptive.ptx')},
                    'libraries': [p.name for p in libraries], 'proxies': len(proxies)}
        (rt / 'xpu2-runtime.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        report.update(status='passed', runtime=str(rt), **manifest)
    except Exception as error:
        report.update(status='failed', error=str(error))
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'runtime', 'build', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--cpu-test', action='store_true', help='allow GPU program sources changed since the build (CPU-only tests)')
    a = p.parse_args()
    r = build_session(a.source, a.runtime, a.build, a.output, a.cpu_test)
    print(json.dumps({k: v for k, v in r.items() if k in ('status', 'error', 'runtime', 'proxies')}, indent=2), flush=True)
    raise SystemExit(r['status'] != 'passed')

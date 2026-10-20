"""Apply an xpu2 renderer patch module on top of an existing XPU2 candidate build.

The candidate (for example test-results/xpu2-disabled-route-build) keeps its
link list, its patched PathIntegratorUtil.cc with generated headers, and its ISPC
objects. Files the patch module names are compiled here: PathIntegratorUtil.cc
from the candidate's patched copy, anything else from the current upstream tree
(only for files unchanged since the candidate's objects were built). The new
objects replace their counterparts in the candidate's link list, and the
renderer DLL is relinked into a copy of the candidate's runtime.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from xpu2.reference import digest_file, inventory, verify
from xpu2_baseline import environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='MoonRayForModo checkout (build/, upstream/, toolchain/)')
    p.add_argument('--base', type=Path, required=True, help='XPU2 candidate build folder')
    p.add_argument('--variant', required=True, help='xpu2 patch module')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, base, out = a.source.resolve(), a.base.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = source / 'build/native-renderer-avx'
    lib = source / 'upstream/openmoonray/moonray/moonray/lib/rendering'
    toolbin = source / 'toolchain/msys64/ucrt64/bin'
    report = dict(status='incomplete', base=str(base), variant=a.variant)
    try:
        b = json.loads((base / 'report.json').read_text())
        if b['status'] != 'built' or digest_file(base / 'runtime/librendering_rndr.dll') != b['dll_sha256']:
            raise ValueError('Candidate build drift')
        held = inventory(base / 'runtime')
        runtime = out / 'runtime'
        # Hard links keep the copy cheap; the DLL is replaced, never written through.
        for f in (base / 'runtime').rglob('*'):
            q = runtime / f.relative_to(base / 'runtime')
            if f.is_dir():
                q.mkdir(parents=True, exist_ok=True)
            else:
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(f, q)
        for f in list(base.glob('*.h')) + list(base.glob('*.inc')):
            shutil.copyfile(f, out / f.name)
        module = importlib.import_module('xpu2.' + a.variant)
        candidate_copies = {'pbr/integrator/PathIntegratorUtil.cc': base / 'PathIntegratorUtil.cc'}
        # Other files the candidate links from its own patched copies (an object outside the
        # build tree with its source beside it), such as ShadeBundleHandler.cc.
        base_rsp = (base / 'link.rsp').read_text().splitlines()
        for f in module.FILES:
            stem = Path(f).stem
            own = [Path(l.strip().strip('"')) for l in base_rsp if l.strip().strip('"').endswith('/' + stem + '.obj')]
            if f not in candidate_copies and len(own) == 1 and own[0].with_suffix('.cc').exists():
                candidate_copies[f] = own[0].with_suffix('.cc')
        texts, hashes = {}, {}
        for f in module.FILES:
            path = candidate_copies.get(f, lib / f)
            texts[f] = path.read_text(encoding='utf-8'); hashes[str(path)] = digest_file(path)
        for h in module.HEADERS:
            shutil.copyfile(ROOT / 'xpu2' / h, out / h); hashes[str(ROOT / 'xpu2' / h)] = digest_file(ROOT / 'xpu2' / h)
        hashes[str(ROOT / 'xpu2' / (a.variant + '.py'))] = digest_file(ROOT / 'xpu2' / (a.variant + '.py'))
        # HEADER_FILES: patched copies of MoonRay headers {lib-relative source: include path}, shadowing originals.
        overrides = getattr(module, 'HEADER_FILES', {})
        for f in overrides:
            texts[f] = (lib / f).read_text(encoding='utf-8'); hashes[str(lib / f)] = digest_file(lib / f)
        patched = module.patch(texts)
        for f, inc in overrides.items():
            (out / inc).parent.mkdir(parents=True, exist_ok=True)
            (out / inc).write_text(patched[f], encoding='utf-8')
        rsp = (base / 'link.rsp').read_text()
        lines = rsp.splitlines()
        env = environment(runtime, out); env['PATH'] = str(toolbin) + os.pathsep + env['PATH']
        replaced = {}
        for f in module.FILES:
            name = Path(f).name
            copy = out / name
            copy.write_text(patched[f], encoding='utf-8')
            target = 'moonray/lib/rendering/pbr/CMakeFiles/rendering_pbr_objects.dir/' + f[len('pbr/'):] + '.obj'
            command = subprocess.run([str(toolbin / 'ninja.exe'), '-t', 'commands', target], cwd=build,
                                     capture_output=True, text=True, check=True).stdout.splitlines()[-1]
            flags = shlex.split(command)
            flags = flags[:flags.index('-MD')]
            flags = [flags[0], '-I' + str(out)] + flags[1:] + [
                '-iquote', str((lib / f).parent), '-I' + str(ROOT / 'xpu2'),
                '-I' + str(source / 'toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include')]
            obj = out / (name + '.obj')
            with (out / ('compile-' + name + '.log')).open('w') as log:
                r = subprocess.run([*flags, '-c', str(copy), '-o', str(obj)], cwd=out, env=env, stdout=log, stderr=log,
                                   timeout=900, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode:
                raise ValueError('Compile failed: ' + f)
            # The candidate's own object for this file.
            if f in candidate_copies:
                old = [i for i, l in enumerate(lines) if l.strip().strip('"').endswith('/' + Path(f).stem + '.obj')]
            else:
                old = [i for i, l in enumerate(lines) if l.strip().strip('"').endswith('-' + name + '.obj')
                       or l.strip().strip('"').endswith('/' + name + '.obj')]
            if len(old) != 1:
                raise ValueError('No unique link input for ' + f + ': ' + str(old))
            replaced[lines[old[0]]] = str(obj)
            lines[old[0]] = '"' + str(obj).replace('\\', '/') + '"'
        # NEW_SOURCES: extra translation units from xpu2/, compiled with the Subsurface flags and linked in.
        for src in getattr(module, 'NEW_SOURCES', ()):
            path = ROOT / 'xpu2' / src
            hashes[str(path)] = digest_file(path)
            target = 'moonray/lib/rendering/pbr/CMakeFiles/rendering_pbr_objects.dir/integrator/PathIntegratorSubsurface.cc.obj'
            command = subprocess.run([str(toolbin / 'ninja.exe'), '-t', 'commands', target], cwd=build,
                                     capture_output=True, text=True, check=True).stdout.splitlines()[-1]
            flags = shlex.split(command)
            flags = flags[:flags.index('-MD')]
            flags = [flags[0], '-I' + str(out)] + flags[1:] + ['-I' + str(ROOT / 'xpu2'),
                '-I' + str(source / 'toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include')]
            obj = out / (Path(src).name + '.obj')
            with (out / ('compile-' + Path(src).name + '.log')).open('w') as log:
                r = subprocess.run([*flags, '-c', str(path), '-o', str(obj)], cwd=out, env=env, stdout=log, stderr=log,
                                   timeout=900, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode:
                raise ValueError('Compile failed: ' + src)
            lines.insert(0, '"' + str(obj).replace('\\', '/') + '"')
        (out / 'link.rsp').write_text('\n'.join(lines) + '\n')
        dll = runtime / 'librendering_rndr.dll'
        dll.unlink()
        with (out / 'link.log').open('w') as log:
            r = subprocess.run([str(toolbin / 'g++.exe'), '-O3', '-shared', '-o', str(dll), '@' + str(out / 'link.rsp')],
                               cwd=out, env=env, stdout=log, stderr=log, timeout=900,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode:
            raise ValueError('Link failed')
        verify(base / 'runtime', held)
        for f, h in hashes.items():
            if digest_file(Path(f)) != h:
                raise ValueError('Input changed during build ' + f)
        report.update(status='built', input_hashes=hashes, replaced=replaced, dll_sha256=digest_file(dll),
                      base_report_sha256=digest_file(base / 'report.json'), experimental_only=True)
    except Exception as e:
        import traceback
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], report.get('error', ''), flush=True)
    return 0 if report['status'] == 'built' else 1


if __name__ == '__main__':
    raise SystemExit(main())

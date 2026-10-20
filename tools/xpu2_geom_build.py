"""Build an isolated librendering_geom.dll or librendering_rndr.dll from the native build tree.

Only the files a patch module names are recompiled, from patched copies in the
output folder; every other object comes unchanged from the build tree, which is
never written. `--variant baseline` relinks the unmodified objects, so a
baseline and a variant share the same toolchain and inputs.
"""
import argparse
import importlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from xpu2.reference import digest_file

DLLS = {'geom': 'bin/librendering_geom.dll', 'rndr': 'bin/librendering_rndr.dll'}


def ninja_tokens(text):
    """Split a ninja path list; '$ ' is an escaped space, '$:' an escaped colon."""
    out, cur, i = [], '', 0
    while i < len(text):
        c = text[i]
        if c == '$' and i + 1 < len(text):
            cur += text[i + 1]; i += 2; continue
        if c == ' ':
            if cur:
                out.append(cur); cur = ''
        else:
            cur += c
        i += 1
    if cur:
        out.append(cur)
    return out


def stanza(build_ninja, target):
    text = build_ninja.read_text()
    head = 'build ' + target.replace(':', '$:') + ' '
    start = text.index('\n' + head) + 1
    end = text.index('\n\n', start)
    lines = text[start:end].split('\n')
    first = lines[0][len('build '):]
    _, rest = first.split(': ', 1)
    rule, rest = rest.split(' ', 1)
    explicit = ninja_tokens(rest.split(' | ', 1)[0].split(' || ', 1)[0])
    variables = dict(l.strip().split(' = ', 1) for l in lines[1:] if ' = ' in l)
    return rule, explicit, variables


def object_for(objects, source):
    """The link input compiled from a source path relative to lib/rendering."""
    library, rest = source.split('/', 1)
    found = [o for o in objects if o.endswith('/' + rest + '.obj') and '/' + library + '/' in o]
    if len(found) != 1:
        raise ValueError('No unique object for ' + source + ': ' + str(found))
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='MoonRayForModo checkout holding build/ and upstream/')
    p.add_argument('--variant', required=True, help="xpu2 patch module name, or 'baseline'")
    p.add_argument('--dll', choices=sorted(DLLS), default='geom')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = source / 'build/native-renderer-avx'
    lib = source / 'upstream/openmoonray/moonray/moonray/lib/rendering'
    toolbin = source / 'toolchain/msys64/ucrt64/bin'
    env = {'PATH': str(toolbin), 'SYSTEMROOT': r'C:\Windows', 'TEMP': str(out), 'TMP': str(out)}
    report = dict(status='incomplete', variant=a.variant, dll=a.dll, source=str(source))
    try:
        rule, explicit, variables = stanza(build / 'build.ninja', DLLS[a.dll])
        objects = [o for o in explicit if o.endswith(('.obj', '.o'))]
        if len(objects) < 40:
            raise ValueError('Unexpected geom object list')
        hashes = {str(build / o): digest_file(build / o) for o in objects}
        replaced = {}
        if a.variant != 'baseline':
            module = importlib.import_module('xpu2.' + a.variant)
            # HEADER_FILES: patched copies of MoonRay headers, {lib-relative source: include path}.
            overrides = getattr(module, 'HEADER_FILES', {})
            texts = {f: (lib / f).read_text(encoding='utf-8') for f in (*module.FILES, *overrides)}
            for f in (*module.FILES, *overrides):
                hashes[str(lib / f)] = digest_file(lib / f)
            patched = module.patch(texts)
            include = out / 'include'; include.mkdir()
            for h in module.HEADERS:
                shutil.copyfile(ROOT / 'xpu2' / h, include / h); hashes[str(ROOT / 'xpu2' / h)] = digest_file(ROOT / 'xpu2' / h)
            for f, path in overrides.items():
                (include / path).parent.mkdir(parents=True, exist_ok=True)
                (include / path).write_text(patched[f], encoding='utf-8')
            hashes[str(ROOT / 'xpu2' / (a.variant + '.py'))] = digest_file(ROOT / 'xpu2' / (a.variant + '.py'))
            for f in module.FILES:
                copy = out / 'src' / f; copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_text(patched[f], encoding='utf-8')
                target = object_for(objects, f)
                command = subprocess.run([str(toolbin / 'ninja.exe'), '-t', 'commands', target], cwd=build, env=env,
                                         capture_output=True, text=True, check=True).stdout.splitlines()[-1]
                flags = shlex.split(command)
                flags = flags[:flags.index('-MD')] if '-MD' in flags else flags[:flags.index('-o')]
                obj = out / 'obj' / (f.replace('/', '_') + '.obj'); obj.parent.mkdir(exist_ok=True)
                # Patched headers first, so they shadow the originals.
                args = [flags[0], '-I' + str(include), *flags[1:], '-iquote', str((lib / f).parent),
                        '-c', str(copy), '-o', str(obj)]
                with (out / ('compile-' + f.replace('/', '_') + '.log')).open('w') as log:
                    r = subprocess.run(args, cwd=build, env=env, stdout=log, stderr=log, timeout=900)
                if r.returncode:
                    raise ValueError('Compile failed: ' + f)
                replaced[target] = str(obj)
            missing = set(replaced) - set(objects)
            if missing:
                raise ValueError('Patched objects not in link: ' + str(sorted(missing)))
        link_inputs = [replaced.get(o, o) for o in objects]
        libraries = shlex.split(variables.get('LINK_LIBRARIES', ''))
        rsp = out / 'link.rsp'
        rsp.write_text('\n'.join('"%s"' % x.replace('\\', '/') for x in [*link_inputs, *libraries]) + '\n')
        dll = out / Path(DLLS[a.dll]).name
        args = [str(toolbin / 'g++.exe'), *variables['LANGUAGE_COMPILE_FLAGS'].split(), *variables['LINK_FLAGS'].split(),
                '-o', str(dll), '-Wl,--out-implib,' + str(out / (Path(DLLS[a.dll]).name + '.a')),
                '-Wl,--major-image-version,0,--minor-image-version,0', '@' + str(rsp)]
        with (out / 'link.log').open('w') as log:
            r = subprocess.run(args, cwd=build, env=env, stdout=log, stderr=log, timeout=900)
        if r.returncode:
            raise ValueError('Link failed')
        for f, h in hashes.items():
            if digest_file(Path(f)) != h:
                raise ValueError('Input changed during build: ' + f)
        report.update(status='built', rule=rule, input_hashes=hashes, replaced=replaced,
                      dll_sha256=digest_file(dll), experimental_only=True)
    except Exception as e:
        import traceback
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], report.get('error', ''), flush=True)
    return 0 if report['status'] == 'built' else 1


if __name__ == '__main__':
    raise SystemExit(main())

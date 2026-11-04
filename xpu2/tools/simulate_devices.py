"""The multi-device simulation app: XPU2's multi-device extension (interactive/multi_device.h) tested on a machine with
one graphics card and one processor group, where it otherwise never runs.

simulate_devices.py --exe <moonray_xpu2.exe> --scene <scene.rdla> [--scene <another>] --output <new folder>
                    [--suite quick|full] [--cards 3] [--groups 4] [--only <case>,...] [--timeout s] [--env KEY=VALUE]

Each scene is rendered headless (interactive/session_harness.py, as the kit drives a preview) on simulated machines:
more cards (XPU2_SIMULATE_CARDS: lanes on the one card, each with its own context, scene and film), more processor
groups (XPU2_SIMULATE_GROUPS), a smaller card beside a larger (XPU2_SIMULATE_CARD_THREADS), a card that fails as it
takes the scene (XPU2_SIMULATE_CARD_FAIL), and realtime's denoiser handed from card to card and to the processor
(XPU2_SIMULATE_DENOISE_FAIL). Two one-card renders measure the scene's own noise, the second with the frame's other
samples (XPU2_SIMULATE_SEED); every other render must come out as the first does, within that noise, without falling
back, and with the Terminal lines its machine should give.

What it shows: that the lanes claim, launch, merge and audit correctly, and that a device that cannot hands its work on.
What it cannot: speed. Simulated cards share one card's time (three lanes on one card run slower than one), and
simulated groups one group's processors.

It writes <output>/report.html and report.json, and each render's folder (its pictures, protocol.log, report.json).
Exit status: 0 when every case passes, 1 otherwise. Opens no Modo; stops if one is open."""
import argparse
import html
import json
import math
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / 'interactive' / 'session_harness.py'
# The kit's switches for its previews, as its own tests set them.
KIT_ENV = {'MOONRAY_MODO_OPAQUE_ALPHA': '1', 'MOONRAY_MODO_CAMERA_LIGHT_LPE': '1'}


class Case:
    """One simulated machine: its switches, what its Terminal must say and must not, and what it is compared with."""

    def __init__(self, name, what, env=None, expect=(), forbid=(r'left out',), realtime=False, against='base-a',
                 tolerance=None, needs=None):
        self.name, self.what, self.env, self.expect, self.forbid = name, what, dict(env or {}), list(expect), list(forbid)
        # needs: (pattern, why) the Terminal must show for the case to say anything, else it is skipped: a scene that
        # decodes no texture tiles has none to move.
        self.realtime, self.against, self.tolerance, self.needs = realtime, against, tolerance, needs


def cases(suite, cards, groups):
    """The cases of a suite. quick: the main paths; full: every count up to cards and groups, and every hand-off."""
    out = [Case('base-a', 'one card, one processor group', forbid=(r'left out', r'renders with the first', r'processor groups')),
           # The scene's noise: the frame's other samples (XPU2_SIMULATE_SEED). Two renders with the one seed are all but
           # the same picture, a sample being its pixel, index and seed, so they would measure nothing.
           Case('base-b', 'one card, another seed: the scene\'s own noise', {'XPU2_SIMULATE_SEED': 1},
                forbid=(r'left out', r'renders with the first'), against=None)]
    card_counts = range(2, cards + 1) if suite == 'full' else [2]
    group_counts = range(2, groups + 1) if suite == 'full' else [2]
    for n in card_counts:
        out.append(Case('cards-%d' % n, '%d cards' % n, {'XPU2_SIMULATE_CARDS': n},
                        expect=[r'\(lane %d\) renders with the first' % k for k in range(1, n)]))
    for k in group_counts:
        out.append(Case('groups-%d' % k, '%d processor groups' % k, {'XPU2_SIMULATE_GROUPS': k},
                        expect=[r'%d processor groups' % k]))
    out.append(Case('cards-2-groups-2', '2 cards and 2 processor groups', {'XPU2_SIMULATE_CARDS': 2, 'XPU2_SIMULATE_GROUPS': 2},
                    expect=[r'\(lane 1\) renders with the first', r'2 processor groups']))
    out.append(Case('uneven-cards', 'a large card and one an eighth as wide', {'XPU2_SIMULATE_CARDS': 2, 'XPU2_SIMULATE_CARD_THREADS': '65536,8192'},
                    expect=[r'\(lane 1\) renders with the first']))
    out.append(Case('card-fails', 'three cards, the third failing as it takes the scene',
                    {'XPU2_SIMULATE_CARDS': 3, 'XPU2_SIMULATE_CARD_FAIL': 2},
                    expect=[r'left out: failed \(simulated, XPU2_SIMULATE_CARD_FAIL\)'], forbid=()))
    # Device Management's switches: cards and processor groups left out of the render.
    out.append(Case('cards-chosen', 'three cards, the second switched off', {'XPU2_SIMULATE_CARDS': 3, 'XPU2_CARD_LIST': '0,2'},
                    expect=[r'\(lane 1\) renders with the first'], forbid=(r'left out', r'\(lane 2\)')))
    out.append(Case('groups-chosen', 'three processor groups, only the second switched on', {'XPU2_SIMULATE_GROUPS': 3, 'XPU2_GROUP_LIST': '1'},
                    expect=[r'1 processor groups']))
    # NUMA: workers a share on each node, each card's lane on its own node, and a node short of memory moving its texture
    # tiles to the other (multi_device.h, host_texture.h).
    out.append(Case('numa-2', 'two NUMA nodes', {'XPU2_SIMULATE_NODES': 2}, expect=[r'2 NUMA nodes']))
    out.append(Case('numa-cards', 'two cards, each near a node of its own', {'XPU2_SIMULATE_CARDS': 2, 'XPU2_SIMULATE_NODES': 2,
                    'XPU2_SIMULATE_CARD_NODES': '0,1'}, expect=[r'\(lane 1\) near node 1', r'first graphics card near node 0']))
    out.append(Case('numa-spill', 'two NUMA nodes, the first holding 1 MB of texture tiles: the rest moved to the second',
                    {'XPU2_SIMULATE_NODES': 2, 'XPU2_SIMULATE_NODE_MB': '1,0'}, expect=[r'moved to node 1'],
                    needs=(r'[1-9]\d* of \d+ finest tiles decoded', 'the scene decoded no fine texture tiles')))
    # Realtime's denoiser: OptiX on the first card that runs it, else Open Image Denoise on the processor.
    dn = [Case('denoise-optix', 'realtime, denoised by OptiX on the first card', realtime=True, against=None,
               expect=[r'denoised by OptiX on lane 0']),
          Case('denoise-cpu', 'realtime, the only card\'s OptiX denoiser failing: the processor', {'XPU2_SIMULATE_DENOISE_FAIL': 1},
               realtime=True, against='denoise-optix', tolerance=0.1,
               expect=[r'OptiX\'s denoiser on lane 0: failed', r'pictures denoised by Open Image Denoise on the processor'])]
    if suite == 'full' or cards >= 2:
        dn += [Case('denoise-next-card', 'realtime on 2 cards, the first card\'s OptiX denoiser failing: the second\'s',
                    {'XPU2_SIMULATE_CARDS': 2, 'XPU2_SIMULATE_DENOISE_FAIL': 1}, realtime=True, against='denoise-optix', tolerance=0.1,
                    expect=[r'OptiX\'s denoiser on lane 0: failed', r'denoised by OptiX on lane 1'])]
    if suite == 'full':
        dn += [Case('denoise-all-fail', 'realtime on 2 cards, both OptiX denoisers failing: the processor',
                    {'XPU2_SIMULATE_CARDS': 2, 'XPU2_SIMULATE_DENOISE_FAIL': 2}, realtime=True, against='denoise-optix', tolerance=0.1,
                    expect=[r'OptiX\'s denoiser on lane 1: failed', r'pictures denoised by Open Image Denoise on the processor'])]
    return out + dn


def realtime_scene(folder, samples=16):
    """A scene file read after the scene's own: realtime with the kit's denoiser guides, as the kit writes them for a
    preview with RT on and Denoised Beauty chosen (rdla.py)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'realtime.rdla'
    # Form 2, on, samples, by time, no noise target, first generation, most samples, CPU workers, realtime.
    values = [2, 1, samples, 0, 0, 0, 0, 1, 1]
    lines = ['UserData("/modo/xpu2/generations") {', '  ["float_key"] = "xpu2_generations",',
             '  ["float_values_0"] = {%s},' % ', '.join(str(v) for v in values), '}']
    for key, attributes in (('albedo', {'result': 7, 'material_aov': '"albedo"'}), ('normal', {'result': 3, 'state_variable': 2})):
        lines += ['RenderOutput("/modo/preview/denoise_%s") {' % key,
                  '  ["file_name"] = "%s",' % (folder / ('denoise_%s.exr' % key)).as_posix(),
                  '  ["channel_format"] = 0,', '  ["compression"] = 1,']
        lines += ['  ["%s"] = %s,' % (k, v) for k, v in attributes.items()]
        lines.append('}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def pfm(path):
    data = path.read_bytes()
    parts, at = [], 0
    while len(parts) < 3:
        end = data.index(b'\n', at)
        parts.append(data[at:end].decode().strip())
        at = end + 1
    w, h = map(int, parts[1].split())
    scale = float(parts[2])
    n = w * h * (3 if parts[0] == 'PF' else 1)
    return w, h, struct.unpack(('<' if scale < 0 else '>') + 'f' * n, data[at:at + 4 * n])


def last_picture(folder):
    frames = [p for p in folder.glob('frame-*.pfm') if re.fullmatch(r'frame-\d+\.pfm', p.name)]
    return max(frames, key=lambda p: p.stat().st_mtime) if frames else None


def compare(a, b):
    """Mean ratio of b to a and the RMS of their difference, over finite values; None where they cannot be compared."""
    if not a or not b or a[:2] != b[:2]:
        return None
    pairs = [(x, y) for x, y in zip(a[2], b[2]) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return None
    ma = sum(x for x, _ in pairs) / len(pairs)
    mb = sum(y for _, y in pairs) / len(pairs)
    rms = math.sqrt(sum((x - y) ** 2 for x, y in pairs) / len(pairs))
    return {'mean': ma, 'ratio': mb / ma if ma else float('nan'), 'rms': rms, 'nonfinite': len(a[2]) - len(pairs)}


def run_case(case, exe, scene, extra_scene, out, timeout, extra_env):
    folder = out / case.name
    args = [sys.executable, str(HARNESS), '--exe', str(exe), '--output', str(folder), '--timeout', str(timeout)]
    for s in scene + ([extra_scene] if extra_scene else []):
        args += ['--scene', str(s)]
    if case.realtime:
        args += ['--view', 'denoised_beauty']
    env = dict(KIT_ENV)
    env.update(extra_env)
    env.update({k: str(v) for k, v in case.env.items()})
    for k, v in env.items():
        args += ['--env', '%s=%s' % (k, v)]
    began = time.monotonic()
    done = subprocess.run(args, capture_output=True, text=True, errors='replace')
    seconds = time.monotonic() - began
    report = {}
    if (folder / 'report.json').is_file():
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    log = (folder / 'protocol.log').read_text(encoding='utf-8', errors='replace') if (folder / 'protocol.log').is_file() else ''
    finished = re.findall(r'finished in ([\d.]+) s \(CPU (\d+) samples, GPU (\d+)\)', log)
    picture = last_picture(folder)
    return {'case': case, 'folder': folder, 'report': report, 'log': log, 'harness_exit': done.returncode,
            'harness_tail': (done.stdout + done.stderr)[-600:], 'seconds': seconds,
            'render': finished[-1] if finished else None, 'picture': pfm(picture) if picture else None}


def judge(result, results):
    """Pass or fail, and why."""
    case, report, log = result['case'], result['report'], result['log']
    why = []
    if result['harness_exit'] != 0 or report.get('exit_code') not in (0, None):
        why.append('the session or its harness exited with an error')
    if report.get('stuck'):
        why.append('the session stopped answering')
    if report.get('fallback'):
        why.append('fell back: ' + report['fallback'])
    if not any(e[1] == 'DONE' for e in report.get('events', [])):
        why.append('no frame finished')
    for pattern in case.expect:
        if not re.search(pattern, log):
            why.append('the Terminal lacks: ' + pattern)
    for pattern in case.forbid:
        if re.search(pattern, log):
            why.append('the Terminal says: ' + pattern)
    result['comparison'] = None
    if case.against:
        ref = results.get(case.against)
        cmp = compare(ref['picture'] if ref else None, result['picture'])
        result['comparison'] = cmp
        if cmp is None:
            why.append('no picture to compare with %s' % case.against)
        else:
            if cmp['nonfinite']:
                why.append('%d values not finite' % cmp['nonfinite'])
            if case.tolerance is not None:  # a different denoiser: the same picture, broadly
                if abs(cmp['ratio'] - 1) > case.tolerance:
                    why.append('mean %.4f of %s\'s' % (cmp['ratio'], case.against))
            else:
                noise = results.get('base-b', {}).get('comparison') or {}
                floor = noise.get('rms', 0.0)
                rel = abs(noise.get('ratio', 1.0) - 1)
                # Within the scene's own noise: twice the RMS of two one-card renders, and a mean ratio within three
                # times theirs (at least 0.5%).
                if cmp['rms'] > 2.0 * floor + 1e-6:
                    why.append('RMS %.5f past twice the noise (%.5f)' % (cmp['rms'], floor))
                if abs(cmp['ratio'] - 1) > max(0.005, 3 * rel):
                    why.append('mean ratio %.4f' % cmp['ratio'])
    result['skipped'] = None
    if case.needs and not re.search(case.needs[0], log):
        result['skipped'], why = case.needs[1], []
    result['passed'] = not why
    result['why'] = why
    return result


def page(scenes_results, out, exe):
    rows = []
    for scene, results in scenes_results:
        rows.append('<tr class="scene"><th colspan="7">%s</th></tr>' % html.escape(str(scene)))
        for r in results:
            c = r.get('comparison') or {}
            render = r['render']
            rows.append('<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
                'pass' if r['passed'] else 'fail', html.escape(r['case'].name), html.escape(r['case'].what),
                'pass' if r['passed'] else 'FAIL',
                '%s s' % render[0] if render else '%.0f s' % r['seconds'],
                '%s / %s' % (render[1], render[2]) if render else '',
                'mean %.4f, RMS %.5f' % (c['ratio'], c['rms']) if c else '',
                html.escape('; '.join(r['why'])) or html.escape(', '.join('%s=%s' % kv for kv in r['case'].env.items()))))
    passed = all(r['passed'] for _, rs in scenes_results for r in rs)
    text = '''<!doctype html><html><head><meta charset="utf-8"><title>XPU2 multi-device simulation</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--pass:#1a7f37;--fail:#c62828;--band:#f5f5f7}
@media (prefers-color-scheme: dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--line:#3a3a3c;--pass:#4cc366;--fail:#ff6b6b;--band:#2c2c2e}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px 16px;max-width:1200px}
table{border-collapse:collapse;width:100%%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr.scene th{background:var(--band);font-weight:600}td:nth-child(3){font-weight:600}
tr.pass td:nth-child(3){color:var(--pass)}tr.fail td:nth-child(3){color:var(--fail)}p{color:var(--muted)}
</style></head><body>
<h1>XPU2 multi-device simulation: %s</h1>
<p>%s. Session: %s. Simulated cards share one card's time and simulated groups one group's processors, so the times say
nothing about real machines; the pictures and the Terminal lines say whether the lanes claim, merge and hand work on
correctly. A render passes within twice the RMS of two one-card renders with different seeds (base-a, base-b).</p>
<table><tr><th>Case</th><th>Machine</th><th>Verdict</th><th>Render</th><th>Samples CPU / GPU</th><th>Against base-a</th><th>Notes</th></tr>
%s</table></body></html>
''' % ('all pass' if passed else 'failures', time.strftime('%Y-%m-%d %H:%M'), html.escape(str(exe)), '\n'.join(rows))
    (out / 'report.html').write_text(text, encoding='utf-8')
    data = [{'scene': str(scene), 'cases': [{'case': r['case'].name, 'machine': r['case'].what, 'env': r['case'].env,
                                             'passed': r['passed'], 'why': r['why'], 'render': r['render'],
                                             'comparison': r.get('comparison'), 'fallback': r['report'].get('fallback'),
                                             'folder': str(r['folder'])} for r in results]}
            for scene, results in scenes_results]
    (out / 'report.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exe', type=Path, required=True, help='moonray_xpu2.exe of a session built with the extension')
    p.add_argument('--scene', type=Path, action='append', required=True,
                   help='a scene (.rdla, with its .rdlb beside it if any); each is tested on its own')
    p.add_argument('--output', type=Path, required=True, help='a new folder')
    p.add_argument('--suite', choices=('quick', 'full'), default='quick')
    p.add_argument('--cards', type=int, default=3, help='most simulated cards (full suite)')
    p.add_argument('--groups', type=int, default=3, help='most simulated processor groups (full suite)')
    p.add_argument('--only', default='', help='these cases alone, by name (base-a and base-b always run)')
    p.add_argument('--timeout', type=float, default=600, help='seconds a render may take')
    p.add_argument('--env', action='append', default=[], help='KEY=VALUE for every render')
    p.add_argument('--no-denoise', action='store_true', help='leave out the realtime denoiser cases')
    p.add_argument('--oidn', type=Path, help="the folder holding Open Image Denoise (MoonRay's runtime), where the session "
                   "has none beside it: an installed kit has, a test build has not")
    a = p.parse_args()
    if subprocess.run(['tasklist'], capture_output=True, text=True).stdout.lower().count('modo.exe'):
        sys.exit('Modo is open: close it first (the renders take the graphics card)')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    extra_env = dict(v.split('=', 1) for v in a.env)
    if a.oidn:
        extra_env['XPU2_SESSION_OIDN_DIR'] = str(a.oidn.resolve())
    chosen = cases(a.suite, max(2, a.cards), max(2, a.groups))
    if a.only:
        wanted = set(a.only.split(',')) | {'base-a', 'base-b'}
        if any(c.name.startswith('denoise-') for c in chosen if c.name in wanted):
            wanted.add('denoise-optix')
        chosen = [c for c in chosen if c.name in wanted]
    if a.no_denoise:
        chosen = [c for c in chosen if not c.realtime]
    scenes_results = []
    for i, scene in enumerate(a.scene):
        scene = scene.resolve()
        files = [scene] + ([scene.with_suffix('.rdlb')] if scene.with_suffix('.rdlb').is_file() else [])
        realtime = realtime_scene(out / ('scene-%d' % i))
        results, by_name = [], {}
        for case in chosen:
            print('%s: %s (%s)' % (scene.name, case.name, case.what), flush=True)
            r = run_case(case, a.exe.resolve(), files, realtime if case.realtime else None, out / ('scene-%d' % i), a.timeout, extra_env)
            by_name[case.name] = r
            if case.name == 'base-b':  # the scene's own noise, against base-a
                r['comparison'] = compare(by_name['base-a']['picture'] if 'base-a' in by_name else None, r['picture'])
            judge(r, by_name)
            if case.name == 'base-b':
                r['comparison'] = compare(by_name['base-a']['picture'] if 'base-a' in by_name else None, r['picture'])
            print('  %s%s' % ('skipped: ' + r['skipped'] if r.get('skipped') else 'pass' if r['passed'] else 'FAIL',
                              (': ' + '; '.join(r['why'])) if r['why'] else ''), flush=True)
            results.append(r)
        scenes_results.append((scene, results))
    passed = page(scenes_results, out, a.exe)
    print('%s; report: %s' % ('all pass' if passed else 'failures', out / 'report.html'))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

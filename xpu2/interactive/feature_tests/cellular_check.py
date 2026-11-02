"""cellular_check.py <folder> <xpu2 exe> <scene.rdla with OUTPUT> [--reference-spp-root 32]: XPU2's adaptive frame by
cellular generations and by MoonRay's trees, and MoonRay's adaptive frame, each against MoonRay's uniform render at
root^2 samples a pixel (the converged image): block error, mean ratio, time. The converged reference is kept between
runs."""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
HARNESS = S.parent / 'session_harness.py'
RT = Path(r'C:\Users\Raphael Tobar\AppData\Roaming\Luxology\Kits\MoonRayForModo\runtime')
MOONRAY = RT / 'moonray.exe'
TOOL = RT / 'oiiotool.exe'
p = argparse.ArgumentParser()
p.add_argument('folder', type=Path)
p.add_argument('xpu2', type=Path)
p.add_argument('scene', type=Path)
p.add_argument('--reference-spp-root', type=int, default=32)
p.add_argument('--threads', type=int, default=11)
p.add_argument('--runs', default='reference,moonray,cellular,trees')
a = p.parse_args()
folder = a.folder.resolve()
folder.mkdir(parents=True, exist_ok=True)
base = a.scene.read_text(encoding='utf-8')


def run(name, exe, mode, text, env=()):
    out = folder / ('%s.exr' % name)
    scene = folder / ('%s.rdla' % name)
    scene.write_text(text.replace('OUTPUT', out.as_posix()), encoding='utf-8')
    if out.exists():
        out.unlink()
    log = folder / ('%s-harness' % name)
    if log.exists():
        shutil.rmtree(log)
    args = [sys.executable, str(HARNESS), '--exe', str(exe), '--scene', str(scene), '--output', str(log), '--threads', str(a.threads),
            '--mode', mode, '--timeout', '3600']
    for e in env:
        args += ['--env', e]
    subprocess.run(args, capture_output=True, text=True)
    rep = json.loads((log / 'report.json').read_text()) if (log / 'report.json').exists() else {}
    return out, rep


results = {}
runs = a.runs.split(',')
reference = folder / 'reference.exr'
if 'reference' in runs and not reference.exists():
    uniform = re.sub(r'\["sampling_mode"\]=\d+', '["sampling_mode"]=0', base)
    uniform = re.sub(r'\["pixel_samples"\]=\d+', '["pixel_samples"]=%d' % a.reference_spp_root, uniform)
    _, rep = run('reference', MOONRAY, 'vectorized', uniform)
    results['reference'] = {'seconds': rep.get('seconds')}
uniform_text = re.sub(r'\["pixel_samples"\]=\d+', '["pixel_samples"]=16', re.sub(r'\["sampling_mode"\]=\d+', '["sampling_mode"]=0', base))
plan = {'moonray': (MOONRAY, 'vectorized', ()), 'cellular': (a.xpu2, 'xpu', ('XPU2_SESSION_MODE=0',)),
        'uniform': (a.xpu2, 'xpu', ('XPU2_SESSION_MODE=0',)), 'moonray-uniform': (MOONRAY, 'vectorized', ()),
        'trees': (a.xpu2, 'xpu', ('XPU2_SESSION_MODE=0', 'XPU2_SESSION_ADAPTIVE=trees'))}
for name in runs:
    if name not in plan:
        continue
    exe, mode, env = plan[name]
    out, rep = run(name, exe, mode, uniform_text if 'uniform' in name else base, env)
    entry = {'seconds': rep.get('seconds'), 'fallback': rep.get('fallback')}
    if out.exists() and reference.exists():
        cmp = subprocess.run([sys.executable, str(S / 'block_cmp.py'), str(TOOL), str(reference), str(out)], capture_output=True, text=True,
                             env={'PATH': str(RT), 'SYSTEMROOT': r'C:\Windows'}).stdout
        entry['compare'] = cmp.strip()
    results[name] = entry
    print(name, json.dumps(entry), flush=True)
(folder / ('results-%s.json' % '-'.join(runs))).write_text(json.dumps(results, indent=2), encoding='utf-8')

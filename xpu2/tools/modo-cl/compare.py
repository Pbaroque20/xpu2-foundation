"""compare.py <scene name> <width> <height> [--samples N] [--moonray] [--xpu2 <session N>] [--mode 0|1|2]: one of the
exported scenes (scenes/<name>/<name>.rdla) at a smaller size and uniform samples (pixel_samples N, 8 = 64 a pixel),
rendered by the installed kit's MoonRay (vector mode) as the reference and by an XPU2 session through the harness
(uniform exactly, XPU2_SESSION_UNIFORM_BUDGET=0); the beauty of each saved as scenes/<name>/cmp/<who>.exr and the
display error between them printed (8-bit steps at exposure 1, sRGB)."""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RT = Path(os.environ['APPDATA']) / 'Luxology' / 'Kits' / 'MoonRayForModo' / 'runtime'
HARNESS = HERE.parents[1] / 'interactive' / 'session_harness.py'
TR = Path(r'C:\Users\Raphael Tobar\MoonRayForModo-xpu2\test-results')
p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('width', type=int)
p.add_argument('height', type=int)
p.add_argument('--samples', type=int, default=8)
p.add_argument('--moonray', action='store_true')
p.add_argument('--xpu2', default='')
p.add_argument('--mode', default='2')
p.add_argument('--timeout', type=int, default=900)
p.add_argument('--extra', default='', help='more RDLA for the overlay (both renderers)')
p.add_argument('--tag', default='', help='a name for this comparison, kept apart from the plain one')
p.add_argument('--env', action='append', default=[], help='KEY=VALUE for both renderers (the kit sets its Source Overrides so)')
a = p.parse_args()
folder = HERE / 'scenes' / a.name
scene = folder / (a.name + '.rdla')
out = folder / ('cmp' + ('-' + a.tag if a.tag else ''))
out.mkdir(exist_ok=True)
overlay = out / 'overlay.rdla'
overlay.write_text('SceneVariables { ["image_width"] = %d, ["image_height"] = %d, ["pixel_samples"] = %d, ["sampling_mode"] = 0 }\n'
                   % (a.width, a.height, a.samples) + (a.extra + '\n' if a.extra else ''), encoding='utf-8')
env = dict(os.environ, PATH=str(RT) + os.pathsep + os.environ.get('PATH', ''), RDL2_DSO_PATH=str(RT))
env.update(dict(v.split('=', 1) for v in a.env))
for k in ('PYTHONHOME', 'PYTHONPATH'):
    env.pop(k, None)
written = folder / (a.name + '.exr')


def beauty(source, target):
    subprocess.run([str(RT / 'oiiotool.exe'), str(source), '--ch', 'R,G,B', '-d', 'float', '-o', str(target)], check=True, env=env,
                   capture_output=True)


def pixels(path):
    pfm = path.with_suffix('.pfm')
    subprocess.run([str(RT / 'oiiotool.exe'), str(path), '--ch', '0,1,2', '-o', str(pfm)], check=True, env=env, capture_output=True)
    data = pfm.read_bytes()
    parts = data.split(b'\n', 3)
    w, h = map(int, parts[1].split())
    return struct.unpack('<%df' % (w * h * 3), parts[3][:w * h * 12])


if a.moonray:
    if written.exists():
        written.unlink()
    began = time.time()
    r = subprocess.run([str(RT / 'moonray.exe'), '-in', str(scene), '-in', str(overlay), '-exec_mode', 'vector', '-info'],
                       env=env, capture_output=True, text=True, timeout=a.timeout, cwd=str(out))
    (out / 'moonray.log').write_text(r.stdout + r.stderr, encoding='utf-8')
    print('MoonRay exit', r.returncode, 'in %.1f s' % (time.time() - began))
    beauty(written, out / 'moonray.exr')
if a.xpu2:
    exe = TR / ('xpu2-session-%s' % a.xpu2) / 'runtime-xpu2' / 'moonray_xpu2.exe'
    run = out / ('x' + a.xpu2 + '-m' + a.mode)
    shutil.rmtree(run, ignore_errors=True)
    if written.exists():
        written.unlink()
    began = time.time()
    r = subprocess.run([sys.executable, str(HARNESS), '--exe', str(exe), '--scene', str(scene), '--scene', str(overlay), '--threads', '12',
                        '--env', 'XPU2_SESSION_UNIFORM_BUDGET=0', '--env', 'XPU2_SESSION_MODE=' + a.mode] + [x for v in a.env for x in ('--env', v)] + [ '--output', str(run),
                        '--timeout', str(a.timeout)], capture_output=True, text=True)
    print('XPU2', r.stdout.strip()[-300:], 'in %.1f s' % (time.time() - began))
    lines = (run / 'protocol.log').read_text(encoding='utf-8', errors='replace').splitlines()
    for line in lines:
        if 'FALLBACK' in line or 'finished in' in line or 'scene ready' in line:
            print('  ', line.strip()[:240])
    if written.exists():
        beauty(written, out / ('xpu2-%s-m%s.exr' % (a.xpu2, a.mode)))
if (out / 'moonray.exr').exists():
    ref = pixels(out / 'moonray.exr')

    def shown(v):
        v = min(max(v, 0.0), 1.0)
        return 255.0 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)
    for other in sorted(out.glob('xpu2-*.exr')):
        img = pixels(other)
        if len(img) != len(ref):
            print(other.name, 'size differs')
            continue
        err = (sum((shown(x) - shown(y)) ** 2 for x, y in zip(img, ref)) / len(ref)) ** 0.5
        mean_ref, mean_img = sum(ref) / len(ref), sum(img) / len(img)
        print('%-24s display error %.2f (8-bit steps); mean %.4f against %.4f (ratio %.4f)' % (other.name, err, mean_img, mean_ref,
                                                                                             mean_img / mean_ref if mean_ref else 0))

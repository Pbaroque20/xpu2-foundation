"""xpu2_only.py <folder> <xpu2 exe> <case> [--mode N] [--name xpu2]: renders <case>-<name>.exr with XPU2 alone, from the
case's MoonRay scene that feature_compare.py wrote (its outputs pointed at the new name)."""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runq
HARNESS = Path(__file__).resolve().parents[1] / 'session_harness.py'
p = argparse.ArgumentParser()
p.add_argument('folder', type=Path); p.add_argument('xpu2', type=Path); p.add_argument('case')
p.add_argument('--mode', default='0'); p.add_argument('--name', default='xpu2'); p.add_argument('--threads', type=int, default=0)
a = p.parse_args()
folder = a.folder.resolve()
text = (folder / ('%s-moonray.rdla' % a.case)).read_text(encoding='utf-8')
out = folder / ('%s-%s.exr' % (a.case, a.name))
scene = folder / ('%s-%s.rdla' % (a.case, a.name))
scene.write_text(text.replace((folder / ('%s-moonray.exr' % a.case)).as_posix(), out.as_posix()), encoding='utf-8')
if out.exists(): out.unlink()
log = folder / ('%s-%s-harness' % (a.case, a.name))
if log.exists(): shutil.rmtree(log)
with runq.slot():
    r = subprocess.run([sys.executable, str(HARNESS), '--exe', str(a.xpu2), '--scene', str(scene), '--output', str(log), '--threads', str(a.threads or runq.threads()),
                        '--mode', 'xpu', '--timeout', '1800', '--env', 'XPU2_SESSION_MODE=' + a.mode], capture_output=True, text=True)
rep = json.loads((log / 'report.json').read_text()) if (log / 'report.json').exists() else {}
print(a.case, a.name, 'seconds', rep.get('seconds'), 'fallback', rep.get('fallback'), 'written', out.exists())
if not out.exists(): print(r.stdout[-1500:], r.stderr[-1500:])

"""Make test-results/xpu2-candidate-042: the disabled-route candidate's build files with a runtime staged on release 0.4.2."""
import json, shutil, subprocess, sys
from pathlib import Path

X = Path(r'C:\Users\Raphael Tobar\MoonRayForModo-xpu2')
sys.path[:0] = [str(Path(__file__).resolve().parents[3])]  # the repository, for xpu2.reference
from xpu2.reference import digest_file

old = X / 'test-results/xpu2-disabled-route-build'
new = X / 'test-results/xpu2-candidate-042'
rel = Path(r'C:\Users\Raphael Tobar\MoonRayForModo\runtime\release-0.4.2')
subprocess.run([sys.executable, str(X / 'tools/xpu2_stage_runtime.py'), '--base', str(rel), '--output', str(new),
                '--replace', 'librendering_rndr.dll=' + str(old / 'runtime/librendering_rndr.dll')], check=True)
staged = json.loads((new / 'report.json').read_text())
for f in old.iterdir():
    if f.is_file() and f.name != 'report.json':
        shutil.copyfile(f, new / f.name)
report = json.loads((old / 'report.json').read_text())
report.update(rebased_from=str(old), runtime_base=str(rel), staging=staged,
              dll_sha256=digest_file(new / 'runtime/librendering_rndr.dll'))
assert report['dll_sha256'] == json.loads((old / 'report.json').read_text())['dll_sha256']
(new / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print('candidate folder', new)

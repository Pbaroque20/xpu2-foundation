"""Check the frozen 12-worker image gate against held-out original-XPU renders.

The envelopes and their five reference images come unchanged from an earlier
material-ceiling report. New original-XPU renders are scored exactly like a
candidate: if they also fail, the gate measures original XPU's own variation,
not XPU2. An optional candidate build (all XPU2 options unset) is scored the
same way, and is also compared word for word against original XPU at 1 worker.
Each run needs a fresh output directory. No envelope is refitted.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from xpu2.reference import digest_file, inventory, verify
from xpu2.images import read_rgba
from xpu2.material_job_metrics import METRICS, compare, qualified
from xpu2_cpu_cost_probe import renderer_pids
from xpu2_baseline import environment
from xpu2_repeatability import successful, gpu_identity


def render(runtime, scene, folder, threads, timeout=900):
    """One native render; refuses to time alongside another MoonRay process."""
    folder.mkdir()
    if renderer_pids():
        raise ValueError('Competing renderer active; no render started')
    env = environment(runtime, folder)
    start = time.perf_counter(); competition = set()
    with (folder / 'render.log').open('w') as log:
        proc = subprocess.Popen([str(runtime / 'moonray.exe'), '-in', str(scene), '-out', str(folder / 'beauty.exr'),
                                 '-exec_mode', 'xpu', '-threads', str(threads), '-info', '-stats', str(folder / 'stats.csv')],
                                env=env, cwd=folder, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                competition.update(pid for pid in renderer_pids() if pid != proc.pid)
                if time.monotonic() > deadline:
                    raise TimeoutError('Render exceeded %d seconds' % timeout)
                try:
                    proc.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait(timeout=10)
    wall = time.perf_counter() - start
    if competition:
        raise ValueError('Competing renderer during render ' + str(sorted(competition)))
    text = (folder / 'render.log').read_text(errors='replace')
    stats = (folder / 'stats.csv').read_text(errors='replace') if (folder / 'stats.csv').exists() else ''
    execution = successful('xpu', threads, proc.returncode, text, stats, (folder / 'beauty.exr').exists())
    if not execution['passed']:
        raise ValueError('Native execution failed in ' + str(folder))
    image = read_rgba(runtime, folder / 'beauty.exr', folder, 'raw', env)
    return dict(folder=str(folder), wall_seconds=wall, execution=execution, image_sha256=image['sha256']), image


def margins(comparisons, bound):
    """Largest ratio of each metric to its frozen bound over all reference comparisons."""
    out = {}
    for c in 'RGBA':
        for m in METRICS:
            worst = max(row[c][m] for row in comparisons)
            limit = bound[c][m]
            # A nonzero difference against a zero bound has no finite ratio.
            out[c + '.' + m] = (worst / limit) if limit else (0.0 if worst == 0 else 'exceeds_zero_bound')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='Original XPU runtime')
    p.add_argument('--frozen', type=Path, required=True, help='Material-ceiling 12-worker report.json')
    p.add_argument('--candidate', type=Path, help='Candidate build folder (report.json + runtime)')
    p.add_argument('--heldout', type=int, default=5)
    p.add_argument('--names', default='diffuse,layered,textured,many_adaptive')
    p.add_argument('--skip-one-worker', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', scope=__doc__.strip().splitlines()[0], heldout=a.heldout, fixtures={})

    def save():
        temporary = out / 'report.pending.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, out / 'report.json')
    try:
        reference = a.reference.resolve()
        frozen_path = a.frozen.resolve(); frozen = json.loads(frozen_path.read_text())
        if frozen.get('threads') != 12 or frozen.get('bridge_enabled') is not False:
            raise ValueError('Frozen report must be the 12-worker, bridge-disabled ceiling run')
        report['frozen'] = dict(path=str(frozen_path), sha256=digest_file(frozen_path))
        candidate = None
        if a.candidate:
            build = a.candidate.resolve(); b = json.loads((build / 'report.json').read_text())
            candidate = build / 'runtime'
            if b['status'] != 'built' or digest_file(candidate / 'librendering_rndr.dll') != b['dll_sha256']:
                raise ValueError('Candidate build drift')
            report['candidate'] = dict(build=str(build), stage=b.get('stage'), dll_sha256=b['dll_sha256'],
                                       report_sha256=digest_file(build / 'report.json'))
        held = {str(r): inventory(r) for r in filter(None, (reference, candidate))}
        frozen_reference = next(k for k in frozen['runtime_inventories'] if k.endswith('control'))
        if frozen['runtime_inventories'][frozen_reference] != held[str(reference)]:
            raise ValueError('Reference runtime differs from the frozen report')
        sources = {str(f): digest_file(f) for f in (Path(__file__), ROOT / 'xpu2/material_job_metrics.py')}
        report.update(gpu=gpu_identity(), logical_cpus=os.cpu_count(), source_hashes=sources)
        if report['gpu'] != frozen['gpu']:
            raise ValueError('GPU/driver differs from the frozen report')
        env = environment(reference, out)
        for name in a.names.split(','):
            scene = next(Path(f) for f in frozen['asset_hashes'] if Path(f).name == name + '.rdla')
            if digest_file(scene) != frozen['asset_hashes'][str(scene)]:
                raise ValueError('Fixture changed: ' + str(scene))
            bound = frozen['envelopes'][name]
            refs = sorted((r for r in frozen['runs'] if r['fixture'] == name and r['label'].startswith('reference')),
                          key=lambda r: r['label'])
            if len(refs) != 5:
                raise ValueError('Expected five frozen references for ' + name)
            images = []
            for r in refs:
                exr = Path(r['artifact_folder']) / 'beauty.exr'
                if digest_file(exr) != r['image_sha256']:
                    raise ValueError('Frozen reference image changed: ' + str(exr))
                images.append(read_rgba(reference, exr, out, name + '-' + r['label'], env))
            entry = report['fixtures'][name] = dict(scene=str(scene), bound=bound, runs=[])
            save()
            # Interleave so slow drift in machine state affects both groups alike.
            plan = []
            for i in range(a.heldout):
                plan.append(('xpu_heldout%d' % i, reference))
                if candidate:
                    plan.append(('candidate%d' % i, candidate))
            for label, runtime in plan:
                print(name, label, flush=True)
                row, image = render(runtime, scene, out / (name + '-' + label), 12)
                comparisons = [compare(ref, image) for ref in images]
                row.update(label=label, qualified=qualified(comparisons, bound), margins=margins(comparisons, bound),
                           comparisons=comparisons)
                entry['runs'].append(row); save()
            for group in ('xpu_heldout', 'candidate'):
                rows = [r for r in entry['runs'] if r['label'].startswith(group)]
                if rows:
                    entry[group + '_failed'] = sum(not r['qualified'] for r in rows)
                    entry[group + '_runs'] = len(rows)
            if candidate and not a.skip_one_worker:
                print(name, 'one-worker exactness', flush=True)
                r1, i1 = render(reference, scene, out / (name + '-xpu-1'), 1)
                c1, j1 = render(candidate, scene, out / (name + '-candidate-1'), 1)
                exact = qualified([compare(i1, j1)])
                entry['one_worker'] = dict(reference=r1, candidate=c1, bit_exact=exact)
            save()
        for r in filter(None, (reference, candidate)):
            verify(r, held[str(r)])
        for f, h in sources.items():
            if digest_file(Path(f)) != h:
                raise ValueError('Source changed during measurement ' + f)
        fx = report['fixtures'].values()
        report['xpu_heldout_failed'] = sum(e.get('xpu_heldout_failed', 0) for e in fx)
        report['xpu_heldout_runs'] = sum(e.get('xpu_heldout_runs', 0) for e in fx)
        if candidate:
            report['candidate_failed'] = sum(e.get('candidate_failed', 0) for e in fx)
            report['candidate_runs'] = sum(e.get('candidate_runs', 0) for e in fx)
            report['one_worker_exact'] = all(e.get('one_worker', {}).get('bit_exact', True) for e in fx)
        report['status'] = 'complete'
    except Exception as e:
        import traceback
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
    save(); print(report['status'], report.get('error', ''), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

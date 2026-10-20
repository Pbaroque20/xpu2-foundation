"""Material/maps and selected-light frame ceilings on real MoonRay scenes.

Runs the sampled CPU-cost build (GPU shading disabled) on scenes made of an
.rdla plus optional .rdlb, rendered from their own folder so relative texture
paths resolve. Sampling is forced to fixed pixel samples so original XPU and
the candidate do identical work; one-worker images must match word for word.
Each scene gets: original XPU, candidate off, sampled 1/64, sampled 1/16,
candidate off again. Each run needs a fresh output directory.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from xpu2.reference import digest_file, inventory, verify
from xpu2.images import read_rgba
from xpu2.material_job_metrics import compare, qualified, cpu_totals
from xpu2_cpu_cost_probe import renderer_pids
from xpu2_baseline import environment
from xpu2_repeatability import successful, gpu_identity


def seconds(log, label):
    m = re.search(re.escape(label) + r'\s*=\s*(\d+):(\d+):(\d+(?:\.\d+)?)', log)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def percent(log, label):
    m = re.search(re.escape(label) + r'\s*=\s*([0-9.]+)%', log)
    return float(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True, help='cpu_cost_full build folder')
    p.add_argument('--reference', type=Path, required=True, help='Original XPU runtime')
    p.add_argument('--scenes', type=Path, nargs='+', required=True, help='Folders holding scene.rdla [+ scene.rdlb]')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--res', default='4', help='MoonRay -res divisor')
    p.add_argument('--pixel-samples', default='4', help='Square root of fixed samples per pixel')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', threads=a.threads, res=a.res, pixel_samples=a.pixel_samples,
                  bridge_enabled=False, scenes={})

    def save():
        temporary = out / 'report.pending.json'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temporary, out / 'report.json')
    try:
        build = a.build.resolve(); rt = build / 'runtime'; ref = a.reference.resolve()
        b = json.loads((build / 'report.json').read_text())
        if b['status'] != 'built' or digest_file(rt / 'librendering_rndr.dll') != b['dll_sha256']:
            raise ValueError('Invalid CPU scope build')
        held = {str(r): inventory(r) for r in (rt, ref)}
        sources = {str(f): digest_file(f) for f in (Path(__file__), ROOT / 'xpu2/material_job_metrics.py')}
        report.update(build=str(build), build_dll_sha256=b['dll_sha256'], reference=str(ref), gpu=gpu_identity(),
                      logical_cpus=os.cpu_count(), source_hashes=sources)
        for folder in a.scenes:
            folder = folder.resolve()
            inputs = [f for f in (folder / 'scene.rdla', folder / 'scene.rdlb') if f.exists()]
            if not inputs or inputs[0].suffix != '.rdla':
                raise ValueError('Missing scene.rdla in ' + str(folder))
            entry = report['scenes'][folder.name] = dict(folder=str(folder), inputs={str(f): digest_file(f) for f in inputs},
                                                         runs=[])
            reference_image = None
            for label, rate, seed in (('reference', 0, 0), ('off_before', 0, 0), ('sample64', 64, 173),
                                      ('sample16', 16, 761), ('off_after', 0, 0)):
                runtime = ref if label == 'reference' else rt
                target = out / (folder.name + '-' + label); target.mkdir()
                env = environment(runtime, target)
                if rate:
                    env.update(XPU2_CPU_COST=str(target / 'cpu.csv'), XPU2_CPU_COST_RATE=str(rate),
                               XPU2_CPU_COST_SEED=str(seed))
                if renderer_pids():
                    raise ValueError('Competing renderer active; no timing started')
                print(folder.name, label, flush=True)
                command = [str(runtime / 'moonray.exe')]
                for f in inputs:
                    command += ['-in', f.name]
                command += ['-out', str(target / 'beauty.exr'), '-exec_mode', 'xpu', '-threads', str(a.threads),
                            '-res', a.res, '-scene_var', 'sampling_mode', '0', '-scene_var', 'pixel_samples',
                            a.pixel_samples, '-info', '-stats', str(target / 'stats.csv')]
                start = time.perf_counter(); competition = set()
                with (target / 'render.log').open('w') as log:
                    proc = subprocess.Popen(command, env=env, cwd=folder, stdout=log, stderr=log,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
                    deadline = time.monotonic() + a.timeout
                    try:
                        while proc.poll() is None:
                            competition.update(pid for pid in renderer_pids() if pid != proc.pid)
                            if time.monotonic() > deadline:
                                raise TimeoutError('Render exceeded timeout')
                            try:
                                proc.wait(timeout=.25)
                            except subprocess.TimeoutExpired:
                                pass
                    finally:
                        if proc.poll() is None:
                            proc.kill(); proc.wait(timeout=10)
                wall = time.perf_counter() - start
                if competition:
                    raise ValueError('Competing renderer contaminated timing ' + str(sorted(competition)))
                text = (target / 'render.log').read_text(errors='replace')
                stats = (target / 'stats.csv').read_text(errors='replace') if (target / 'stats.csv').exists() else ''
                execution = successful('xpu', a.threads, proc.returncode, text, stats, (target / 'beauty.exr').exists())
                row = dict(label=label, rate=rate, seed=seed, folder=str(target), process_wall_seconds=wall,
                           execution=execution, mcrt_seconds=seconds(text, 'Mcrt time'),
                           total_seconds=seconds(text, 'Total time'),
                           moonray_shading_percent=percent(text, 'Shading % of MCRT time'))
                entry['runs'].append(row); save()
                if not execution['passed']:
                    raise ValueError('Native execution failed %s %s' % (folder.name, label))
                image = read_rgba(runtime, target / 'beauty.exr', target, 'raw', env)
                row['image_sha256'] = image['sha256']
                if label == 'reference':
                    reference_image = image
                else:
                    row['bit_exact'] = qualified([compare(reference_image, image)])
                if rate:
                    row['cpu'] = cpu_totals(target / 'cpu.csv')
                save()
            sampled = [r for r in entry['runs'] if r['rate']]
            entry['frame_seconds_reference'] = entry['runs'][0]['total_seconds']
            entry['mcrt_seconds_reference'] = entry['runs'][0]['mcrt_seconds']
            entry['material_maps_worker_seconds'] = [r['cpu']['material_maps_worker_seconds'] for r in sampled]
            entry['selected_light_worker_seconds'] = [r['cpu']['selected_lighting_worker_seconds'] for r in sampled]
            entry['all_bit_exact'] = all(r.get('bit_exact', True) for r in entry['runs'])
            save()
        for r in (rt, ref):
            verify(r, held[str(r)])
        for f, h in sources.items():
            if digest_file(Path(f)) != h:
                raise ValueError('Source changed during measurement ' + f)
        report['status'] = 'complete'
    except Exception as e:
        import traceback
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
    save(); print(report['status'], report.get('error', ''), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

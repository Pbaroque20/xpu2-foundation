"""compare_folder.py <folder> --xpu2 <session N> [--moonray] [--samples N] [--tag t] [--env K=V ...] [--memory-mb M]: a
scene folder's preview.rdla + preview.rdlb at uniform samples, rendered by the pinned reference MoonRay
(interactive-reference-042, vector mode) and by an XPU2 session through the harness (uniform exactly); the beauties
saved under <folder>/cmp-<tag>/ and the display error between them printed (8-bit steps, sRGB). Every render runs in
a Windows job object capped at --memory-mb (default: half the machine's free RAM) and at below-normal priority, so a
render that asks for too much fails instead of taking the machine down."""
import argparse
import ctypes
import ctypes.wintypes as w
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

TR = Path(r'C:\Users\Raphael Tobar\MoonRayForModo-xpu2\test-results')
RT = TR / 'interactive-reference-042'
HARNESS = Path(__file__).resolve().parents[2] / 'interactive' / 'session_harness.py'
KIT_ENV = ['MOONRAY_MODO_%s=1' % k for k in ('TOON_TERMINATOR', 'TOON_ALBEDO', 'DENOISE_CLEAN_GUIDES', 'CRYPTO_VOLUMES', 'FOG_ALPHA', 'CRYPTO_ONCE',
                                            'USD_SPECULAR', 'OPAQUE_ALPHA', 'TRANSMISSION_ROUGHNESS')]
p = argparse.ArgumentParser()
p.add_argument('folder', type=Path)
p.add_argument('--xpu2', default='')
p.add_argument('--moonray', action='store_true')
p.add_argument('--samples', type=int, default=4)
p.add_argument('--tag', default='')
p.add_argument('--env', action='append', default=[])
p.add_argument('--timeout', type=int, default=900)
p.add_argument('--memory-mb', type=int, default=0)
a = p.parse_args()
a.folder = a.folder.resolve()
if 'modo.exe' in subprocess.run(['tasklist'], capture_output=True, text=True).stdout.lower():
    sys.exit('Modo is running: no GPU tests while the user tests in Modo')
out = a.folder / ('cmp' + ('-' + a.tag if a.tag else ''))
out.mkdir(exist_ok=True)
overlay = out / 'overlay.rdla'
overlay.write_text('SceneVariables { ["pixel_samples"] = %d, ["sampling_mode"] = 0 }\n' % a.samples, encoding='utf-8')
env = dict(os.environ, PATH=str(RT) + os.pathsep + os.environ.get('PATH', ''), RDL2_DSO_PATH=str(RT))
for v in KIT_ENV + a.env:
    k, x = v.split('=', 1)
    env[k] = x
for k in ('PYTHONHOME', 'PYTHONPATH'):
    env.pop(k, None)
written = a.folder / 'preview.exr'

k32 = ctypes.WinDLL('kernel32', use_last_error=True)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]


class BASIC(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', w.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class EXTENDED(ctypes.Structure):
    _fields_ = [('Basic', BASIC), ('Io', IO_COUNTERS), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


k32.CreateJobObjectW.restype = w.HANDLE
k32.OpenProcess.restype = w.HANDLE
k32.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
k32.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
k32.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]


def free_mb():
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    k32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys >> 20


def capped(args, **kw):
    """args run in a job of their own (it and every process it starts) capped at the memory limit: (result, peak MB)."""
    limit = (a.memory_mb or free_mb() // 2) << 20
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x200 | 0x2000  # JOB_MEMORY, KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = limit
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    proc = subprocess.Popen(args, creationflags=0x4 | 0x4000, **kw)  # suspended until it is in the job; below normal
    handle = k32.OpenProcess(0x1F0FFF, False, proc.pid)
    k32.AssignProcessToJobObject(job, handle)
    # Resume its first thread (NtResumeProcess).
    ctypes.WinDLL('ntdll').NtResumeProcess(w.HANDLE(handle))
    try:
        stdout, stderr = proc.communicate(timeout=a.timeout + 60)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    k32.QueryInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info), None)
    k32.CloseHandle(handle)
    k32.CloseHandle(job)
    return proc.returncode, stdout, stderr, info.PeakJobMemoryUsed >> 20, limit >> 20


def beauty(source, target):
    subprocess.run([str(RT / 'oiiotool.exe'), str(source), '--ch', 'R,G,B', '-d', 'float', '-o', str(target)], check=True, env=env, capture_output=True)


def pixels(path):
    pfm = path.with_suffix('.pfm')
    subprocess.run([str(RT / 'oiiotool.exe'), str(path), '--ch', '0,1,2', '-o', str(pfm)], check=True, env=env, capture_output=True)
    data = pfm.read_bytes()
    parts = data.split(b'\n', 3)
    wd, ht = map(int, parts[1].split())
    return struct.unpack('<%df' % (wd * ht * 3), parts[3][:wd * ht * 12])


if a.moonray:
    if written.exists():
        written.unlink()
    began = time.time()
    code, so, se, peak, limit = capped([str(RT / 'moonray.exe'), '-in', str(a.folder / 'preview.rdla'), '-in', str(a.folder / 'preview.rdlb'), '-in', str(overlay),
                                        '-exec_mode', 'vector', '-info'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(out))
    (out / 'moonray.log').write_text((so or '') + (se or ''), encoding='utf-8')
    print('MoonRay exit %s in %.1f s, peak %d MB of a %d MB cap' % (code, time.time() - began, peak, limit))
    if written.exists():
        beauty(written, out / 'moonray.exr')
if a.xpu2:
    exe = TR / ('xpu2-session-%s' % a.xpu2) / 'runtime-xpu2' / 'moonray_xpu2.exe'
    run = out / ('x' + a.xpu2)
    shutil.rmtree(run, ignore_errors=True)
    if written.exists():
        written.unlink()
    began = time.time()
    args = [sys.executable, str(HARNESS), '--exe', str(exe), '--scene', str(a.folder / 'preview.rdla'), '--scene', str(a.folder / 'preview.rdlb'),
            '--scene', str(overlay), '--threads', '12', '--env', 'XPU2_SESSION_UNIFORM_BUDGET=0', '--env', 'XPU2_SESSION_MODE=2',
            '--output', str(run), '--timeout', str(a.timeout)]
    for v in KIT_ENV + a.env:
        args += ['--env', v]
    code, so, se, peak, limit = capped(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print('XPU2 exit %s in %.1f s, peak %d MB of a %d MB cap: %s' % (code, time.time() - began, peak, limit, (so or '').strip()[-200:]))
    log = (run / 'protocol.log').read_text(encoding='utf-8', errors='replace').splitlines() if (run / 'protocol.log').exists() else []
    for line in log:
        if any(k in line for k in ('FALLBACK', 'finished in', 'prepared', 'memory', 'budget', 'within the plan')):
            print('  ', line.strip()[:300])
    if written.exists():
        beauty(written, out / ('xpu2-%s.exr' % a.xpu2))
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
        mr, mi = sum(ref) / len(ref), sum(img) / len(img)
        print('%-16s display error %.2f (8-bit steps); mean %.4f against %.4f (ratio %.4f)' % (other.name, err, mi, mr, mi / mr if mr else 0))

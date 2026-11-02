"""run_watched.py <session N> <scene folder> <tag> [--env K=V ...] [--timeout s] [--stop-after-ready s]: the XPU2 harness on
<scene folder>/preview.rdla + preview.rdlb with the kit's Source Override env, the GPU's memory in use sampled every
second meanwhile (nvidia-smi) and the process's own peak working set; writes <scene folder>/<tag>/ (protocol.log,
report.json) and prints the protocol's XPU2 lines with the memory peaks."""
import argparse
import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TR = Path(r'C:\Users\Raphael Tobar\MoonRayForModo-xpu2\test-results')
HARNESS = Path(__file__).resolve().parents[2] / 'interactive' / 'session_harness.py'
KIT_ENV = ['MOONRAY_MODO_%s=1' % k for k in ('TOON_TERMINATOR', 'TOON_ALBEDO', 'DENOISE_CLEAN_GUIDES', 'CRYPTO_VOLUMES', 'FOG_ALPHA', 'CRYPTO_ONCE',
                                            'USD_SPECULAR', 'OPAQUE_ALPHA', 'TRANSMISSION_ROUGHNESS')]
p = argparse.ArgumentParser()
p.add_argument('session')
p.add_argument('folder', type=Path)
p.add_argument('tag')
p.add_argument('--env', action='append', default=[])
p.add_argument('--timeout', type=int, default=900)
p.add_argument('--extra', default='', help='an overlay .rdla sent with the scene')
p.add_argument('--memory-binary', action='store_true', help='the .rdlb handed over in shared memory (session_harness.py)')
p.add_argument('--min-free-ram-gb', type=float, default=8.0, help='stop the run when the machine has less RAM free')
p.add_argument('--max-gpu-mb', type=int, default=21500, help='stop the run when the GPU holds more')
a = p.parse_args()
if 'modo.exe' in subprocess.run(['tasklist'], capture_output=True, text=True).stdout.lower():
    sys.exit('Modo is running: no GPU tests while the user tests in Modo')
out = a.folder / a.tag
exe = TR / ('xpu2-session-%s' % a.session) / 'runtime-xpu2' / 'moonray_xpu2.exe'
args = [sys.executable, str(HARNESS), '--exe', str(exe), '--scene', str(a.folder / 'preview.rdla'), '--scene', str(a.folder / 'preview.rdlb')]
if a.extra:
    args += ['--scene', a.extra]
args += ['--output', str(out), '--threads', '12', '--timeout', str(a.timeout)] + (['--memory-binary'] if a.memory_binary else [])
for v in KIT_ENV + a.env:
    args += ['--env', v]
samples = []
ram = []
stopped = []
done = threading.Event()


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_ram_gb():
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2**30


def stop_session(why):
    # Too much of the machine taken: the session is ended (the harness then reports it) before the machine slows.
    if not stopped:
        stopped.append(why)
        subprocess.run(['taskkill', '/F', '/IM', 'moonray_xpu2.exe'], capture_output=True)


def sample():
    while not done.is_set():
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        try:
            samples.append((time.time(), int(r.stdout.strip().splitlines()[0])))
            if samples[-1][1] > a.max_gpu_mb:
                stop_session('GPU memory %d MiB' % samples[-1][1])
        except (ValueError, IndexError):
            pass
        ram.append((time.time(), free_ram_gb()))
        if ram[-1][1] < a.min_free_ram_gb:
            stop_session('free RAM %.1f GB' % ram[-1][1])
        done.wait(1.0)


base = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
watcher = threading.Thread(target=sample, daemon=True)
began = time.time()
watcher.start()
ram0 = free_ram_gb()
r = subprocess.run(args, capture_output=True, text=True, creationflags=0x4000)  # BELOW_NORMAL, inherited by the session
done.set()
watcher.join()
print('harness:', r.stdout.strip()[-400:], r.stderr.strip()[-400:], 'in %.1f s' % (time.time() - began))
log = (out / 'protocol.log').read_text(encoding='utf-8', errors='replace').splitlines() if (out / 'protocol.log').exists() else []
for line in log:
    s = line.strip()
    if any(k in s for k in ('XPU2', 'xpu2', 'FALLBACK', 'APPLIED', 'DONE', 'budget', 'GPU')):
        print('  ', s[:400])
if samples:
    peak = max(samples, key=lambda x: x[1])
    print('GPU memory in use: %s MiB before, peak %d MiB at %.0f s' % (base, peak[1], peak[0] - began))
if ram:
    low = min(ram, key=lambda x: x[1])
    print('free RAM: %.1f GB before, lowest %.1f GB at %.0f s (the session took up to %.1f GB)' % (ram0, low[1], low[0] - began, ram0 - low[1]))
if stopped:
    print('STOPPED by the watchdog:', stopped[0])

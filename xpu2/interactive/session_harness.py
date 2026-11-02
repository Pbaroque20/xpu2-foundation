"""Drive a preview session as the MoonRay for Modo kit does (its persistent protocol v1), without Modo: for testing
XPU2's session (moonray_xpu2.exe) and comparing it with the kit's own MoonRay.

session_harness.py --exe <moonray.exe or moonray_xpu2.exe> --scene <scene.rdla> [--scene <more>] --output <folder>
                   [--delta <delta.rdla> ...] [--threads n] [--mode xpu] [--env KEY=VALUE ...] [--timeout s]

The first scene files start the session; each delta is sent once the frame before it is done. Every progressive
picture is taken from shared memory as the kit takes it (rows as they come, MoonRay's film order) and the last of
each frame is saved as <output>/frame-<id>.pfm; the protocol lines are kept in <output>/protocol.log and a summary
in <output>/report.json."""
import argparse
import ctypes
import json
import os
import re
import struct
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024


def receive(line, pid):
    """The kit's shared_image.receive: maps the publication, copies it, acknowledges it."""
    _, generation, key, width, height, name = line.split()
    generation, width, height = int(generation), int(width), int(height)
    if not re.fullmatch(r'Local\\MoonRayForModo_' + str(int(pid)) + r'_\d+', name):
        raise ValueError('Unexpected image mapping owner ' + name)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    api.OpenFileMappingW.restype = wintypes.HANDLE
    api.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    api.OpenEventW.restype = wintypes.HANDLE
    api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
    api.MapViewOfFile.restype = ctypes.c_void_p
    api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.SetEvent.argtypes = [wintypes.HANDLE]
    ack = api.OpenEventW(2, False, name + '_ack')
    mapping = view = None
    try:
        size = width * height * 12
        mapping = api.OpenFileMappingW(4, False, name)
        if not mapping:
            raise OSError('Image mapping unavailable')
        view = api.MapViewOfFile(mapping, 4, 0, 0, size)
        if not view:
            raise OSError('Cannot map image pixels')
        return generation, key, width, height, ctypes.string_at(view, size)
    finally:
        if view:
            api.UnmapViewOfFile(view)
        if mapping:
            api.CloseHandle(mapping)
        if ack:
            api.SetEvent(ack)
            api.CloseHandle(ack)


def save_pfm(path, width, height, pixels):
    """MoonRay's film order is PFM's own: the bottom row first."""
    with open(path, 'wb') as f:
        f.write(b'PF\n%d %d\n-1.0\n' % (width, height))
        f.write(pixels)


def environment(exe, extra):
    runtime = Path(exe).resolve().parent
    env = dict(os.environ)
    env['PATH'] = str(runtime) + os.pathsep + env.get('PATH', '')
    env['RDL2_DSO_PATH'] = str(runtime / 'rdl2dso') if (runtime / 'rdl2dso').is_dir() else str(runtime)
    env['REZ_MOONRAY_ROOT'] = str(runtime)
    env.update(extra)
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--scene', type=Path, action='append', required=True)
    p.add_argument('--delta', type=Path, action='append', default=[])
    p.add_argument('--reload', action='append', default=[], help="a whole scene sent after the deltas (the kit's 'full' command): "
                   "its files joined by |, as the kit names a text and its binary geometry")
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--mode', default='xpu')
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--stop-after', type=float, default=0, help='send the next delta this many seconds into a frame')
    p.add_argument('--drain-after', type=float, default=0, help='ask the last frame to finish with what it has after this many seconds')
    p.add_argument('--keep-pictures', action='store_true', help='save every progressive picture, as frame-<id>-<seconds>.pfm')
    p.add_argument('--memory-binary', action='store_true', help="hand a .rdlb scene over in shared memory, as the kit does for XPU2 "
                   "(modo-memory-rdlb:<length>:<name>)")
    p.add_argument('--view', default='', help="the buffer the kit shows (its view.txt), e.g. denoised_beauty for realtime's denoised pictures")
    a = p.parse_args()
    out = a.output.resolve()
    session = out / 'session'
    session.mkdir(parents=True, exist_ok=False)
    if a.view:
        (session / 'view.txt').write_text(a.view, encoding='ascii')
    extra = dict(v.split('=', 1) for v in a.env)
    env = environment(a.exe, extra)
    # MoonRay's two-stage EXR writer reads TMPDIR (as the kit sets it).
    env.update(TMPDIR=str(session), MOONRAY_MODO_SESSION=str(session), MOONRAY_MODO_GENERATION='1', MOONRAY_MODO_HOST=str(os.getpid()),
               MOONRAY_MODO_SHARED='1', MOONRAY_MODO_BUCKETS='1', MOONRAY_MODO_BUCKET_SIZE=extra.get('MOONRAY_MODO_BUCKET_SIZE', '32'))
    args = [str(a.exe.resolve())]
    held = []
    for s in a.scene:
        if a.memory_binary and s.suffix == '.rdlb':
            import mmap
            import uuid
            data = s.read_bytes()
            name = 'Local\\MoonRayForModoScene_%d_%s' % (os.getpid(), uuid.uuid4().hex)
            mapping = mmap.mmap(-1, len(data), tagname=name, access=mmap.ACCESS_WRITE)
            mapping.write(data)
            held.append(mapping)
            args += ['-in', 'modo-memory-rdlb:%d:%s' % (len(data), name)]
        else:
            args += ['-in', str(s.resolve())]
    args += ['-out', str(session / 'main.exr'), '-exec_mode', a.mode, '-info']
    if a.threads:
        args += ['-threads', str(a.threads)]
    if os.environ.get('XPU2_HARNESS_GDB'):
        # Debugging: the session under gdb, its backtrace on a fault in the log.
        args = [os.environ['XPU2_HARNESS_GDB'], '-batch', '-ex', 'set pagination off', '-ex', 'run', '-ex', 'bt 40', '-ex', 'info registers rip', '--args'] + args
    began = time.monotonic()
    proc = subprocess.Popen(args, cwd=session, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    report = {'frames': {}, 'events': [], 'fallback': None, 'tiles': 0, 'workers': 0, 'worker_rects_max': 0}
    lines = []
    state = {'applied': None, 'done': None, 'last': {}}
    lock = threading.Lock()

    def reader():
        for raw in proc.stdout:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            t = time.monotonic() - began
            with lock:
                lines.append('%8.3f %s' % (t, line if line.startswith('@@MODO_') else line[:400]))
                if line.startswith('@@MODO_SHARED '):
                    try:
                        g, key, w, h, pixels = receive(line, proc.pid)
                        f = report['frames'].setdefault(str(g), {'pictures': 0, 'first_picture': t})
                        f['pictures'] += 1
                        f.setdefault('keys', {}).setdefault(key, 0)
                        f['keys'][key] += 1
                        f['last_picture'] = t
                        state['last'][g] = (w, h, pixels)
                        if a.keep_pictures:
                            save_pfm(out / ('frame-%d-%07.3f.pfm' % (g, t)), w, h, pixels)
                    except (OSError, ValueError) as e:
                        lines.append('         picture skipped: %s' % e)
                elif line.startswith('@@MODO_SESSION '):
                    _, kind, g = line.split()
                    report['events'].append([round(t, 3), kind, int(g)])
                    if kind == 'APPLIED':
                        state['applied'] = int(g)
                    if kind in ('DONE', 'CANCELED'):
                        state['done'] = (kind, int(g))
                elif line.startswith('@@MODO_TILES '):
                    report['tiles'] += 1
                elif line.startswith('@@MODO_WORKERS '):
                    report['workers'] += 1
                    report['worker_rects_max'] = max(report['worker_rects_max'], len(line.split()) - 4)
                elif line.startswith('@@MODO_XPU2 FALLBACK'):
                    report['fallback'] = line[len('@@MODO_XPU2 FALLBACK '):]

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    def wait(pred, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end and proc.poll() is None:
            with lock:
                if pred():
                    return True
            time.sleep(0.01)
        with lock:
            return pred()

    def send(identity, mode, delta='', full=''):
        staged = session / 'command.tmp'
        staged.write_text('%d\n%s\n%s\n%s\n' % (identity, mode, delta, full), encoding='utf-8')
        staged.replace(session / 'command.txt')

    identity = 1
    try:
        for delta in a.delta:
            if a.stop_after:
                time.sleep(a.stop_after)
            else:
                wait(lambda: state['done'] and state['done'][1] == identity, a.timeout)
            identity += 1
            with lock:
                state['done'] = None
            send(identity, 'delta', str(delta.resolve()).replace('\\', '/'), str(a.scene[0].resolve()).replace('\\', '/'))
            wait(lambda: state['applied'] == identity, a.timeout)
        for whole in a.reload:
            wait(lambda: state['done'] and state['done'][1] == identity, a.timeout)
            identity += 1
            with lock:
                state['done'] = None
            files = '|'.join(str(Path(f).resolve()).replace('\\', '/') for f in whole.split('|'))
            send(identity, 'full', files.split('|')[0], files)
            wait(lambda: state['applied'] == identity, a.timeout)
        if a.drain_after:
            time.sleep(a.drain_after)
            (session / 'drain.tmp').write_text(str(identity), encoding='ascii')
            (session / 'drain.tmp').replace(session / 'drain.txt')
        wait(lambda: state['done'] and state['done'][1] == identity, a.timeout)
        send(identity + 1, 'quit')
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            # A frame that never ended: its lines are what is wanted.
            report['stuck'] = True
    finally:
        if proc.poll() is None:
            proc.kill()
        thread.join(5)
    report['exit_code'] = proc.returncode
    report['seconds'] = round(time.monotonic() - began, 3)
    for g, (w, h, pixels) in state['last'].items():
        save_pfm(out / ('frame-%d.pfm' % g), w, h, pixels)
        report['frames'][str(g)].update(width=w, height=h)
    (out / 'protocol.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('events', 'fallback', 'tiles', 'workers', 'worker_rects_max', 'exit_code', 'seconds')}))


if __name__ == '__main__':
    main()

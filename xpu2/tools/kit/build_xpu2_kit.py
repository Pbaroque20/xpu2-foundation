"""Retired 2026-10-30: XPU2 is part of the standard kit since 0.5 (kit/MoonRayForModo), packaged by
tools/package_development_release.py. Kept as a record; what follows says what this script did.

Make the private XPU2 test kit: the installed MoonRay for Modo kit (0.4.3.1) with XPU2 beside MoonRay XPU.

    python build_xpu2_kit.py --session <xpu2-session-N folder> [--base <kit folder>]

Writes MoonRayForModo/ next to this script (replacing an earlier one): the base kit, its runtime untouched, plus
runtime-xpu2/ (XPU2's own runtime from the session build) and these changes:
  - Preferences > MoonRay > XPU2: GPU renderer MoonRay XPU / XPU2, XPU2's adaptive passes on the
    graphics card, Compile XPU2 Shaders and the shaders' state;
  - the preview session starts XPU2 in MoonRay's place where it is chosen and the render's execution mode is XPU
    or Auto, and starts MoonRay instead where XPU2 says it does not render the scene (@@MODO_XPU2 FALLBACK) or ends
    unexpectedly; final renders, MoonRay LS and MoonLightIPR are MoonRay's as before.
Private: never into a repository, never published. install_xpu2.py installs it, with a backup of the kit."""
import argparse
import json
import shutil
import time
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
KIT = HERE / 'MoonRayForModo'
INSTALLED = Path(os.environ['APPDATA']) / 'Luxology' / 'Kits' / 'MoonRayForModo'
MARKERS = ('XPU-PLUS-EXPERIMENTAL.txt', 'XPU2-EXPERIMENTAL.txt', 'XPU2-README.txt')


class Patch:
    def __init__(self, path):
        self.path = path
        text = path.read_bytes().decode('utf-8')
        self.crlf = '\r\n' in text
        self.text = text.replace('\r\n', '\n')

    def rep(self, old, new):
        if self.text.count(old) != 1:
            raise RuntimeError('%s: expected once: %r' % (self.path.name, old[:80]))
        self.text = self.text.replace(old, new)

    def save(self):
        text = self.text.replace('\n', '\r\n') if self.crlf else self.text
        self.path.write_bytes(text.encode('utf-8'))


NATIVE = '''

# XPU2: its own runtime beside the kit's, chosen in Preferences > MoonRay > XPU2.
def xpu2_runtime():
    """XPU2's runtime: its preview session (moonray_xpu2.exe), GPU programs and the libraries it was built with."""
    return Path(__file__).resolve().parents[2] / 'runtime-xpu2'


def _setting(key, default):
    try:
        from PySide2 import QtCore
    except ImportError:
        return default
    return QtCore.QSettings('MoonRayForModo', settings_name()).value(key, default)


def xpu2():
    """Whether XPU2 renders the preview where MoonRay would render on the graphics card: on by default (release 0.5,
    asked 2026-10-28); a choice of MoonRay XPU in the preferences is kept."""
    return str(_setting('xpu2/renderer', 'xpu2')) == 'xpu2' and (xpu2_runtime() / 'moonray_xpu2.exe').is_file()


def xpu2_chain():
    """Whether XPU2 makes an adaptive render's passes on the graphics card: quicker, without the outlines of the work,
    which are known only where the passes are made on the processor."""
    return str(_setting('xpu2/adaptive_chain', 'false')).lower() in ('true', '1')


def xpu2_devices():
    """Which devices XPU2 renders with: 'both' (the processor and the graphics card, the default), 'gpu' or 'cpu'."""
    value = str(_setting('xpu2/devices', 'both'))
    return value if value in ('both', 'gpu', 'cpu') else 'both'


def xpu2_upscale():
    """Whether realtime's pictures shown from a quarter of their pixels while things move are made full size by OptiX's
    upscaling denoiser (Lower Resolution While Moving, on by default), or enlarged as they are."""
    return str(_setting('xpu2/upscale', 'true')).lower() in ('true', '1')


def xpu2_environment(runtime):
    """What XPU2's session starts with over environment(runtime): its own libraries and scene classes first, and the
    kit's MoonRay named for what XPU2 leaves to it."""
    root = xpu2_runtime()
    return {'PATH': str(root) + os.pathsep + environment(runtime)['PATH'], 'RDL2_DSO_PATH': str(root),
            'REZ_MOONRAY_ROOT': str(root), 'MOONRAY_XPU2_MOONRAY': str(Path(runtime) / 'moonray.exe'),
            'XPU2_SESSION_CHAIN': '1' if xpu2_chain() else '0',
            # Its devices (Preferences > MoonRay > XPU2 > Devices): 2 both, 1 the graphics card, 0 the processor.
            'XPU2_SESSION_MODE': {'both': '2', 'gpu': '1', 'cpu': '0'}[xpu2_devices()],
            # Lower Resolution While Moving (Preferences > MoonRay > XPU2): OptiX's upscaling denoiser for those pictures.
            'XPU2_SESSION_RT_UPSCALE': '1' if xpu2_upscale() else '0',
            # XPU2's tessellations kept in the cache folder as the Scene cache says (cache_folder).
            'XPU2_SESSION_GEOMETRY_CACHE': '1' if __import__('moonray_modo.cache_folder', fromlist=['scene_cache']).scene_cache() else '0'}


def xpu2_shaders_log():
    """Where Compile XPU2 Shaders writes how it is going."""
    return Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'MoonRayForModo' / 'xpu2-shaders.log'


_shaders_asked = [0.0, None]


def xpu2_shaders_compiled():
    """Whether XPU2's runtime finds its mark of compiled shaders (moonray_xpu2 --shaders-compiled), asked at most every
    few seconds: without them XPU2 does not render (persistent), and says where to compile them."""
    import subprocess, time
    if time.monotonic() - _shaders_asked[0] < 5 and _shaders_asked[1] is not None:
        return _shaders_asked[1]
    exe = xpu2_runtime() / 'moonray_xpu2.exe'
    env = dict(os.environ)
    env.update(xpu2_environment(default_runtime()))
    for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    try:
        result = subprocess.run([str(exe), '--shaders-compiled'], cwd=str(exe.parent), env=env, capture_output=True,
                                timeout=30, creationflags=0x08000000).returncode == 0
    except (OSError, subprocess.SubprocessError):
        result = False
    _shaders_asked[0], _shaders_asked[1] = time.monotonic(), result
    return result


# What a preview says when XPU2 is chosen and its shaders are not compiled: nothing is rendered.
XPU2_NOT_COMPILED = ("Render unavailable: MoonRay XPU2's shaders not compiled. "
                     "System > Preferences > MoonRay > XPU2 > Compile XPU2 Shaders.")
'''

COMMANDS = '''"""XPU2: compiling its GPU programs into OptiX's cache, and what state they are in."""
import os
import sys
package_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "python")
if package_root not in sys.path: sys.path.insert(0, package_root)
import lx
import lxu.command


def _running(pid):
    import ctypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle: return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259




def compiled():
    """Whether XPU2's runtime finds its mark of compiled shaders (native.xpu2_shaders_compiled)."""
    from moonray_modo import native
    return native.xpu2_shaders_compiled()


def _counts(lines):
    """The log's last count of tasks ("N compile tasks done, M waiting, T s", every ten seconds): (done, waiting,
    seconds), or None before the first."""
    for line in reversed(lines):
        words = line.replace(',', ' ').split()
        if 'tasks' in words and 'done' in words and 'waiting' in words:
            try:
                return (int(words[words.index('tasks') - 2]), int(words[words.index('waiting') - 1]),
                        float(words[words.index('waiting') + 1]) if len(words) > words.index('waiting') + 1 else 0.0)
            except (ValueError, IndexError):
                return None
    return None


def _progress(lines):
    """How far a compile has come, in percent; None before the first count."""
    counts = _counts(lines)
    if counts is None or not counts[0] + counts[1]:
        return None
    return int(100 * counts[0] / (counts[0] + counts[1]))


def _minutes(seconds):
    return '%d min' % max(1, round(seconds / 60.0)) if seconds >= 60 else '%d s' % max(1, round(seconds))


def last_compile():
    """How long the last whole compile took on this machine, in seconds (kept beside the log as it ends), or None."""
    from moonray_modo import native
    try:
        return float(native.xpu2_shaders_log().with_suffix('.took').read_text().strip())
    except (OSError, ValueError):
        return None


def how_long():
    """What a message says of how long a compile takes: the last one here, or that the state shows it as it goes."""
    took = last_compile()
    return 'it took %s here last time' % _minutes(took) if took else 'how long it has left shows as it goes'


def _keep_took(lines):
    """A finished compile's time ("compiled in T s") kept beside the log, for the next messages."""
    from moonray_modo import native
    for line in reversed(lines):
        if 'compiled in' in line:
            try:
                seconds = float(line.split('compiled in', 1)[1].split()[0])
                native.xpu2_shaders_log().with_suffix('.took').write_text('%.0f' % seconds)
            except (OSError, ValueError, IndexError):
                pass
            return


def state():
    """The shaders' state in a few words: compiling (how far, from the compile's log), compiled, or not."""
    from moonray_modo import native
    log = native.xpu2_shaders_log()
    pid = log.with_suffix('.pid')
    try: lines = [l for l in log.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    except OSError: lines = []
    try: compiling = _running(pid.read_text().strip())
    except (OSError, ValueError): compiling = False
    last = lines[-1] if lines else ''
    if compiling:
        # How far, in percent of its compile tasks: 0% from the moment it starts (asked 2026-10-30: the state changes at
        # once, no message box).
        counts = _counts(lines)
        if counts is None or not counts[0] + counts[1]:
            return 'Compiling 0%'
        done, waiting, seconds = counts
        return 'Compiling %d%%' % int(100 * done / (done + waiting))
    _keep_took(lines)
    if compiled(): return 'Compiled'
    # Just that (asked 2026-10-30); why a compile failed is in its log and the Render Log.
    return 'Not compiled'


import lxifc

# The form's own notifier: told while the shaders compile, it asks the form to look again at the button and the
# shaders' state (how far along, then Compiled and Re-compile).
NOTIFIER = 'moonray.xpu2.shaders'


class ShadersNotifier(lxifc.Notifier):
    clients = {}
    def noti_Name(self): return NOTIFIER
    def noti_SetArgs(self, args): pass
    def noti_Args(self): return ''
    def noti_AddClient(self, event): self.clients[event.__peekobj__()] = event
    def noti_RemoveClient(self, event): self.clients.pop(event.__peekobj__(), None)


def _notify():
    for event in list(ShadersNotifier.clients.values()):
        try: lx.object.CommandEvent(event).Event(lx.symbol.fCMDNOTIFY_CHANGE_ALL)
        except (RuntimeError, LookupError): pass


_ticking = []


def _tick():
    """While the shaders compile, the form told every two seconds; once more as they are done, and the watch ended."""
    if not state().startswith('Compiling'):
        for timer in _ticking: timer.stop()
        del _ticking[:]
        from moonray_modo import native
        native._shaders_asked[1] = None    # whether they are compiled, asked again at once
    _notify()


def _watch():
    """Follow a compile under way (started here, or still running from before Modo was opened)."""
    if _ticking: return
    try:
        from PySide2 import QtCore
    except ImportError:
        return
    timer = QtCore.QTimer()
    timer.setInterval(2000)
    timer.timeout.connect(_tick)
    timer.start()
    _ticking.append(timer)


class CompileShaders(lxu.command.BasicCommand):
    """Compile XPU2's GPU programs into OptiX's disk cache in the background, once (the state says how long it has left),
    after which every XPU2 preview starts in seconds. Modo stays usable meanwhile."""
    def cmd_Flags(self): return lx.symbol.fCMD_UI
    # Greyed while a compile is under way: pressed again, it would only start another.
    def basic_Enable(self, msg): return not state().startswith('Compiling')
    # Named for what it does now: compiled, the shaders can only be compiled again (after a driver update).
    def basic_ButtonName(self): return 'Re-compile XPU2 Shaders' if state() == 'Compiled' else 'Compile XPU2 Shaders'
    def basic_Notifier(self, index): return (NOTIFIER, '') if index == 0 else None
    def basic_Execute(self, msg, flags):
        import subprocess
        from moonray_modo import native
        exe = native.xpu2_runtime() / 'moonray_xpu2.exe'
        if not exe.is_file():
            lx.out('MoonRay XPU2: no runtime-xpu2 in this kit'); return
        if state().startswith('Compiling'):
            lx.out('MoonRay XPU2: the shaders are compiling already'); return
        log = native.xpu2_shaders_log(); log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(native.xpu2_environment(native.default_runtime()))
        for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'): env.pop(key, None)
        with open(log, 'w', encoding='utf-8') as out:
            process = subprocess.Popen([str(exe), '--compile-shaders'], cwd=str(exe.parent), env=env, stdout=out,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       creationflags=0x00000008 | 0x08000000 | 0x00004000)  # detached, no window, below normal
        log.with_suffix('.pid').write_text(str(process.pid))
        native._shaders_asked[1] = None
        _watch()
        _notify()
        lx.out('MoonRay XPU2: compiling the shaders in the background (%s); see %s' % (how_long(), log))


class ShaderState(lxu.command.BasicCommand):
    """The state of XPU2's shaders (read only)."""
    def __init__(self):
        super().__init__()
        self.dyna_Add('value', lx.symbol.sTYPE_STRING)
        self.basic_SetFlags(0, lx.symbol.fCMDARG_QUERY)
    def cmd_Flags(self): return lx.symbol.fCMD_UI
    def basic_Enable(self, msg): return True
    def basic_Notifier(self, index): return (NOTIFIER, '') if index == 0 else None
    def cmd_Query(self, index, query):
        held = state()
        # A compile found under way (one still running from before Modo was opened) is followed from here.
        if held.startswith('Compiling'): _watch()
        lx.object.ValueArray(query).AddString(held)
    def basic_Execute(self, msg, flags): pass


lx.bless(ShadersNotifier, NOTIFIER)
lx.bless(CompileShaders, 'moonray.xpu2.compileShaders')
lx.bless(ShaderState, 'moonray.xpu2.shaderState')
'''

SHEET_ENTRY = '''      <list type="Control" val="sub MoonRayXPU2:sheet">
        <atom type="Label">XPU2</atom>
        <atom type="ShowLabel">1</atom>
        <atom type="StartCollapsed">0</atom>
        <atom type="Hash">MoonRayXPU2:sheet</atom>
      </list>
'''

SHEET = '''    <!-- XPU2: an asynchronous CPU and GPU renderer in MoonRay XPU's place for previews. -->
    <hash type="Sheet" key="MoonRayXPU2:sheet">
      <atom type="Label">XPU2</atom>
      <atom type="Layout">properties</atom>
      <list type="Control" val="cmd moonray.pref.xpu2 ?">
        <atom type="Label">GPU renderer</atom>
        <atom type="Tooltip">What renders the preview when its execution mode is XPU or Auto. MoonRay XPU, as before; or XPU2, the asynchronous renderer: the processor and the graphics card each take their own work from one grid. A scene XPU2 does not render yet (see the Render Log) is rendered by MoonRay XPU instead. A running preview restarts with the change</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.xpu2Devices ?">
        <atom type="Label">Devices</atom>
        <atom type="Tooltip">What XPU2 renders with: the processor and the graphics card together (the default), the graphics card only, or the processor only (no shaders to compile). A running preview restarts with the change</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.xpu2Upscale ?">
        <atom type="Label">Lower resolution while moving</atom>
        <atom type="Tooltip">While the view or the scene moves, realtime shows each new picture from a quarter of its pixels at first. On, OptiX's upscaling denoiser makes those pictures full size, sharper; off, they are enlarged as they are. A running preview restarts with the change</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.xpu2Chain ?">
        <atom type="Label">Adaptive passes on the graphics card</atom>
        <atom type="Tooltip">XPU2 makes each pass of an adaptive render on the graphics card: quicker, but the outlines of the work are not shown, since only the card knows them. Off, the passes are made on the processor between the card's work and the outlines show, for uniform and adaptive sampling alike</atom>
      </list>
      <list type="Control" val="cmd moonray.xpu2.shaderState ?">
        <atom type="Label">Shaders</atom>
        <atom type="Tooltip">Whether XPU2's GPU programs are compiled. Until they are, XPU2 renders nothing on the graphics card: the preview says so</atom>
      </list>
      <list type="Control" val="cmd moonray.xpu2.compileShaders">
        <atom type="Tooltip">Compiles XPU2's GPU programs into the graphics driver's shader cache, in the background, once; the state beside it shows how far it has come and about how long it has left. Every XPU2 preview after starts in seconds. Again after a graphics driver update</atom>
      </list>
    </hash>
'''

README = '''MoonRay for Modo {version} with XPU2

XPU2 is a renderer for MoonRay previews: MoonRay's path tracer and shader library ported to an
asynchronous design, where the processor and the graphics card each take their own work from one grid. It is not
affiliated with or endorsed by DreamWorks Animation.

It is the GPU renderer by default: System > Preferences > MoonRay > XPU2 > GPU renderer (MoonRay XPU or XPU2).
It renders the preview when the render's execution mode is XPU or Auto. Final renders, MoonRay LS and MoonLightIPR
are MoonRay's as before. XPU2 previews render the buffers the preview asks for (beauty, alpha and the AOVs) and are
denoised as MoonRay's are. Smart Render and Safe Denoising are MoonRay's: XPU2 is quick enough to need neither, and they
are greyed while it renders the preview.

First, once: Compile XPU2 Shaders on the same page (in the background; Modo stays usable, and the page shows how far
it has come and about how long is left). Until the shaders are compiled, XPU2 does not render: the preview says so.

A scene XPU2 does not render yet is rendered by MoonRay XPU instead; the Render Log says why ("MoonRay XPU2: ..."). Not
rendered yet, among others: deep output, continuing a saved render, a camera medium, normal-oriented curves, volumes bound to curves
or points, and scenes over 8 million pixels or 16 million triangles. What the graphics card's programs do not run yet
(points, geometry in motion, VDB, heterogeneous and emissive volumes) the processor renders in the same picture.

The preview shows the outlines of the work for uniform and adaptive sampling: each processor thread's 8-pixel tile,
and the grid cells the graphics card's queued work holds. With "Adaptive passes on the graphics card" on, adaptive
renders are quicker and show no outlines.

How XPU2 samples

XPU2 renders a preview in generations. Every pixel first gets a few samples, a first look. Then each generation
looks at where the picture is still noisy and sends the next samples there, and parts that are already clean stop.
Plain areas such as a wall or the sky finish quickly; hard ones such as glossy reflections, soft shadows and glass get
more. Only where the samples go changes, not how each one is worked out, so the picture settles to MoonRay's.

The settings are in Render Settings, in their own group, XPU2 Sampling:

- Generation Samples: What the render is worth, in samples a pixel on average (default: 16).
- Halting Rule: Time: stop after about the time a plain render of that many samples takes, spent where the noise is. Exact: every pixel gets exactly that many (default: Time).
- Halt at Noise: A part of the picture stops once it is this clean (lower is cleaner, and slower) (default: 10).
- First Look: Samples every pixel gets before the generations choose (default: 4).
- Most Samples: The most one pixel may get (default: 16 times the average).
- CPU Workers: The processor works beside the graphics card (default: On).
- Realtime Samples: While RT follows the scene, the share of Generation Samples it takes: lower settles sooner, noisier (default: 100%).

These settings always decide how XPU2 samples a preview. MoonRay's Sampling Mode and its adaptive settings are for
MoonRay renders: while XPU2 renders the preview they are greyed and show what XPU2 takes, and final renders use them as
always. A bake camera, or a camera with a pixel sample map, is sampled evenly with Pixel Samples, as MoonRay does.

Which devices XPU2 uses is in System > Preferences > MoonRay > XPU2 > Devices: the processor and the graphics card
together (the default), the graphics card only, or the processor only, which needs no shaders compiled.

The line under the picture says how it samples: XPU2 Gen. 16 ps avg (4-256), noise 10 is 16 samples a pixel on
average, 4 to 256 in any one pixel, each part stopping at noise 10. The MoonRay Terminal shows each generation as it
goes, with how much of the picture is still sampling.

runtime-xpu2/ is XPU2's own runtime (its preview session and the libraries it was built with); runtime/ is the
kit's, with the MoonLightIPR branch's denoiser (denoise.exe, libdenoiser.dll): Keep Fine Detail only over fur, hair and
curves, and "Denoiser cleans its guides first" in MoonRay Source Overrides.
'''


REPO = Path(r'C:\Users\Raphael Tobar\MoonRayForModo')
# The denoiser changes of the MoonLightIPR branch (commit f7a4c66), carried into the release kit this one is made from:
# Keep Fine Detail only over fur, hair and curves; "Denoiser cleans its guides first" (a MoonRay Source Override) and the
# runtime's denoise.exe and libdenoiser.dll built with it.
DENOISER_COMMIT = 'f7a4c66'
DENOISER_BUILD = REPO / 'build' / 'native-renderer-avx' / 'bin'


def denoiser(kit):
    import subprocess
    py = kit / 'python' / 'moonray_modo'
    for name in ('denoising.py', 'native.py'):  # the release's were the branch's before the change
        text = subprocess.check_output(['git', '-C', str(REPO), 'show', '%s:kit/MoonRayForModo/python/moonray_modo/%s' % (DENOISER_COMMIT, name)])
        (py / name).write_bytes(text)
    p = Patch(py / 'preferences.py')
    p.rep("""    'toon_albedo': (True, bool),""", """    'toon_albedo': (True, bool),
    # Open Image Denoise cleans the albedo and normal guides first and trusts them (native.denoise_clean_guides).
    'denoise_clean_guides': (True, bool),""")
    p.save()
    p = Patch(py / 'scene_settings.py')
    p.rep("""'Hair, fur and other things a pixel or so wide, which a denoiser smooths away, are let through from the render as it was. More keeps more of them and more of their grain; Off is the denoiser alone'""",
          """'Fur, hair and curves, which a denoiser smooths away, are let through from the render as it was, where the render has them; elsewhere the picture is the denoiser\\'s. More keeps more of them and more of their grain; Off is the denoiser alone'""")
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_preferences.py')
    p.rep("""lx.bless(switch('toon_albedo'),'moonray.pref.toonAlbedo')""", """lx.bless(switch('toon_albedo'),'moonray.pref.toonAlbedo')
lx.bless(switch('denoise_clean_guides'),'moonray.pref.denoiseCleanGuides')""")
    p.save()
    p = Patch(kit / 'preferences.cfg')
    p.rep("""and a denoised toon stays as noisy as it was rendered. It takes effect when the next render starts</atom>
      </list>""", """and a denoised toon stays as noisy as it was rendered. It takes effect when the next render starts</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.denoiseCleanGuides ?">
        <atom type="Label">Denoiser cleans its guides first</atom>
        <atom type="Tooltip">Open Image Denoise cleans the albedo and normal guides before the picture and then trusts them fully, so less of the render's noise is left on textured, bumpy and see-through surfaces. Off, as MoonRay itself does: the guides are used as rendered and trusted less. OptiX has no such mode and is unchanged. It takes effect at the next denoising</atom>
      </list>""")
    p.save()
    for name in ('denoise.exe', 'libdenoiser.dll'):
        shutil.copyfile(DENOISER_BUILD / name, kit / 'runtime' / name)


# XPU2's sampling generations in the Render item's MoonRay settings (Sampling, after Target Error): kept with the scene
# as 'xpu2_generations' and written into a preview's scene as UserData("/modo/xpu2/generations"), which XPU2's plan
# builder reads (cpu_scene.h) and MoonRay leaves alone. Defaults: 16 samples a pixel on average and a noise target of
# 10 (his choice, 2026-10-25); the rest are what XPU2 does without them.
GENERATION_FIELDS = [
    ('xpu2_generations', 'on', 'bool', 'XPU2 Sampling Generations',
     'XPU2 previews sample in generations: each pass goes where the picture is still noisy, and simple areas stop early. '
     'Off: XPU2 samples as MoonRay does, uniform or adaptive. MoonRay renders are unchanged', None, None, None),
    ('xpu2_samples', 'samples', 'int', 'Generation Samples',
     'What the render is worth, in samples a pixel on average: the generations share them out where the picture is '
     'still noisy. 0: Pixel Samples squared. XPU2 previews always sample this way: MoonRay\'s Sampling Mode and adaptive '
     'settings are for MoonRay renders', 0, 4096, None),
    ('xpu2_spend', 'spend', 'choice', 'Halting Rule',
     'Time = Gen. Samples: the render stops after about the time a plain render of Generation Samples would take, that '
     'time spent where the noise is. Exact = Gen. Samples: every pixel gets exactly Generation Samples, however long '
     'that takes', None, None,
     [('Time = Gen. Samples', 'time'), ('Exact = Gen. Samples', 'samples')]),
    ('xpu2_noise', 'noise', 'float', 'Halt at Noise',
     'A part of the picture stops early once it is this clean, in Target Error\'s units (lower is cleaner, and slower). '
     '0: none, every pixel gets its share. MoonRay\'s Target Error is for MoonRay renders', 0.0, 1000.0, None),
    ('xpu2_seed', 'seed', 'int', 'First Look',
     'Samples every pixel gets first, before the generations choose where the rest go. 0: an eighth of Generation '
     'Samples, 16 at least', 0, 16384, None),
    ('xpu2_most', 'most', 'int', 'Most Samples',
     'The most samples one pixel may get. 0: 16 times the average', 0, 16384, None),
    ('xpu2_cpu', 'cpu', 'bool', 'CPU Workers',
     'The processor\'s workers take part in the generations beside the graphics card. Off: the graphics card takes '
     'every sample. Which devices XPU2 uses at all: Preferences > MoonRay > XPU2 > Devices', None, None, None),
    ('xpu2_realtime', 'realtime', 'int', 'Realtime Samples',
     'While realtime follows the scene (RT), the share of Generation Samples XPU2 takes, in percent: lower settles '
     'sooner and noisier. Previews started with Render and final renders take them all', 1, 100, None),
]


def generations(kit):
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'scene_settings.py')
    p.rep("""    'denoising': {'engine': 'off', 'preview': True, 'final': True, 'safe': False, 'smart': 'off', 'detail': 'some'},""",
          """    'denoising': {'engine': 'off', 'preview': True, 'final': True, 'safe': False, 'smart': 'off', 'detail': 'some'},
    # XPU2's sampling generations: 16 samples a pixel on average and a noise target of 10 by
    # default; the rest are what XPU2 does without them.
    'xpu2_generations': {'on': True, 'samples': 16, 'spend': 'time', 'noise': 10.0, 'seed': 4, 'most': 0, 'cpu': True, 'realtime': 100},""")
    p.rep("""key in ('recovery', 'denoising', 'deep', 'exr')""", """key in ('recovery', 'denoising', 'deep', 'exr', 'xpu2_generations')""")
    fields = ''.join("""        field(%r, ('xpu2_generations', %r), %r, %r, %r%s),
""" % (key, member, kind, label, tip,
       (', %r, %r' % (low, high) if low is not None else '') + (', choices=%r' % (choices,) if choices else ''))
                     for key, member, kind, label, tip, low, high, choices in GENERATION_FIELDS if member != 'on')
    # A group of their own, XPU2 Sampling, after MoonRay's Sampling (asked 2026-10-28).
    p.rep("""    ('Bounces', False, [
""", """    ('XPU2 Sampling', False, [
""" + fields + """    ]),
    ('Bounces', False, [
""")
    p.save()
    p = Patch(py / 'scene_setup.py')
    p.rep("""    scene['execution_mode'] = values['execution_mode']""", """    scene['execution_mode'] = values['execution_mode']
    # XPU2's sampling generations, written into a preview's scene for XPU2 (rdla.py).
    scene['xpu2_generations'] = values['xpu2_generations']""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("""    if safe and safe.get('data'):""", """    # XPU2's sampling generations: its render settings in a preview, as a UserData that XPU2
    # reads (cpu_scene.h: 1, on, samples, spend exactly, noise target, first generation, most samples, CPU workers,
    # realtime: the preview following the scene with RT on, panel.py) and MoonRay leaves alone.
    generation=scene.get('xpu2_generations')
    if generation and not output_file:
        values=[1,1 if generation.get('on',True) else 0,int(generation.get('samples',0)),1 if generation.get('spend')=='samples' else 0,
                float(generation.get('noise',0.0)),int(generation.get('seed',0)),int(generation.get('most',0)),1 if generation.get('cpu',True) else 0,
                1 if scene.get('_realtime') else 0]
        lines += ['UserData("/modo/xpu2/generations") {','  ["float_key"] = "xpu2_generations",',
                  '  ["float_values_0"] = {%s},' % ', '.join('%g' % v for v in values),'}']
    if safe and safe.get('data'):""")
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_render_settings.py')
    p.rep("""                if entry['key'] in NEEDS_DENOISER:""", """                # XPU2's sampling generations' own settings, greyed with generations off.
                if entry['key'] in %r:
                    try:return bool(scene_settings.complete(properties.scene_settings())['xpu2_generations'].get('on',True))
                    except Exception:return True
                if entry['key'] in NEEDS_DENOISER:""" % (tuple(f[0] for f in GENERATION_FIELDS[1:]),))
    # And the other way: with generations on, MoonRay's own sampling settings are greyed (XPU2 previews use the XPU2
    # ones; final renders, MoonRay's, still use them).
    p.rep("""            def basic_Enable(self,msg):
""", """            def basic_Enable(self,msg):
                # XPU2's sampling generations: on, MoonRay's own sampling settings are greyed
                # (XPU2 previews use the XPU2 ones; final renders, MoonRay's, still use them).
                if entry['key'] in ('sampling_mode','samples','min_adaptive_samples','max_adaptive_samples','target_adaptive_error'):
                    try:
                        if scene_settings.complete(properties.scene_settings())['xpu2_generations'].get('on',True):return False
                    except Exception:pass
""")
    # Generation Samples left at 0 shows what XPU2 takes then: a uniform scene's Pixel Samples, squared.
    p.rep("""                value=scene_settings.get(scene_settings.complete(properties.scene_settings()),entry)
""", """                value=scene_settings.get(scene_settings.complete(properties.scene_settings()),entry)
                # Generation Samples left at 0: what XPU2 takes then, a uniform scene's Pixel Samples
                # squared, Pixel Samples itself being greyed while generations are on.
                if entry['key']=='xpu2_samples' and not value:
                    try:
                        held=scene_settings.complete(properties.scene_settings())
                        value=int(held.get('samples',0))**2
                    except Exception:pass
""")
    p.save()
    from xml.sax.saxutils import escape
    controls = ''.join("""      <list type="Control" val="cmd moonray.render.%s ?">
        <atom type="Label">%s</atom>
        <atom type="Tooltip">%s</atom>
      </list>
""" % (key, escape(label), escape(tip)) for key, member, _, label, tip, _, _, _ in GENERATION_FIELDS if member != 'on')
    p = Patch(kit / 'render_settings.cfg')
    p.rep("""        <atom type="Hash">MoonRayRender_1:sheet</atom>
      </list>
""", """        <atom type="Hash">MoonRayRender_1:sheet</atom>
      </list>
      <list type="Control" val="sub MoonRayRenderXPU2:sheet">
        <atom type="Label">XPU2 Sampling</atom>
        <atom type="ShowLabel">1</atom>
        <atom type="StartCollapsed">0</atom>
        <atom type="Hash">MoonRayRenderXPU2:sheet</atom>
      </list>
""")
    p.rep("""    <hash type="Sheet" key="MoonRayRender_2:sheet">""", """    <hash type="Sheet" key="MoonRayRenderXPU2:sheet">
      <atom type="Label">XPU2 Sampling</atom>
      <atom type="Layout">properties</atom>
""" + controls + """    </hash>
    <hash type="Sheet" key="MoonRayRender_2:sheet">""")
    p.save()


# Realtime in the preview window: an RT switch of its own left of the engine (MoonLightIPR, or MoonRay where XPU2
# renders the preview), the Render button rendering once; the one greyed while the other is in use. A preview that
# follows the scene marks its scene _realtime, which rdla.py writes into the generations' UserData for XPU2 (its
# realtime frames, cpu_scene.h).
def realtime(kit):
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""        self.ipr_mode.setChecked(self._live())
        row.addWidget(self.preview_engine)
""", """        # Realtime is off as the window opens: its switch, RT, turns it on.
        self.ipr_mode.setChecked(False)
        # Realtime: a switch of its own left of the engine, the Render button rendering once.
        # On, the preview follows the scene as it is edited: MoonLightIPR, or MoonRay where XPU2 renders the preview.
        # Each is greyed while the other is in use: Render while RT is on, RT while a render runs or stops.
        self.rt=QtWidgets.QPushButton('RT Off');self.rt.setCheckable(True)
        self.rt.setStyleSheet('QPushButton:checked{background-color:#f49c1c;color:#1a1a1a;border:1px solid #ffb84d;border-radius:3px;font-weight:bold;}')
        self.rt.toggled.connect(self._rt_toggled)
        row.addWidget(self.rt)
        row.addWidget(self.preview_engine)
""")
    p.rep("""    def _live(self):
        \"\"\"Whether the engine chosen is the realtime one, which follows the scene as it is edited.\"\"\"""",
          """    def _rt_capable(self):
        \"\"\"Whether the engine chosen can follow the scene in realtime: MoonLightIPR, or MoonRay
        where XPU2 renders the preview (Preferences > MoonRay > XPU2).\"\"\"
        engine=self.preview_engine.currentData()
        if engine=='moonlightipr':return True
        if engine!='moonray':return False
        try:
            from . import native
            return bool(native.xpu2())
        except Exception:return False

    def _rt_toggled(self,on):
        \"\"\"The RT switch: realtime on, the preview follows the scene from now; off, realtime ends, its picture kept, and
        the Render button is free again. Not while a render runs or stops: Stop it first.\"\"\"
        if getattr(self,'_rt_syncing',False):return
        if on and (not self._rt_capable() or self._output_busy() or self._rt_blocked()):
            if not self._rt_capable():
                self.status.setText('Realtime is MoonLightIPR\\'s, or MoonRay\\'s with XPU2 chosen in Preferences > MoonRay > XPU2.')
            self._sync_rt();return
        if on:
            self.ipr_mode.setChecked(True)
            self.start_render()
        else:
            self.ipr_mode.setChecked(False)
            # As MoonLightIPR rests when realtime goes off: the render stops where it stands, its picture kept.
            if self._rendering and not getattr(self,'_stopping',False) and not self._output_busy():self.stop()
        self._sync_rt()
        self._set_rendering(self._rendering)

    def _rt_blocked(self):
        \"\"\"Whether a render other than realtime's runs, or one stops: RT is greyed until it is done.\"\"\"
        return bool((self._rendering and not self.ipr_mode.isChecked()) or getattr(self,'_stopping',False))

    def _sync_rt(self):
        \"\"\"Keep the RT switch saying whether realtime is on, and whether it can be with the engine chosen.\"\"\"
        if not hasattr(self,'rt') or not hasattr(self,'renderer'):return
        self._rt_syncing=True
        try:
            on=bool(self.ipr_mode.isChecked())
            self.rt.setChecked(on);self.rt.setText('RT On' if on else 'RT Off')
            self.rt.setEnabled(self._rt_capable() and not self._output_busy() and not self._rt_blocked())
            self.rt.setToolTip('Realtime is on: the preview follows the scene as you edit it. Press to switch it off.' if on else
                               'A render is under way. Stop it, or let it finish, to switch realtime on.' if self._rt_blocked() else
                               'Realtime is off. Press to have the preview follow the scene as you edit it (MoonLightIPR, or '
                               'MoonRay with XPU2).' if self._rt_capable() else
                               'Realtime is MoonLightIPR\\'s, or MoonRay\\'s with XPU2 chosen in Preferences > MoonRay > XPU2.')
        finally:self._rt_syncing=False

    def _live(self):
        \"\"\"Whether the engine chosen is the realtime one, which follows the scene as it is edited.\"\"\"""")
    # A change of engine leaves realtime off; its switch turns it on.
    p.rep("""            if self.ipr_mode.isChecked()!=live:self.ipr_mode.setChecked(live)""",
          """            if self.ipr_mode.isChecked():self.ipr_mode.setChecked(False)""")
    p.rep("""        self.timer.setInterval(60 if self._engine()=='moonlightipr' else 150)""",
          """        self.timer.setInterval(60 if self._engine()=='moonlightipr' or self._rt_capable() else 150)""")
    # Render renders once and Stop stops, realtime with it.
    p.rep("""    def _render_or_stop(self):
        if self._rendering or self._following:self.stop()
        else:self.start_render()""", """    def _render_or_stop(self):
        if self._rendering or self._following:
            if self.ipr_mode.isChecked():self.ipr_mode.setChecked(False)
            self.stop();self._sync_rt()
        else:self.start_render()""")
    # The Render button is no longer realtime's: it says Render or Stop whatever the engine.
    p.rep("""        if self._live():
            # A realtime engine is switched on and off, not started and stopped.""", """        self._sync_rt()
        if self.ipr_mode.isChecked():
            # Realtime is on: this button is greyed, and RT Off gives it back.
            self.start.setChecked(False);self.start.setCheckable(False)
            self.start.setEnabled(False);self.start.setText('Render')
            self.start.setToolTip('Realtime is on. Switch RT off to render once.')
            return
        if self._engine()=='moonlightipr' and not self._rendering:
            # MoonLightIPR renders in realtime alone: RT turns it on, and this button is greyed.
            self.start.setChecked(False);self.start.setCheckable(False)
            self.start.setEnabled(False);self.start.setText('Render')
            self.start.setToolTip('MoonLightIPR renders in realtime: switch RT on.')
            return
        if False:
            # (Realtime has its switch, RT, left of the engine; this button renders once.)
            # A realtime engine is switched on and off, not started and stopped.""")
    # A preview that follows the scene with MoonRay is XPU2's realtime: its scene says so.
    p.rep("""        if engine=='moonray':scene=dict(scene,_pixel_samples=int(grid)**2)
        self.renderer.submit(scene, self.runtime_path(), width, height, grid,""",
          """        if engine=='moonray':scene=dict(scene,_pixel_samples=int(grid)**2)
        if engine=='moonray' and self._following and not output:scene=dict(scene,_realtime=True)
        self.renderer.submit(scene, self.runtime_path(), width, height, grid,""")
    # Realtime renders at the realtime preview size (Preferences), as MoonLightIPR does; Render renders the picture.
    p.rep("""        width = scene_w if final or not (self._live() or self._low()) else min(scene_w,size)""",
          """        width = scene_w if final or not (self._live() or self._low() or (self.ipr_mode.isChecked() and self._rt_capable())) else min(scene_w,size)""")
    # A region is rendered with Render, whatever the engine; the button says Stopping while a render stops.
    p.text = p.text.replace("""%('RT Off' if self._live() else 'Render')""", """%'Render'""")
    p.rep("""        self._stopping=True
        if not self._live():
            self.start.setEnabled(False);self.start.setText('Stopping')""", """        self._stopping=True
        self._sync_rt()
        if True:
            self.start.setEnabled(False);self.start.setText('Stopping')""")
    # XPU2's realtime denoises its own pictures where Denoised Beauty is chosen (xpu2_session.cc, RealtimeDenoiser): the
    # buffer stays chosen as realtime comes on.
    p.rep("""            if self.buffer.currentData()=='denoised_beauty' and self._engine()!='moonlightipr':""",
          """            if self.buffer.currentData()=='denoised_beauty' and self._engine()!='moonlightipr' and not (self.ipr_mode.isChecked() and self._rt_capable()):""")
    p.save()
    # The buffer chosen goes to XPU2 as its view, Denoised Beauty among them; MoonRay kept loaded is given the beauty for it.
    p = Patch(kit / 'python' / 'moonray_modo' / 'render.py')
    p.rep("""        if key=='denoised_beauty':return # Post-process result, not a native render output.""",
          """        if key=='denoised_beauty' and not native.xpu2():return # Post-process result, not a native render output.""")
    p.save()
    p = Patch(kit / 'python' / 'moonray_modo' / 'persistent.py')
    p.rep("""        self.view=key
        staged=self.root/'view.tmp';staged.write_text(key,encoding='ascii');staged.replace(self.root/'view.txt')""",
          """        self.view=key
        # XPU2's realtime denoises its pictures where Denoised Beauty is chosen; MoonRay kept loaded hands over its beauty.
        if key=='denoised_beauty' and not getattr(self,'xpu2_active',False):key='beauty'
        staged=self.root/'view.tmp';staged.write_text(key,encoding='ascii');staged.replace(self.root/'view.txt')""")
    p.save()
    # A realtime frame's denoised picture, handed over in memory as it went, stays the frame's until the kit's denoiser
    # makes one from its files (or for good, where none is made).
    p = Patch(kit / 'python' / 'moonray_modo' / 'buffer_cache.py')
    p.rep("""            self.frame=frame;self.progressive=None;self.progressive_frames.clear();self.progressive_scene=None""",
          """            carried=self.progressive_frames.get('denoised_beauty')
            if carried is not None and self.progressive_scene is snapshot and 'denoised_beauty' not in copied and 'denoised_beauty' in carried.get('linear',{}):
                frame['linear']={'denoised_beauty':carried['linear']['denoised_beauty']}
            self.frame=frame;self.progressive=None;self.progressive_frames.clear();self.progressive_scene=None""")
    p.save()


# The denoised beauty and the Denoiser in Render Settings agree: OptiX unless the scene chooses otherwise, and with the
# Denoiser Off no Denoised Beauty in the preview's list, whatever the engine (MoonLightIPR's and XPU2's realtime
# denoising included, which go with the list).
def denoised_buffer(kit):
    p = Patch(kit / 'python' / 'moonray_modo' / 'scene_settings.py')
    p.rep("""    'denoising': {'engine': 'off', 'preview': True,""",
          """    # OptiX unless the scene chooses: the graphics card's denoiser, the quickest.
    'denoising': {'engine': 'optix', 'preview': True,""")
    p.save()
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""        for row in range(1,self.buffer.count()):
            self.buffer.view().setRowHidden(row,low or (only and not (denoises and self.buffer.itemData(row)=='denoised_beauty')))""",
          """        # The Denoiser set to Off in Render Settings is off everywhere: the denoised beauty is not in the list.
        on=self._denoiser_on()
        for row in range(1,self.buffer.count()):
            denoised=self.buffer.itemData(row)=='denoised_beauty'
            self.buffer.view().setRowHidden(row,low or (denoised and not on) or (only and not (denoises and denoised)))
        if not on and self.buffer.currentData()=='denoised_beauty':self.buffer.setCurrentIndex(self.buffer.findData('beauty'))""")
    p.rep("""    def _moonlightipr_denoises(self):
        \"\"\"Whether MoonLightIPR's list has the denoised beauty in it: always. It denoises with the graphics card's own
        denoiser, whatever the scene's denoiser is set to, when that buffer is the one chosen.\"\"\"
        return True""", """    def _denoiser_on(self):
        \"\"\"Whether Render Settings > Denoiser names a denoiser. Off is off everywhere: no denoised beauty in the list.\"\"\"
        try:return scene_settings.complete(self._settings_values_stored())['denoising']['engine']!='off'
        except Exception:return True

    def _moonlightipr_denoises(self):
        \"\"\"Whether MoonLightIPR's list has the denoised beauty in it: whenever the scene's Denoiser is not Off. It
        denoises with the graphics card's own denoiser, whichever the scene names, when that buffer is the one chosen.\"\"\"
        return self._denoiser_on()""")
    p.rep("""        # MoonRay LS is a quick look and is not denoised. MoonLightIPR denoises as it goes.
        if self._low() and self.buffer.currentData()=='denoised_beauty':""",
          """        # MoonRay LS is a quick look and is not denoised. MoonLightIPR denoises as it goes. With the Denoiser Off
        # there is no denoised beauty.
        if (self._low() or not self._denoiser_on()) and self.buffer.currentData()=='denoised_beauty':""")
    p.rep("""        engine=self._preview_denoiser()
        self._buffer_picked=None
        if self._engine()=='moonlightipr':""", """        engine=self._preview_denoiser()
        self._buffer_picked=None
        # Denoised Beauty is in the list while a denoiser is chosen.
        self._limit_buffers()
        if self._engine()=='moonlightipr':""")
    # XPU2's realtime denoises its own pictures: a denoiser chosen with realtime on shows them.
    p.rep("""        self.renderer.buffers.denoiser.request(engine)
        self.buffer.setCurrentIndex(self.buffer.findData('beauty' if engine=='off' else 'denoised_beauty'))""",
          """        self.renderer.buffers.denoiser.request(engine)
        realtime=self.ipr_mode.isChecked() and self._rt_capable() and self._denoiser_on()
        self.buffer.setCurrentIndex(self.buffer.findData('beauty' if engine=='off' and not realtime else 'denoised_beauty'))""")
    p.save()


# The preview's engines follow Preferences > MoonRay > XPU2: with XPU2 they are MoonRay and MoonLightIPR, with XPU
# MoonRay, MoonRay LS and MoonLightIPR. Kept so as the window is open (its poll), MoonRay LS chosen, or the window's
# opening engine, becoming MoonRay as XPU2 comes on.
def engines(kit):
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""                                       'Output renders always use MoonRay.')
""", """                                       'Output renders always use MoonRay.')
        self._sync_engines()
""")
    p.rep("""        \"\"\"Follow the scene's settings, which are edited in Modo's properties, not here.\"\"\"
        if self.disposed:return
""", """        \"\"\"Follow the scene's settings, which are edited in Modo's properties, not here.\"\"\"
        if self.disposed:return
        # And the engines Preferences > MoonRay > XPU2 allows.
        self._sync_engines()
""")
    p.rep("""    def _rt_capable(self):""", """    def _sync_engines(self):
        \"\"\"The engines in the list, as Preferences > MoonRay > XPU2 has it: with XPU2, MoonRay and
        MoonLightIPR; with XPU, MoonRay LS between them. MoonRay LS chosen as XPU2 comes on becomes MoonRay, an engine
        changed as any other; the list otherwise changes without changing the engine chosen.\"\"\"
        try:
            from . import native
            xpu2=bool(native.xpu2())
        except Exception:xpu2=False
        if xpu2==getattr(self,'_engines_xpu2',None):return
        self._engines_xpu2=xpu2
        if not hasattr(self,'_engine_tip'):self._engine_tip=self.preview_engine.toolTip()
        row=self.preview_engine.findData('moonray_ls')
        if xpu2 and row>=0:
            if self.preview_engine.currentData()=='moonray_ls':self.preview_engine.setCurrentIndex(self.preview_engine.findData('moonray'))
            blocker=QtCore.QSignalBlocker(self.preview_engine)
            self.preview_engine.removeItem(self.preview_engine.findData('moonray_ls'))
            del blocker
        elif not xpu2 and row<0:
            blocker=QtCore.QSignalBlocker(self.preview_engine)
            self.preview_engine.insertItem(self.preview_engine.findData('moonray')+1,'MoonRay LS','moonray_ls')
            del blocker
        self.preview_engine.setToolTip('\\n'.join(line for line in self._engine_tip.split('\\n') if not (xpu2 and line.startswith('MoonRay LS'))))
        # RT is MoonRay's with XPU2 alone.
        if hasattr(self,'_sync_rt'):self._sync_rt()

    def _rt_capable(self):""")
    p.save()


REPO_KIT = Path(r'C:\Users\Raphael Tobar\MoonRayForModo\kit\MoonRayForModo\python\moonray_modo')


def repo_fixes(kit):
    """Two of the MoonLightIPR branch's kit commits, taken from the repo: a Shader Tree displacement whose height is black
    everywhere is left out (8f1ae7f, graph.zero_height), and the preview window's memory meter (a998b25)."""
    py = kit / 'python' / 'moonray_modo'
    source = (REPO_KIT / 'graph.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    start = source.index('def zero_height(binding, lines):')
    end = source.index('    return bodies\n', start) + len('    return bodies\n')
    p = Patch(py / 'graph.py')
    p.rep('DISPLACEMENT = None\n', 'DISPLACEMENT = None\n\n\n' + source[start:end])
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("        elif shader_displacement.get(tag) and shader_displacement[tag][1]:",
          "        elif shader_displacement.get(tag) and shader_displacement[tag][1] and not graph_module.zero_height(shader_displacement[tag][0], lines):")
    p.save()
    shutil.copy2(REPO_KIT / 'memory_meter.py', py / 'memory_meter.py')
    p = Patch(py / 'panel.py')
    p.rep("""        self.image_info=QtWidgets.QLabel('');self.image_info.setEnabled(False)
        right.addStretch(1)
        right.addWidget(self.image_info)""", """        self.image_info=QtWidgets.QLabel('');self.image_info.setEnabled(False)
        right.addStretch(1)
        # The memory the renderer, the machine and the graphics card have, beside the picture's details.
        from .memory_meter import MemoryMeter
        self.memory_meter=MemoryMeter(self._renderer_processes)
        right.addWidget(self.memory_meter)
        right.addWidget(self.image_info)""")
    p.rep("""    def _focus_armed(self,armed):""", """    def _renderer_processes(self):
        \"\"\"The ids of the renderer's processes running now (a final render, the preview session, MoonLightIPR's), for the
        memory meter.\"\"\"
        renderer=getattr(self,'renderer',None)
        if renderer is None:return []
        found=[]
        for holder in (renderer,getattr(renderer,'session',None),getattr(renderer,'moonlightipr',None)):
            process=getattr(holder,'process',None) if holder is not None else None
            try:
                if process is not None and process.state()!=QtCore.QProcess.NotRunning:found.append(int(process.processId()))
            except RuntimeError:pass  # its Qt object already gone
        return found

    def _focus_armed(self,armed):""")
    p.save()


def displacement_choice(kit):
    """The main kit's displacement tessellation choice (MoonLightIPR branch, uncommitted 2026-10-26): Follow Modo (the
    render item's Displacement Rate as adaptive_error), a fixed dicing camera held at the Dicing Frame, or uniform."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'options.py')
    p.rep("""    'enable_displacement': (1, 0, 1, 'Displacement'),""", """    'enable_displacement': (1, 0, 1, 'Displacement'),
    # How a displaced polygon mesh is tessellated where the object says nothing (the kit's own, OWN): 1 follows Modo's
    # Displacement Rate from the render camera; 2 the same from a dicing camera held at the Dicing Frame; 0 uniformly.
    'displacement_tessellation': (1, 0, 2, 'Displacement tessellation'),
    'displacement_adaptive_error': (0.0, 0.0, 64.0, 'Adaptive error override (pixels; 0 = Modo\\'s Displacement Rate)'),
    'dicing_frame': (-1.0, -1000000.0, 1000000.0, 'Dicing frame (-1 = the first frame)'),""")
    p.rep("""OWN = ('batch_tile_order', 'bucket_size', 'important_first')""",
          """OWN = ('batch_tile_order', 'bucket_size', 'important_first', 'displacement_adaptive_error', 'displacement_tessellation', 'dicing_frame')""")
    p.rep("""ENUMS['bucket_size'] = """, """ENUMS['displacement_tessellation'] = [('Uniform (mesh resolution)',0),('Follow Modo (Displacement Rate)',1),('Fixed dicing camera',2)]
ENUMS['bucket_size'] = """)
    p.save()
    p = Patch(py / 'scene_settings.py')
    line = next(l for l in p.text.split('\n') if l.startswith("        render_field('enable_displacement', 'Displacement',"))
    p.rep(line + '\n', line + '\n' + """        render_field('displacement_tessellation', 'Displacement Tessellation', 'How subdivision surfaces and displaced polygon meshes are cut into triangles. Follow Modo: finely where they are large on screen and coarsely where small, to Modo''s Displacement Rate (Render item), as Modo does: far less geometry and a quicker start. Fixed dicing camera: the same, worked out from the Dicing Camera (or the render camera) held at the Dicing Frame, so the triangles stay put while the camera moves (no shimmer in an animation). Uniform: by each mesh''s resolution (a subdivision surface''s level), the same whatever the camera. An object''s own Subdivision Screen Error wins; instances stay uniform'),
        render_field('displacement_adaptive_error', 'Adaptive Error Override', 'Pixels of screen error in place of Modo''s Displacement Rate (lower is finer). 0: Modo''s'),
        render_field('dicing_frame', 'Dicing Frame', 'The frame the fixed dicing camera is held at. -1: the scene''s first frame'),
""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("""            mesh=dict(mesh,adaptive_error=settings['adaptive_error'])""",
          """            # The object's own screen error; else, as the scene's Displacement Tessellation says, Modo's Displacement Rate
            # (or the Adaptive Error Override) or none (uniform); an instancer's reference stays uniform.
            if settings['adaptive_error']:error=settings['adaptive_error']
            elif 'instances' in mesh or held_render['displacement_tessellation']==0:error=0.0
            else:error=held_render['displacement_adaptive_error'] or float(scene.get('displacement_rate',1.0))
            mesh=dict(mesh,adaptive_error=error)""")
    p.save()
    p = Patch(py / 'host.py')
    p.rep("""def world_matrix(item):
    rows = lx.object.Matrix(item.channel('worldMatrix').get()).Get4()""", """def world_matrix(item, time=None):
    rows = lx.object.Matrix(item.channel('worldMatrix').get() if time is None else item.channel('worldMatrix').get(time)).Get4()""")
    p.rep("""def dicing_view(scene, identity, width, height, pixel_aspect, warnings):""",
          """def dicing_view(scene, identity, width, height, pixel_aspect, warnings, time=None):""")
    p.rep("""    if item.id == scene.renderCamera.id:
        return None""", """    if item.id == scene.renderCamera.id and time is None:
        return None""")
    p.rep("""    return {'matrix': world_matrix(item), 'identity': item.id""", """    return {'matrix': world_matrix(item, time), 'identity': item.id""")
    start = p.text.index("    # Another camera for MoonRay's adaptive tessellation, where one is chosen.")
    end = p.text.index("            result['dicing_camera'] = view", start) + len("            result['dicing_camera'] = view")
    p.text = p.text[:start] + """    # Modo's Displacement Rate (pixels), which displaced meshes are tessellated to where the scene follows Modo.
    result['displacement_rate'] = float(channel(render, 'dispRate', 1.0))
    # Another camera for MoonRay's adaptive tessellation: the one chosen; with a fixed dicing camera, it (or the render
    # camera) held at the Dicing Frame.
    stored = properties.scene_settings()
    chosen = stored.get('dicing_camera') or ''
    rendering = stored.get('render', {}) if isinstance(stored.get('render'), dict) else {}
    if int(rendering.get('displacement_tessellation', 1)) == 2:
        frame = float(rendering.get('dicing_frame', -1.0))
        if frame < 0:
            frame = float(channel(render, 'first', 0))
        fps = float(scene.fps) if getattr(scene, 'fps', 0) else 24.0
        view = dicing_view(scene, chosen or scene.renderCamera.id, width, height, pixel_aspect, warnings, time=frame / fps)
        if view:
            result['dicing_camera'] = view
    elif chosen:
        view = dicing_view(scene, chosen, width, height, pixel_aspect, warnings)
        if view:
            result['dicing_camera'] = view""" + p.text[end:]
    p.save()


def geometry_rows(kit):
    """The main kit's geometry kept as Modo hands it over (MoonLightIPR branch, uncommitted 2026-10-26): rows.py, the
    adapter's rows read as the lists they stand for and written to the binary scene as they came, and the adapter's
    arithmetic on them (MR_uv_transform, MR_zigzag4). evaluated, scene_digest, coordinates and rdlb are the base kit's
    files as they are in the repo, taken whole; rdla's few lines patched; the adapter built by tools/build_modo_bridge.py
    into build/modo-bridge-rows."""
    py = kit / 'python' / 'moonray_modo'
    for name in ('rows.py', 'evaluated.py', 'scene_digest.py', 'coordinates.py', 'rdlb.py'):
        shutil.copy2(REPO_KIT / name, py / name)
    shutil.copy2(REPO_KIT.parents[3] / 'build' / 'modo-bridge-rows' / 'MoonRayGeometry.lx', kit / 'bin' / 'MoonRayGeometry.lx')
    p = Patch(py / 'rdla.py')
    p.rep("from . import rdlb, primitive_attributes\n", "from . import rdlb, primitive_attributes, rows\n")
    p.rep("""    if kind == 'vec3' or kind == 'vec2':
        size = 3 if kind == 'vec3' else 2
        encode, value_type = (lambda: rdlb.float_list(source, size)), (rdlb.VEC3F_VECTOR if size == 3 else rdlb.VEC2F_VECTOR)
    elif kind == 'indices':
        encode, value_type = (lambda: rdlb.int_list(numbers('l', chain.from_iterable(source)))), rdlb.INT_VECTOR
    elif kind == 'counts':
        encode, value_type = (lambda: rdlb.int_list(numbers('l', map(len, source)))), rdlb.INT_VECTOR
    elif kind == 'rgb':
        encode, value_type = (lambda: rdlb.float_list(source, 3)), rdlb.RGB_VECTOR""", """    # Lists as Modo handed them over (rows.Rows, rows.Faces) go as the numbers they came in: the same bytes.
    if kind == 'vec3' or kind == 'vec2':
        size = 3 if kind == 'vec3' else 2
        held = rows.numbers(source, size)
        encode, value_type = (lambda: rdlb.float_list(source if held is None else held, size)), (rdlb.VEC3F_VECTOR if size == 3 else rdlb.VEC2F_VECTOR)
    elif kind == 'indices':
        held = rows.corners(source)
        encode, value_type = (lambda: rdlb.int_list(numbers('l', chain.from_iterable(source)) if held is None else held)), rdlb.INT_VECTOR
    elif kind == 'counts':
        held = rows.counts(source)
        encode, value_type = (lambda: rdlb.int_list(numbers('l', map(len, source)) if held is None else held)), rdlb.INT_VECTOR
    elif kind == 'rgb':
        held = rows.numbers(source, 3)
        encode, value_type = (lambda: rdlb.float_list(source if held is None else held, 3)), rdlb.RGB_VECTOR""")
    p.rep("""            # face by face, which says which mesh is wrong.
            if min(map(len, faces)) >= 3:""", """            # face by face, which says which mesh is wrong. Faces as Modo handed them over (rows.Faces) are whole
            # numbers by their kind, their corners and counts looked over as they came.
            held = rows.corners(faces)
            if held is not None and min(rows.counts(faces)) >= 3 and (not len(held) or (min(held) >= 0 and max(held) < len(vertices))):
                return ''
            if min(map(len, faces)) >= 3:""")
    p.rep("""            if len(values) != sum(map(len, faces)):
                raise ValueError('Named UV set must match polygon corners: '+uv_name)""", """            if len(values) != rows.corner_count(faces):
                raise ValueError('Named UV set must match polygon corners: '+uv_name)""")
    p.rep("""('corner_attributes', 6, sum(map(len, faces)))""", """('corner_attributes', 6, rows.corner_count(faces))""")
    p.rep("""                if len(values) != sum(map(len, faces)):
                    raise ValueError('Face-varying %s must match the corner count' % source)""", """                if len(values) != rows.corner_count(faces):
                    raise ValueError('Face-varying %s must match the corner count' % source)""")
    p.save()


def binary_memory(kit):
    """XPU2 is handed the geometry's binary scene in shared memory (modo-memory-rdlb, xpu2_session.cc's readInto), not as a
    file: a heavy scene's gigabyte of geometry no longer goes to the disk and back. MoonRay is still handed the file."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'persistent.py')
    p.rep("""    def _release_mappings(self):""", """    def _write_binary(self,request,payload,memory):
        \"\"\"The geometry's binary scene (rdlb.py): in shared memory where XPU2 is to read it (modo-memory-rdlb), else a
        file beside the text.\"\"\"
        if memory and payload:
            name='Local\\\\MoonRayForModoScene_%d_%s'%(os.getpid(),uuid.uuid4().hex)
            mapping=mmap.mmap(-1,len(payload),tagname=name,access=mmap.ACCESS_WRITE)
            mapping.write(payload)
            self.mappings.setdefault(request['id'],[]).append(mapping)
            return 'modo-memory-rdlb:%d:%s'%(len(payload),name)
        binary=self.root/('%d.full.rdlb'%request['id']);binary.write_bytes(payload)
        return binary

    def _release_mappings(self):""")
    p.rep("""            scene=self._write_scene(request,'full',None,split[0])
            binary=self.root/('%d.full.rdlb'%request['id']);binary.write_bytes(split[1])""", """            scene=self._write_scene(request,'full',None,split[0])
            # Whether XPU2 takes it (as below), which reads the binary scene from shared memory.
            xpu2=bool(native.xpu2() and request['mode'] in ('xpu','auto') and not getattr(self,'xpu2_refused',None))
            binary=self._write_binary(request,split[1],xpu2)""")
    p.rep("""            binary=self.root/('%d.full.rdlb'%request['id']);binary.write_bytes(prepared['binary'])
            full='%s|%s'%(full,binary)""", """            binary=self._write_binary(request,prepared['binary'],getattr(self,'xpu2_active',False))
            full='%s|%s'%(full,binary)""")
    p.save()


def xpu2_smart_render(kit):
    """Smart Render is MoonRay's: while XPU2 renders the preview (Preferences > MoonRay > XPU2) it is greyed in the render
    settings and in the picture's menu, and no XPU2 render is stopped by it. And MoonLightIPR's images at most 1024 a
    side, as the main kit has them since 2026-10-27 (realtime's textures)."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'moonlightipr_materials.py')
    p.rep("MAX_TEXTURE_SIDE = 4096", "# At most 1024 a side: perceptually enough at realtime's sizes (it was 4096, then 1280).\nMAX_TEXTURE_SIDE = 1024")
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_render_settings.py')
    p.rep("""                if entry['key'] in NEEDS_DENOISER:
                    try:return scene_settings.complete(properties.scene_settings())['denoising'].get('engine','off')!='off'""",
          """                # Smart Render is MoonRay's: greyed while XPU2 renders the preview.
                if entry['key']=='smart_render':
                    try:
                        from moonray_modo import native
                        if native.xpu2():return False
                    except Exception:pass
                if entry['key'] in NEEDS_DENOISER:
                    try:return scene_settings.complete(properties.scene_settings())['denoising'].get('engine','off')!='off'""")
    p.save()
    p = Patch(kit / 'python' / 'moonray_modo' / 'render.py')
    p.rep("""            if level!='off' and not self.active['output'] and not self.active['moonlightipr'] and not self.active.get('linear_preview') and not self.active['snapshot'].get('_ipr'):""",
          """            # Not while XPU2 renders the preview: Smart Render is MoonRay's.
            if level!='off' and not self.active['output'] and not self.active['moonlightipr'] and not self.active.get('linear_preview') and not self.active['snapshot'].get('_ipr') and not native.xpu2():""")
    p.save()
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""        menu.addMenu(smart);self._smart_menu=smart""",
          """        # Smart Render is MoonRay's: greyed while XPU2 renders the preview.
        try:
            from . import native
            if native.xpu2():smart.setEnabled(False);smart.setToolTip("Smart Render is MoonRay's; XPU2 renders the preview.")
        except Exception:pass
        menu.addMenu(smart);self._smart_menu=smart""")
    p.rep("""        from . import denoising
        try:return denoising.settings(self._settings_values()['denoising'])['smart']!='off'""",
          """        from . import denoising, native
        try:
            if native.xpu2():return False
            return denoising.settings(self._settings_values()['denoising'])['smart']!='off'""")
    p.save()


def meter_switch(kit):
    """A preference for the preview window's memory meter (Preferences > MoonRay, on): off, the meter is hidden and reads
    nothing, to tell whether its once-a-second reading is what makes the preview hesitate (asked 2026-10-27)."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'preferences.py')
    p.rep("""    'low_latency': (True, bool),""", """    'low_latency': (True, bool),
    # The preview window's memory meter (memory_meter.py).
    'memory_meter': (True, bool),""")
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_preferences.py')
    p.rep("""lx.bless(switch('low_latency'),'moonray.pref.lowLatency')""", """lx.bless(switch('low_latency'),'moonray.pref.lowLatency')
lx.bless(switch('memory_meter'),'moonray.pref.memoryMeter')""")
    p.save()
    p = Patch(kit / 'preferences.cfg')
    p.rep("""      <list type="Control" val="cmd moonray.pref.binaryGeometry ?">""", """      <list type="Control" val="cmd moonray.pref.memoryMeter ?">
        <atom type="Label">Memory meter</atom>
        <atom type="Tooltip">The preview window shows the renderer's RAM, the machine's free RAM and the graphics card's memory beside the picture's details, read once a second. Off, it is hidden and reads nothing</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.binaryGeometry ?">""")
    p.save()
    p = Patch(py / 'memory_meter.py')
    p.rep("""    def refresh(self):
        self.reader.pids = [p for p in (self.processes() or []) if p]""", """    def refresh(self):
        # Off in Preferences > MoonRay (Memory meter): hidden, and nothing read.
        try:
            from . import native
            on = native._switch('memory_meter', 'MOONRAY_MODO_MEMORY_METER')
        except Exception:
            on = True
        self.reader.active = on
        if not on:
            self.hide()
            return
        self.show()
        self.reader.pids = [p for p in (self.processes() or []) if p]""")
    p.save()


def motion_skip(kit):
    """MoonLightIPR's motion-blurred readings left out where they cannot change the picture (the main kit, 2026-10-27):
    a move of items none of whose channels is animated, in a scene last found to hold nothing that moves, is read once
    (a fur scene held Modo 0.8 s at each release for three readings); a scene so found keeps its meshes for the next
    move; clip folders (imageFolder) count as steady items. The block is taken from the repo's panel.py as it is."""
    py = kit / 'python' / 'moonray_modo'
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    start = source.index('            # MoonLightIPR with motion blur: once an edit is let go')
    end = source.index('            if not blurred:', start)
    p = Patch(py / 'panel.py')
    begin = p.text.index('            # MoonLightIPR with motion blur: once an edit is let go')
    finish = p.text.index('            if not blurred:', begin)
    p.text = p.text[:begin] + source[start:end] + p.text[finish:]
    p.save()
    inc = (REPO_KIT / 'incremental.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    a = inc.index('def animated(scene,identities):')
    b = inc.index('def item_matrices(scene,snapshot):')
    p = Patch(py / 'incremental.py')
    p.rep("""'scene','videoStill'}|LIGHTS|MATERIALS""", """'scene','videoStill','imageFolder'}|LIGHTS|MATERIALS""")
    p.rep("""def item_matrices(scene,snapshot):""", inc[a:b] + """def item_matrices(scene,snapshot):""")
    p.save()


def shutter_reading(kit):
    """A MoonLightIPR edit with motion blur reads the scene once and asks the shutter's ends of where the items stand
    (animation.shutter_moves, from the main repo as it is; panel's block comes with motion_skip): the interior's mesh
    edit held Modo 5.7 s for two readings of its evaluated geometry (2026-10-27)."""
    source = (REPO_KIT / 'animation.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    a = source.index('def shutter_moves(snapshot, evaluated=False):')
    b = source.index('\nclass Sequence(QtCore.QObject):')
    p = Patch(kit / 'python' / 'moonray_modo' / 'animation.py')
    p.rep("""from .motion import apply_motion
""", """from .motion import apply_motion


""" + source[a:b].rstrip('\n') + """
""")
    p.save()


def edited_surfaces(kit):
    """A mesh edit to a scene of Modo's evaluated geometry reads again only the edited meshes' surfaces from its render
    cache, the others' kept (main repo, uncommitted 2026-10-27; the adapter's MR_geometry_snapshot_some, in the one
    geometry_detail copies; evaluated.py is copied whole by geometry_rows): an interior's edit read 25 surfaces, 2.5 s.
    incremental.steady and rigid, and the classify rule, are taken from the repo as it is; host.py patched."""
    py = kit / 'python' / 'moonray_modo'
    inc = (REPO_KIT / 'incremental.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'incremental.py')
    a, b = p.text.index('def rigid(scene,snapshot):'), p.text.index('def animated(scene,identities):')
    p.text = p.text[:a] + inc[inc.index('def steady(scene):'):inc.index('def animated(scene,identities):')] + p.text[b:]
    # The rule for an edited mesh, whole as the repo has it (Modo's evaluated geometry where each mesh's geometry is its
    # own, own_geometry, 2026-10-28), in place of the kit's.
    start = "        # Modo's evaluated geometry too, where each mesh's geometry is its own (own_geometry)"
    rule = inc[inc.index(start):inc.index('        return False\n', inc.index(start)) + len('        return False\n')]
    a = p.text.index("        if cache.get('_evaluated_data') is None and not cache.get('extra_geometry'):\n")
    b = p.text.index('        return False\n', a) + len('        return False\n')
    p.text = p.text[:a] + rule + p.text[b:]
    p.save()
    p = Patch(py / 'host.py')
    p.rep("""            cached=reuse_geometry.get('_evaluated_data') if reuse_geometry else None
            data=dict(cached,surfaces=[dict(v) for v in cached['surfaces']]) if cached else evaluated.capture(lx.service.Selection().GetTime())""",
          """            cached=reuse_geometry.get('_evaluated_data') if reuse_geometry else None
            if cached and dirty_meshes:
                # Edited meshes: their surfaces read again, the others' as they were (incremental.classify).
                data=evaluated.capture(lx.service.Selection().GetTime(),only=dirty_meshes,before=cached)
            else:
                data=dict(cached,surfaces=[dict(v) for v in cached['surfaces']]) if cached else evaluated.capture(lx.service.Selection().GetTime())""")
    p.save()


def xpu2_sampling_words(kit):
    """While XPU2 renders the preview its sampling generations are on (their switch greyed), MoonRay's own sampling
    fields are greyed and show what XPU2 takes, and the picture's sampling label says how XPU2 samples it: all worked
    out as XPU2's plan does (xpu2_sampling.py beside this builder, copied into the kit; cpu_scene.h). Asked 2026-10-28."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(Path(__file__).resolve().parent / 'xpu2_sampling.py', py / 'xpu2_sampling.py')
    p = Patch(py / 'panel.py')
    p.rep("""        settings=render_values(settings or {})
        return ('LS ' if engine=='ls' else '')+""", """        settings=render_values(settings or {})
        # XPU2 renders the preview: how it samples it (xpu2_sampling), not MoonRay's Uniform or Adaptive.
        from . import xpu2_sampling
        if engine=='moonray' and xpu2_sampling.active():
            try:return xpu2_sampling.words(self._settings_values())
            except Exception:pass
        return ('LS ' if engine=='ls' else '')+""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("""        values=[1,1 if generation.get('on',True) else 0,""", """        # On: they are whenever XPU2 renders the preview, and only XPU2 reads them (xpu2_sampling). Form 2: they
        # rule, a scene's adaptive sampling being MoonRay's alone (cpu_scene.h).
        from . import xpu2_sampling
        xpu2_sampling.CAMERA[0]=xpu2_sampling.camera_kind(entity_camera)
        values=[2,1,""")
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_render_settings.py')
    p.rep("""                    try:return bool(scene_settings.complete(properties.scene_settings())['xpu2_generations'].get('on',True))
                    except Exception:return True""", """                    return True""")
    p.rep("""                # XPU2's sampling generations: on, MoonRay's own sampling settings are greyed
                # (XPU2 previews use the XPU2 ones; final renders, MoonRay's, still use them).
                if entry['key'] in ('sampling_mode','samples','min_adaptive_samples','max_adaptive_samples','target_adaptive_error'):
                    try:
                        if scene_settings.complete(properties.scene_settings())['xpu2_generations'].get('on',True):return False
                    except Exception:pass
""", """                # XPU2 renders the preview: its sampling generations are on, their switch is
                # greyed, and so are MoonRay's own sampling settings, which show what XPU2 takes (xpu2_sampling), and
                # what XPU2 does not read at all (cpu_scene.h): MoonRay's buckets and their order, and two sampling
                # switches of MoonRay's own. Final renders, MoonRay's, still use them all.
                if entry['key'] in ('xpu2_generations','sampling_mode','samples','min_adaptive_samples','max_adaptive_samples','target_adaptive_error',
                                    'bucket_size','batch_tile_order','important_first','volume_indirect_samples','disable_optimized_hair_sampling'):
                    from moonray_modo import xpu2_sampling
                    if xpu2_sampling.active():return False
""")
    p.rep("""                if kind in ('bool','member','int','choice'):values.AddInt(int(value))""", """                # XPU2 renders the preview: the generations on, MoonRay's sampling fields showing what XPU2 takes.
                from moonray_modo import xpu2_sampling
                if xpu2_sampling.active():
                    try:value=xpu2_sampling.shown(scene_settings.complete(properties.scene_settings()),entry['key'],value)
                    except Exception:pass
                if kind in ('bool','member','int','choice'):values.AddInt(int(value))""")
    p.save()


def kit_version(kit):
    """The kit's version as the main repo has it (0.5 from 2026-10-28, the next release): __init__.py copied, and
    index.cfg's version attribute set to the same."""
    import re
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / '__init__.py', py / '__init__.py')
    held = re.search(r'__version__ = "([^"]+)"', (REPO_KIT / '__init__.py').read_text(encoding='utf-8')).group(1)
    p = Patch(kit / 'index.cfg')
    p.text = re.sub(r'(<configuration kit="MoonRayForModo" version=")[^"]*(")', r'\g<1>' + held + r'\g<2>', p.text, count=1)
    p.save()


def hdri_and_texture_shape(kit):
    """MoonLightIPR as the main repo has it since 2026-10-28 (Dose): a plain HDRI turned half a turn about the
    vertical in its own space, as MoonRay is given it (environments.emit), where it faced the other way; and a texture
    over the size limit resized with its shape kept (moonlightipr_materials.fitted), where --fit padded it to a
    square and a label filled only the middle of its UVs."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'moonlightipr_scene.py')
    p.rep("""                        baked, matrix = held[0], finite(held[1])
                # The image's own axes in world space, which the session turns directions into.
""", """                        baked, matrix = held[0], finite(held[1])
                if item['kind'] == 'image':
                    # Half a turn about the vertical in its own space, as Modo shows a plain image and as MoonRay is given
                    # it (environments.emit): Dose's HDRI faced the other way, the can lit by the room's bright side.
                    matrix = [-v for v in matrix[0:4]] + matrix[4:8] + [-v for v in matrix[8:12]] + matrix[12:16]
                # The image's own axes in world space, which the session turns directions into.
""")
    p.save()
    p = Patch(py / 'moonlightipr_materials.py')
    p.rep("""MAX_TEXTURE_SIDE = 1024
RAMP_STEPS = 257
""", """MAX_TEXTURE_SIDE = 1024


def fitted(width, height, side=MAX_TEXTURE_SIDE):
    \"\"\"An image's size made no larger than side on either axis, its shape kept.\"\"\"
    scale = side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


RAMP_STEPS = 257
""")
    p.rep("MAX_TEXTURE_SIDE, 'moonlightipr-v2', ocio and (space, ocio)", "MAX_TEXTURE_SIDE, 'moonlightipr-v3', ocio and (space, ocio)")
    p.rep("""            args += ['--fit', '%dx%d' % (MAX_TEXTURE_SIDE, MAX_TEXTURE_SIDE)]
""", """            # Its own shape, its longer side at the limit: --fit pads to the square, and Dose's 2103x1673 label filled
            # only the middle of its UVs, the can dark above and below it.
            args += ['--resize', '%dx%d' % fitted(width, height)]
""")
    p.save()


def lights_seen_by_camera(kit):
    """Lights the camera sees, as the main repo has it since 2026-10-28: Modo's Visible to Camera (visCam) and a
    MoonRay light's visible_in_camera forced on reach both engines, and MoonLightIPR shows the light where it stands,
    its picture on it. The repo's MoonLightIPR writes that as its form N, but its layer and filter records have grown
    since the base kit's form L without a form of their own; so the base kit's session (the repo at aba69c9) is
    rebuilt with only this change, as scene mark 'l' (test-results/moonlightipr-form-l, the change put in by
    backport_seen.py there), and the packer here writes the one number more."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'host.py')
    p.rep("""                 'width': float(channel(item, 'width', 1)), 'height': float(channel(item, 'height', 1))}""",
          """                 'width': float(channel(item, 'width', 1)), 'height': float(channel(item, 'height', 1)),
                 # Visible to Camera: the light seen shining where it stands (MoonRay's visible_in_camera).
                 'camera_visible': bool(channel(item, 'visCam', 0))}""")
    p.save()
    p = Patch(py / 'entities.py')
    p.rep("""            if name != 'PortalLight' and value(entity, 'texture'):""",
          """            # Visible in Camera forced on (1): the camera sees it shining. Left to the scene's default, MoonRay hides it.
            if name != 'PortalLight' and 'visible_in_camera' in catalog()[name]['attributes']:
                light['camera_visible'] = int(value(entity, 'visible_in_camera')) == 1
            if name != 'PortalLight' and value(entity, 'texture'):""")
    p.save()
    p = Patch(py / 'moonlightipr_scene.py')
    p.rep("""        local.append(struct.pack('<I20fi', LOCAL_LIGHTS[kind], *(matrix[12:15] + unit(matrix[0:3]) + unit(matrix[4:7])""",
          """        # Seen by the camera where it stands, as MoonRay shows a light whose Visible in Camera is on (a portal never).
        seen = 1 if light.get('camera_visible') and kind != 'PortalLight' else 0
        local.append(struct.pack('<I20fiI', LOCAL_LIGHTS[kind], *(matrix[12:15] + unit(matrix[0:3]) + unit(matrix[4:7])""")
    p.rep("""            + [cone, max(0.0, cone - 2 * float(light.get('soft_edge', 0)))]), texture)""",
          """            + [cone, max(0.0, cone - 2 * float(light.get('soft_edge', 0)))]), texture, seen)""")
    p.rep("""    seen = 1.0 if settings.get('light_seen', True) else 0.0
    return (struct.pack('<I20fiI', MESH_LIGHT, *([0.0] * 12 + [1.0, 1.0, 1.0] + [c / (math.pi * cached[1]) for c in color] + [seen, 0.0]), -1, 0)""",
          """    seen = 1 if settings.get('light_seen', True) else 0
    return (struct.pack('<I20fiII', MESH_LIGHT, *([0.0] * 12 + [1.0, 1.0, 1.0] + [c / (math.pi * cached[1]) for c in color] + [0.0, 0.0]), -1, seen, 0)""")
    p.rep("b'MLSL'", "b'MLSl'")
    p.save()
    built = Path(__file__).resolve().parent.parent / 'moonlightipr-form-l' / 'build' / 'moonlightipr'
    held = kit / 'runtime' / 'moonlightipr'
    shutil.copy2(built / 'bin' / 'moonlightipr_session.exe', held / 'moonlightipr_session.exe')
    for ptx in (built / 'shaders').glob('MoonLightIPRKernel*.ptx'):
        shutil.copy2(ptx, held / ptx.name)


def flowing_toolbar(kit):
    """The preview window's top row as the main repo has it since 2026-10-28: its groups go on to a new line where
    the window is too narrow (flow_layout.py, copied), each control as wide as Modo's style needs for its longest word,
    in place of fixed widths every button was squeezed below and its word cut. The end of panel._toolbar, from its
    sizing on, is the repo's."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'flow_layout.py', py / 'flow_layout.py')
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    start = source.index('        # One height for everything in the row, and as wide as each needs')
    end = source.index('    def _options_menu(self):', start)
    p = Patch(py / 'panel.py')
    old_start = p.text.index('        # One height for everything in the row, and no more width than each needs')
    old_end = p.text.index('    def _options_menu(self):', old_start)
    if p.text[old_start:old_end].count('        return row\n') != 1:
        raise RuntimeError('panel.py: the end of _toolbar is not as expected')
    p.text = p.text[:old_start] + source[start:end] + p.text[old_end:]
    # Gamma and alpha as their signs, square as the channels' letters; R, G and B in their colours.
    p.rep("            button.setStyleSheet('QToolButton:checked { background-color: %s; color: %s; border: 1px solid %s; border-radius: 3px; }'%(lit,ink,lit))",
          "            # R, G and B written in their own colours, lighter than when lit so that they read on Modo's grey; alpha as\n            # the other buttons are.\n            letter={'r':'#ff6b5e','g':'#5fd27c','b':'#6f9dff'}.get(key)\n            button.setStyleSheet(('QToolButton:!checked { color: %s; font-weight: bold; } '%letter if letter else '')\n                                 +'QToolButton:checked { background-color: %s; color: %s; border: 1px solid %s; border-radius: 3px; }'%(lit,ink,lit))")
    # EV and \u03b3 as the repo has them: each toggles between its plain value and the one set, and is lit while the one
    # set is shown (_mark_display). The kit's block, from the EV button to the gamma field, made the repo's.
    start = p.text.index("        reset=QtWidgets.QToolButton();reset.setText('EV');reset.setToolTip('Exposure, in stops. Click to put it back to 0.')")
    end = p.text.index("        row.addWidget(self.gamma)\n", start) + len("        row.addWidget(self.gamma)\n")
    block = source[source.index("        reset=self.exposure_reset=QtWidgets.QToolButton();reset.setText('EV')"):]
    block = block[:block.index("        self._mark_display()\n") + len("        self._mark_display()\n")]
    p.text = p.text[:start] + block + p.text[end:]
    methods = source[source.index('    def _mark_display(self):'):source.index('    def _gamma_changed(self,value):')]
    p.rep('    def _gamma_changed(self,value):', methods + '    def _gamma_changed(self,value):')
    p.rep("            blocker=QtCore.QSignalBlocker(self.exposure);self.exposure.setValue(values['display']['exposure']);del blocker\n",
          "            blocker=QtCore.QSignalBlocker(self.exposure);self.exposure.setValue(values['display']['exposure']);del blocker\n"
          "            # Set quietly: lit by hand, as its signal is held.\n            self._mark_display()\n")
    p.rep("                                      ('a','A','#e8e8e8','#101010','Show the alpha channel. Click again for the whole picture.')):",
          "                                      ('a','\u03b1','#e8e8e8','#101010','Show the alpha (\u03b1) channel. Click again for the whole picture.')):")
    p.save()


def tools_ngons_clear_cache(kit):
    """The main repo's round of 2026-10-28, late: a runtime's maketx and oiiotool found in the runtime chosen as well as
    the kit's, its bin and the folder above (native.tool; a tester was told to reinstall while maketx was there); n-gons
    cut by MoonLightIPR's rules and handed to MoonRay so cut (polygons.cut_mesh in rdla, behind the MoonRay Source
    Override "N-gons triangulated cleanly"; polygons.py is copied whole by ear_clipping); and Clear Cache beside the
    Cache folder (cache_folder.clear; cache_folder.py is copied whole by scene_cache). Each piece of code is taken from
    the repo's files."""
    py = kit / 'python' / 'moonray_modo'

    def between(name, start, end):
        source = (REPO_KIT / name).read_text(encoding='utf-8').replace('\r\n', '\n')
        a = source.index(start)
        return source[a:source.index(end, a)]
    p = Patch(py / 'native.py')
    p.rep("def installation_id():", between('native.py', 'def tool(name, first=None):', 'def installation_id():') + 'def installation_id():')
    p.rep("def source_switches():", between('native.py', 'def ngon_ears():', 'def source_switches():') + 'def source_switches():')
    p.save()
    p = Patch(py / 'textures.py')
    p.rep("""    runtime = Path(native.default_runtime())
    converter = runtime / 'maketx.exe'
    if not converter.is_file():
        raise ValueError('Texture conversion needs maketx.exe in the MoonRay runtime. Reinstall the kit.')
""", """    # Where the runtime's maketx is (native.tool: the chosen runtime, then the kit's), its folder the runtime it runs in.
    converter = native.tool('maketx.exe')
    runtime = converter.parent
""")
    p.save()
    p = Patch(py / 'moonlightipr_materials.py')
    p.rep("""        runtime = Path(self.runtime or native.default_runtime())
        tool = runtime / 'oiiotool.exe'
        if not tool.is_file():
            raise ValueError('Texture conversion needs oiiotool.exe in the MoonRay runtime')
""", """        tool = native.tool('oiiotool.exe', self.runtime)
        runtime = tool.parent
""")
    p.save()
    p = Patch(py / 'environment_layers.py')
    p.rep("""    runtime=Path(native.default_runtime());converter=runtime/'oiiotool.exe'
    if not converter.is_file():raise ValueError('Layered environments require oiiotool.exe')
""", """    converter=native.tool('oiiotool.exe');runtime=converter.parent
""")
    p.rep("""            runtime = Path(native.default_runtime())
            result = subprocess.run([str(runtime / 'oiiotool.exe'), '--info:format=xml', '-v', str(path)], env=native.environment(runtime),""",
          """            tool = native.tool('oiiotool.exe')
            result = subprocess.run([str(tool), '--info:format=xml', '-v', str(path)], env=native.environment(tool.parent),""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("""        vertices = mesh['vertices']
""", between('rdla.py', "        if not mesh.get('subdivision') and mesh.get('faces'):\n", "        vertices = mesh['vertices']\n") + """        vertices = mesh['vertices']
""")
    p.save()
    p = Patch(py / 'preferences.py')
    p.rep("    'transmission_roughness': (True, bool),\n", "    'transmission_roughness': (True, bool),\n    # N-gons ear-clipped whatever their size and winding (native.ngon_ears).\n    'ngon_ears': (True, bool),\n")
    # The button itself comes with the Cache folder's row, which cache_folder takes from the repo whole.
    p.rep("    def _browse_cache(self):", between('preferences.py', '    def _clear_cache(self):', '    def _browse_cache(self):') + '    def _browse_cache(self):')
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_preferences.py')
    p.rep("lx.bless(switch('auto_xpu'),'moonray.pref.autoXpu')\n", "lx.bless(switch('auto_xpu'),'moonray.pref.autoXpu')\nlx.bless(switch('ngon_ears'),'moonray.pref.ngonEars')\n")
    p.save()
    source = (REPO_KIT.parents[1] / 'preferences.cfg').read_text(encoding='utf-8').replace('\r\n', '\n')
    a = source.index('      <list type="Control" val="cmd moonray.pref.ngonEars ?">')
    block = source[a:source.index('      <list type="Control" val="cmd moonray.pref.lightMotion ?">', a)]
    p = Patch(kit / 'preferences.cfg')
    p.rep('      <list type="Control" val="cmd moonray.pref.lightMotion ?">', block + '      <list type="Control" val="cmd moonray.pref.lightMotion ?">')
    p.save()


def about_xpu2(kit):
    """XPU2 credited in the About box beside MoonLightIPR (asked 2026-10-28; XPU2 is to be public with 0.5, and is not
    named in the main repo before then)."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'about.py')
    p.rep(r"MoonLightIPR GPU preview: created by Raphael Tobar · MIT License · not affiliated with DreamWorks Animation\n",
          r"MoonLightIPR GPU preview: created by Raphael Tobar · MIT License · not affiliated with DreamWorks Animation\n"
          r"XPU2: Asynchronous Monte Carlo Path Tracer designed by Raphael Tobar · not affiliated with or endorsed by DreamWorks Animation\n")
    p.save()


def engine_names(kit):
    """MoonRay named by its execution mode in the MoonRay Terminal (main repo, 2026-10-28): MoonRay Vector, Scalar, XPU
    or Auto, and here MoonRay XPU2 where XPU2 renders the preview (asked: "MoonRay XPU2", not "XPU2"). The repo's
    native.engine_name, persistent.Session.engine_name and render.py's naming."""
    py = kit / 'python' / 'moonray_modo'
    # And the popup descriptions kept for good (hint_cache.py, copied): freed past 512, likely a crash opening a
    # viewport's menu.
    shutil.copyfile(REPO_KIT / 'hint_cache.py', py / 'hint_cache.py')
    source = (REPO_KIT / 'native.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    a = source.index('# What MoonRay is called in the MoonRay Terminal')
    p = Patch(py / 'native.py')
    p.rep('def source_switches():', source[a:source.index('def source_switches():', a)] + 'def source_switches():')
    p.save()
    p = Patch(py / 'persistent.py')
    p.rep("        terminal.watch(self.process,'MoonRay session',resident=True)",
          "        # Named in the MoonRay Terminal by the mode it starts in (native.engine_name), or MoonRay XPU2 where XPU2 runs.\n"
          "        terminal.watch(self.process,self.engine_name,resident=True)")
    p.rep("    def select_view(self,key):", '''    def engine_name(self):
        """What the Terminal calls this MoonRay: MoonRay XPU2 where XPU2 renders, else by the execution mode it starts in."""
        if getattr(self,'xpu2_active',False):return 'MoonRay XPU2'
        return native.engine_name((self.sent or self.latest or {}).get('mode'))

    def select_view(self,key):''')
    p.save()
    p = Patch(py / 'render.py')
    p.rep("        terminal.watch(self.process, lambda: 'Denoiser' if self.phase == 'denoise' else 'MoonRay')",
          "        # MoonRay named by the mode it renders in (native.engine_name): MoonRay Vector, Scalar, XPU or Auto.\n"
          "        terminal.watch(self.process, lambda: 'Denoiser' if self.phase == 'denoise' else native.engine_name(getattr(self, 'engine_mode', None)))")
    p.rep("            args=native.arguments(scene, self.image_path, request['threads'],mode)",
          "            args=native.arguments(scene, self.image_path, request['threads'],mode)\n            self.engine_mode=mode")
    p.save()


def details_line(kit):
    """The picture's details (resolution, sampling, noise) as wide as their words at the bottom right, cut short with an
    ellipsis only where the window is too narrow (main repo, 2026-10-28): a third of the row cut them off. The repo's
    ElidedLabel and cells."""
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')

    def between(start, end):
        a = source.index(start)
        return source[a:source.index(end, a)]
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""        cells=[]
        for _ in range(3):
            cell=QtWidgets.QWidget();cell.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Preferred)
            inside=QtWidgets.QHBoxLayout(cell);inside.setContentsMargins(0,0,0,0);inside.setSpacing(8)
            row.addWidget(cell,1);cells.append(inside)
""", between('        cells=[]\n        for index in range(3):\n', '        left,middle,right=cells\n'))
    p.rep("        self.image_info=QtWidgets.QLabel('');self.image_info.setEnabled(False)",
          "        self.image_info=ElidedLabel('');self.image_info.setEnabled(False)")
    p.rep("\nclass Panel(", '\n' + between('class ElidedLabel(QtWidgets.QLabel):', 'class Panel(') + 'class Panel(')
    p.save()


def memory_in_picture(kit):
    """The memory meter at the bottom right of the picture, as the HUD's lines are, while Preview Preferences says
    (Memory in the picture; main repo, 2026-10-28): it was beside the picture's details in the status row, switched in
    Preferences > MoonRay (meter_switch, no longer run). memory_meter.py is the repo's (repo_fixes copies it); the
    panel's and the preferences' lines are taken from the repo. Run after the HUD is in the panel."""
    py = kit / 'python' / 'moonray_modo'
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')

    def between(start, end):
        a = source.index(start)
        return source[a:source.index(end, a)]
    p = Patch(py / 'panel.py')
    p.rep("""        # The memory the renderer, the machine and the graphics card have, beside the picture's details.
        from .memory_meter import MemoryMeter
        self.memory_meter=MemoryMeter(self._renderer_processes)
        right.addWidget(self.memory_meter)
""", '')
    p.rep("""        self.hud.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents);self.hud.hide()
""", """        self.hud.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents);self.hud.hide()
""" + between("        # The memory the renderer, the machine and the graphics card have, at the bottom right", "        self.preview.setContextMenuPolicy("))
    p.rep("""and hasattr(self,'hud') and self.hud.isVisible():self._hud_place()
""", """and hasattr(self,'hud') and self.hud.isVisible():self._hud_place()
        if watched is self.preview and event.type()==QtCore.QEvent.Resize and hasattr(self,'memory_meter'):self._meter_place()
""")
    p.rep("""        if words:self._hud_place();self.hud.raise_()
""", """        if words:self._hud_place();self.hud.raise_()
        self._meter_place()
""")
    p.rep("    def _hud_place(self):", between('    def _meter_place(self):', '    def _hud_place(self):') + '    def _hud_place(self):')
    p.save()
    source = (REPO_KIT / 'preferences.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'preferences.py')
    p.rep("'preview_motion': (False, bool),", "'preview_motion': (False, bool), 'memory_hud': (True, bool),")
    p.rep("        heading = QtWidgets.QLabel('MoonLightIPR quality')", between("        form.addRow(check('memory_hud'", "        heading = QtWidgets.QLabel('MoonLightIPR quality')") +
          "        heading = QtWidgets.QLabel('MoonLightIPR quality')")
    p.save()


def preference_pages(kit):
    """MoonRay's preferences as pages of their own in Modo's Preferences list (main repo, 2026-10-28): General (the
    preview window, lighting, Assign Material), Renderer, XPU (Auto on the graphics card first, from the overrides) and
    Source Overrides, as the repo's preferences.cfg has them; and XPU2 a page of its own, where it was a section of the
    General page. Run after XPU2's section is added (build)."""
    p = Patch(kit / 'preferences.cfg')
    # XPU2 no longer a section of the General page.
    p.rep(SHEET_ENTRY, '')
    pages = (('renderer', 'Renderer', 'MoonRayRendererPage:sheet', 'MoonRayRenderer:sheet', 'Renderer', '80.61'),
             ('xpu', 'XPU', 'MoonRayXPUPage:sheet', 'MoonRayXPU:sheet', 'XPU (graphics card)', '80.62'),
             ('overrides', 'Source Overrides', 'MoonRaySourceOverridesPage:sheet', 'MoonRaySourceOverrides:sheet', 'MoonRay Source Overrides', '80.63'),
             ('xpu2', 'XPU2', 'MoonRayXPU2Page:sheet', 'MoonRayXPU2:sheet', 'XPU2', '80.64'))
    for hash_key in ('MoonRayRenderer:sheet', 'MoonRaySourceOverrides:sheet'):
        start = p.text.index('      <list type="Control" val="sub %s">' % hash_key)
        end = p.text.index('      </list>\n', start) + len('      </list>\n')
        p.text = p.text[:start] + p.text[end:]
    # The Auto XPU control moved to the XPU page's own sheet.
    start = p.text.index('      <list type="Control" val="cmd moonray.pref.autoXpu ?">\n        <atom type="Label">')
    end = p.text.index('      </list>\n', start) + len('      </list>\n')
    auto, p.text = p.text[start:end], p.text[:start] + p.text[end:]
    sheets = ['''    <!-- MoonRay on the graphics card. -->
    <hash type="Sheet" key="MoonRayXPU:sheet">
      <atom type="Label">XPU (graphics card)</atom>
      <atom type="Layout">properties</atom>
''' + auto + '''    </hash>
''']
    for key, name, page, inner, label, ordinal in pages:
        sheets.append('''    <hash type="Sheet" key="%s">
      <atom type="Label">%s</atom>
      <atom type="Layout">properties</atom>
      <list type="Control" val="sub %s">
        <atom type="Label">%s</atom>
        <atom type="ShowLabel">1</atom>
        <atom type="StartCollapsed">0</atom>
        <atom type="Hash">%s</atom>
      </list>
      <atom type="Filter">prefs/moonray/%s:filterPreset</atom>
      <hash type="InCategory" key="prefs:general#head">
        <atom type="Ordinal">%s</atom>
      </hash>
      <atom type="Group">prefs/moonray</atom>
    </hash>
''' % (page, name, inner, label, inner, key, ordinal))
        p.rep('''  <atom type="Filters">
''', '''  <atom type="Filters">
    <hash type="Preset" key="prefs/moonray/%s:filterPreset">
      <atom type="Name">%s</atom>
      <atom type="Category">20385740002:filterCat</atom>
      <atom type="Enable">1</atom>
      <list type="Node">1 .group 0 &quot;&quot;</list>
      <list type="Node">1 prefType moonray/%s</list>
      <list type="Node">-1 .endgroup </list>
    </hash>
''' % (key, name, key))
    p.rep('''    <hash type="PrefCat" key="moonray/general"></hash>
''', '''    <hash type="PrefCat" key="moonray/general"></hash>
''' + ''.join('    <hash type="PrefCat" key="moonray/%s"></hash>\n' % page[0] for page in pages))
    p.rep('''      <hash type="T" key="moonray/general">General</hash>
''', '''      <hash type="T" key="moonray/general">General</hash>
''' + ''.join('      <hash type="T" key="moonray/%s">%s</hash>\n' % page[:2] for page in pages))
    p.rep('''    </hash>
  </atom>
  <atom type="Filters">''', '''    </hash>
''' + ''.join(sheets) + '''  </atom>
  <atom type="Filters">''')
    p.save()


def ear_clipping(kit):
    """MoonLightIPR's n-gons ear-clipped as MoonRay's TriMesh::splitNGons does (polygons.py, copied from the repo,
    2026-10-28): fanned, a concave n-gon was drawn over its own notch. simplify.py comes whole with geometry_detail."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'polygons.py', py / 'polygons.py')
    p = Patch(py / 'moonlightipr_scene.py')
    a = p.text.index('def _fan(size, _held={}):')
    b = p.text.index('\n\n\n', a) + 1
    p.text = p.text[:a] + 'from .polygons import fan as _fan, corners as _corners\n' + p.text[b:]
    p.rep("""            fan = array('I', [start + k for start, size in zip(accumulate([0] + sizes[:-1]), sizes) for k in _fan(size)])""",
          """            # An n-gon ear-clipped as MoonRay does it (polygons.corners), the rest fanned.
            fan = array('I', [start + k for start, face in zip(accumulate([0] + sizes[:-1]), faces) for k in _corners(face, vertices)])""")
    p.rep("""            indices = array('I', [face[k] for face in faces for k in _fan(len(face))])""",
          """            indices = array('I', [face[k] for face in faces for k in _corners(face, vertices)])""")
    p.save()


def scene_read_cache(kit):
    """A whole reading of a scene with no unsaved changes kept by the Scene cache for the next cold start, under a hash
    of the scene file and what else it was read with (scene_reading.py, copied from the repo; panel._capture patched as
    the repo has it, 2026-10-28): the car's 30.7 s reading from Modo loaded back in 1.5 s, the same 334 meshes."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'scene_reading.py', py / 'scene_reading.py')
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'panel.py')
    line = """        scene = host.snapshot(evaluated_geometry=self._settings_values()['surface'] == 2,reuse_geometry=self._geometry_cache if reuse else None,refresh_materials=reuse=='materials' or isinstance(reuse,dict),dirty_meshes=reuse.get('dirty_meshes') if isinstance(reuse,dict) else None)
"""
    start = source.index('        # A whole reading of a scene with no unsaved changes, as the Scene cache kept it')
    end = source.index('            if held_key is not None:scene_reading.keep(held_key,scene_reading.freeze(scene),scene)\n', start)
    p.rep(line, source[start:end] + '            if held_key is not None:scene_reading.keep(held_key,scene_reading.freeze(scene),scene)\n')
    p.save()


def reading_status(kit):
    """Reading Modo's scene graph... said before every whole reading of the scene, realtime (RT on) as Render (main
    repo, 2026-10-28): Modo's thread reads it, and holds still meanwhile. The methods and constant from the repo."""
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    # READING comes with scene_cache, which takes the repo's lines from STOP_TIP to PREPARATIONS.
    head = """    def _capture(self,reuse=False,transform_items=None):
        import modo
        if modo.Scene().renderItem.id != self._scene_id:
            self._scene_id=modo.Scene().renderItem.id;self._geometry_cache=None
"""
    a = source.index('    def _reading(self):')
    b = source.index('        if not reuse or self._geometry_cache is None:self._reading()\n') + len('        if not reuse or self._geometry_cache is None:self._reading()\n')
    p.rep(head, source[a:b])
    p.rep("""        scene=self._configure_snapshot(capture_current(self._settings_values()['surface']==2))
        # Stepping through the shutter""", """        self._reading()
        scene=self._configure_snapshot(capture_current(self._settings_values()['surface']==2))
        # Stepping through the shutter""")
    p.save()


def lock_free_pack(kit):
    """MoonLightIPR's pack lets Python's lock go where it walked a heavy mesh's numbers in one call (main repo,
    uncommitted 2026-10-28): each corner's point gathered, and the finite checks, by the adapter (MR_gather, MR_finite;
    rows.py and simplify.py are copied whole). Joined in Python, a car's pack held Modo's own thread 11 to 29 s."""
    source = (REPO_KIT / 'moonlightipr_scene.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(kit / 'python' / 'moonray_modo' / 'moonlightipr_scene.py')
    a, b = p.text.index('def _by_corner(rows, corners):'), source.index('def _finite(values):')
    end = p.text.index('\n\n\n', a) + 3
    p.text = p.text[:a] + source[b:source.index('\n\n\n', source.index('def _by_corner(rows, corners):')) + 3] + p.text[end:]
    p.rep("""    if not math.isfinite(sum(positions)) or not math.isfinite(sum(shading)) or not math.isfinite(sum(moved)):""",
          """    if not all(_finite(values) for values in (positions, shading, moved)):""")
    p.rep("""    positions = array('f', chain.from_iterable(vertices))
    if not math.isfinite(sum(positions)):""", """    positions = _numbers(vertices, 3)
    if not _finite(positions):""")
    p.save()


def xpu2_rt_no_denoise(kit):
    """MoonRay's realtime under XPU2 has no Denoised Beauty (asked 2026-10-28: its denoised pictures looked bad): the
    buffer is hidden from the list while it follows the scene, Beauty shown where it was chosen, and the denoiser's guides
    are not rendered for it (two of its three outputs at every edit)."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'denoising.py')
    p.rep("""def guides_required(snapshot,final=False,linear=False):
    if snapshot.get('_ipr') and not final:return False
""", """def guides_required(snapshot,final=False,linear=False):
    if snapshot.get('_ipr') and not final:return False
    # MoonRay's realtime under XPU2 has no denoised beauty (panel): its guides only for Lower Resolution While Moving,
    # whose pictures from a quarter of their pixels XPU2 denoises and upscales while the view or the scene moves.
    if snapshot.get('_realtime') and not final:
        try:
            from . import native
            if native.xpu2():return bool(native.xpu2_upscale())
        except Exception:pass
""")
    # Nor is it denoised: with Denoise Preview on, the denoiser asked for the guides it no longer has
    # ("Render unavailable: '_denoise_guides'").
    p.rep("""def enabled(snapshot,final=False,linear=False):
    if snapshot.get('_ipr') and not final:return False
""", """def enabled(snapshot,final=False,linear=False):
    if snapshot.get('_ipr') and not final:return False
    # MoonRay's realtime under XPU2 is not denoised (its denoised pictures looked bad), and has no guides for it.
    if snapshot.get('_realtime') and not final:
        try:
            from . import native
            if native.xpu2():return False
        except Exception:pass
""")
    p.save()
    p = Patch(py / 'panel.py')
    p.rep("""        if engine=='moonray' and self._following and not output:scene=dict(scene,_realtime=True)
""", """        if engine=='moonray' and self._following and not output:scene=dict(scene,_realtime=True)
        if not output:
            from . import native
            self._offer_denoised(not (engine=='moonray' and self._following and native.xpu2()))
""")
    p.rep("""    def _show_planned(self):""", """    def _offer_denoised(self,offered):
        \"\"\"Denoised Beauty in the buffer list, or not: MoonRay's realtime under XPU2 has none (its denoised pictures
        looked bad), and shows Beauty where it was chosen.\"\"\"
        index=self.buffer.findData('denoised_beauty')
        if index<0:return
        view=self.buffer.view()
        if hasattr(view,'setRowHidden'):view.setRowHidden(index,not offered)
        if not offered and self.buffer.currentData()=='denoised_beauty':self.buffer.setCurrentIndex(self.buffer.findData('beauty'))

    def _show_planned(self):""")
    p.save()


def third_party(kit):
    """The kit's third-party notices as the main repo's kit has them: THIRD_PARTY.txt (meshoptimizer cited, 2026-10-28)
    and licenses/, meshoptimizer's MIT notice among them (the adapter and XPU2's session compile it in)."""
    shutil.copyfile(REPO_KIT.parents[1] / 'THIRD_PARTY.txt', kit / 'THIRD_PARTY.txt')
    (kit / 'licenses').mkdir(exist_ok=True)
    for license in (REPO_KIT.parents[1] / 'licenses').glob('*'):
        shutil.copyfile(license, kit / 'licenses' / license.name)


def scene_cache(kit):
    """The Scene cache (main repo, uncommitted 2026-10-28): cache_folder.py copied again (cache_folder copied it before
    it held the Scene cache, and copies it still); MoonLightIPR's heavy packed meshes and MoonRay's realtime simplified
    ones kept in the cache folder (moonlightipr_scene.py and simplify.py, copied whole by geometry_detail); the Scene
    cache switch in the preferences; a preparation said as Caching scene... while it is on; and XPU2's tessellation
    cache following the switch (XPU2_SESSION_GEOMETRY_CACHE)."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'cache_folder.py', py / 'cache_folder.py')
    source = (REPO_KIT / 'preferences.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'preferences.py')
    p.rep("""'cache_folder': ('', str),""", """'cache_folder': ('', str), 'scene_cache': (True, bool),""")
    p.rep("""        form.addRow('Cache folder', cache_row)
""", source[source.index("        form.addRow('Cache folder', cache_row)\n"):source.index("        form.addRow(check('scene_cache'")] +
          source[source.index("        form.addRow(check('scene_cache'"):source.index("what it kept.'))\n") + len("what it kept.'))\n")])
    p.save()
    source = (REPO_KIT / 'moonlightipr_session.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'moonlightipr_session.py')
    p.rep("""class Packer(QtCore.QThread):""", source[source.index('# What the preview says while a scene is made ready'):source.index('class Packer(QtCore.QThread):')] + 'class Packer(QtCore.QThread):')
    p.rep("""        self.packer.finished.connect(self._packed)
        self.packer.start()
""", source[source.index("        self.packer.finished.connect(self._packed)\n"):source.index("        QtCore.QTimer.singleShot(1000, said)\n") + len("        QtCore.QTimer.singleShot(1000, said)\n")])
    p.save()
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'panel.py')
    # Caching scene... only while the cache keeps what it makes, Loading scene from cache... while it reads back (the
    # session says which, 2026-10-28): the panel no longer names every preparation Caching.
    p.rep("""STOP_TIP = 'Stop the render in progress and stop following scene changes.'
""", source[source.index("STOP_TIP = 'Stop the render in progress and stop following scene changes.'\n"):source.index('READING = "Reading Modo\'s scene graph..."\n') + len('READING = "Reading Modo\'s scene graph..."\n')])
    p.save()
    source = (REPO_KIT / 'moonlightipr_scene.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'moonlightipr_scene.py')
    p.rep("""_packed = {}
""", source[source.index('_packed = {}\n'):source.index('PACKED_FORM = 1\n') + len('PACKED_FORM = 1\n')])
    start = source.index("    # A heavy mesh, or one made lighter, as the Scene cache kept it (cache_folder)")
    p.rep("""                 (simplified['triangles'], round(simplified['error'], 6)) if simplified else None)
    cached = _packed.get(signature)
    if cached:
        return cached
""", """                 (simplified['triangles'], round(simplified['error'], 6)) if simplified else None)
    cached = _packed.get(signature)
    if cached:
        return cached
""" + source[start:source.index('            return held\n', start) + len('            return held\n')])
    tail = """    key = hashlib.blake2b(struct.pack('<I', flags) + payload, digest_size=8).digest()
    _packed[signature] = (key, flags, payload, struct.pack('<%di' % UV_SLOTS, *slot_map))
    return _packed[signature]"""
    at = p.text.rindex(tail)
    p.text = p.text[:at] + """    key = hashlib.blake2b(struct.pack('<I', flags) + payload, digest_size=8).digest()
    _packed[signature] = (key, flags, payload, struct.pack('<%di' % UV_SLOTS, *slot_map))
    if lasting:
        from . import cache_folder
        cache_folder.keep('MoonLightIPR/meshes', ('mesh', PACKED_FORM, signature), _packed[signature])
    return _packed[signature]""" + p.text[at + len(tail):]
    p.save()
    # XPU2's tessellation cache follows the switch: its environment (the native.py text at the top of this builder).


def first_tick(kit):
    """A preview's first reading notes the moment and settings it was read at (main repo, uncommitted 2026-10-28):
    following, the first tick found none, took the time for changed, and read the whole scene again with nothing edited
    (the car's 41 s on Modo's thread just after its first picture, MoonLightIPR and MoonRay's realtime alike)."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""                # What Modo has told of up to now is in this reading of the scene.
                self.changes.consume()
            self._submit(scene)""", """                # What Modo has told of up to now is in this reading of the scene.
                self.changes.consume()
                # And the moment and settings it was read at: following, the first tick found none, took the time for
                # changed, and read the whole scene again with nothing edited (a car scene's 41 s on Modo's thread, just
                # after its first picture; MoonRay's realtime the same).
                self._last_time=lx.service.Selection().GetTime();self._settled()
            self._submit(scene)""")
    p.save()


def cache_folder(kit):
    """The Cache folder in the preview's preferences (main repo, uncommitted 2026-10-28): the kit's caches under it
    (cache_folder.py copied from the repo; each cache's place rewritten by cache_sites.py beside this builder), and the
    renderers told of it (MOONRAY_MODO_CACHE; XPU2 keeps its tessellations there, cpu_scene.h geometryCacheFolder)."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'cache_folder.py', py / 'cache_folder.py')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cache_sites
    cache_sites.apply(py)
    source = (REPO_KIT / 'native.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'native.py')
    p.rep("""    env['TMPDIR'] = tempfile.gettempdir()
""", source[source.index("    env['TMPDIR'] = tempfile.gettempdir()\n"):source.index("    env['MOONRAY_MODO_CACHE'] = str(cache_root())\n") + len("    env['MOONRAY_MODO_CACHE'] = str(cache_root())\n")])
    p.save()
    source = (REPO_KIT / 'preferences.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'preferences.py')
    p.rep("""    'runtime': ('', str), 'preview_engine'""", """    'runtime': ('', str), 'cache_folder': ('', str), 'preview_engine'""")
    p.rep("""        form.addRow('MoonRay folder', row)
""", source[source.index("        form.addRow('MoonRay folder', row)\n"):source.index("        form.addRow('Cache folder', cache_row)\n") + len("        form.addRow('Cache folder', cache_row)\n")])
    p.rep("""    def accept(self):
        self.preferences.set('runtime', self.runtime.text().strip())
""", source[source.index("    def _browse_cache(self):\n"):source.index("        forget()\n") + len("        forget()\n")])
    p.save()


def withdrawn_frames(kit):
    """A frame the renderer has already taken back (shared_image.Withdrawn: unread for five seconds while Modo's thread
    was busy, a fast drag of the car) is passed over without a word, not said as "frame skipped" (main repo,
    uncommitted 2026-10-28). shared_image.py is copied from the repo whole."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'shared_image.py', py / 'shared_image.py')
    p = Patch(py / 'moonlightipr_session.py')
    p.rep("""                except (ValueError, OSError) as exc: self.status.emit('MoonLightIPR frame skipped: ' + str(exc))""",
          """                except (ValueError, OSError) as exc:
                    # A frame already taken back (shared_image.Withdrawn) is passed over without a word: newer ones follow.
                    from .shared_image import Withdrawn
                    if not isinstance(exc, Withdrawn): self.status.emit('MoonLightIPR frame skipped: ' + str(exc))""")
    p.save()
    p = Patch(py / 'persistent.py')
    p.rep("""                except (ValueError,OSError) as exc:self.status.emit('Progressive image skipped: '+str(exc))""",
          """                except (ValueError,OSError) as exc:
                    # A frame already taken back (shared_image.Withdrawn) is passed over without a word: newer ones follow.
                    from .shared_image import Withdrawn
                    if not isinstance(exc,Withdrawn):self.status.emit('Progressive image skipped: '+str(exc))""")
    p.save()


def realtime_outputs(kit):
    """MoonRay's realtime renders the beauty and the buffer looked at, not every buffer (outputs.preview), another
    buffer chosen meanwhile rendered from the scene as last read; and a delta made against a scene since replaced is made
    again against the one applied where the text is small, not sent as a full reload (persistent._dispatch). Main repo,
    uncommitted 2026-10-28: every camera move of XPU2 realtime renamed every output, and prepared the scene again."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'outputs.py')
    p.rep("""    for entity in entities.display_filters(scene):result[entities.filter_key(entity)]=entities.filter_output(entity)
    return result""", """    for entity in entities.display_filters(scene):result[entities.filter_key(entity)]=entities.filter_output(entity)
    # MoonRay's realtime (a preview following the scene, _geometry_detail): the beauty, and the buffer looked at where it
    # is another, not every buffer at every edit (each was rendered, and each edit renamed their files, so that none was
    # the camera's alone). The composite is made of the others, which it keeps.
    chosen=scene.get('preview_buffer','beauty')
    if scene.get('_geometry_detail') and chosen!='composite':
        return {key:value for key,value in result.items() if key in ('beauty',chosen)}
    return result""")
    p.save()
    p = Patch(py / 'panel.py')
    p.rep("""            self.renderer.buffers.select(key,values(stored['display']))
""", """            self.renderer.buffers.select(key,values(stored['display']))
            if self._following and self._engine()=='moonray' and key not in ('beauty','denoised_beauty') and self._ipr_refine_scene is not None:
                # MoonRay's realtime renders the beauty and the buffer looked at (outputs.preview): another is rendered
                # from the scene as last read, not read again.
                self._submit(dict(self._ipr_refine_scene,preview_buffer=key))
""")
    p.save()
    source = (REPO_KIT / 'persistent.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    p = Patch(py / 'persistent.py')
    p.rep("""SHARED_LIMIT=256*1024*1024
""", source[source.index('SHARED_LIMIT=256*1024*1024\n'):source.index('DIFF_HERE=4*1024*1024\n') + len('DIFF_HERE=4*1024*1024\n')])
    p.rep("""        if prepared is not None:
            # APPLIED may advance while the writer runs. A stale delta is unsafe; use its full scene.
            delta=prepared['delta'] if prepared['base'] is self.applied else None
        else:delta=difference(self.applied['text'],request['text']) if self.applied else None
""", source[source.index("        delta_bytes=(prepared or {}).get('delta_bytes')\n"):source.index("        full=self._write_scene(request,'full',request['text']")])
    p.rep("""        path=self._write_scene(request,'delta',delta,(prepared or {}).get('delta_bytes')) if delta is not None else full""",
          """        path=self._write_scene(request,'delta',delta,delta_bytes) if delta is not None else full""")
    p.save()


def motion_default(kit):
    """MoonLightIPR's motion blur off by default, a choice stored before 2026-10-27 let go once, and Motion Blur said at
    the foot of the picture while it is on (main repo preferences.py and panel.py, uncommitted)."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'preferences.py')
    p.rep("""'preview_motion': (True, bool),""", """'preview_motion': (False, bool),""")
    p.rep("""                          'Render reads the scene at shutter open and close; IPR updates stay sharp'))""",
          """                          'Off by default: each update then also reads the scene at shutter open and close, which adds to its wait. '
                          'While dragging, updates stay sharp. Said as Motion Blur at the foot of the picture while on'))""")
    p.rep("""            self.store.setValue('runtime_installation', installation)
""", """            self.store.setValue('runtime_installation', installation)
        # Motion blur in MoonLightIPR is off by default since 2026-10-27 (it adds to each update's wait): a choice stored
        # before then is let go once, and one made since is kept.
        if self.store.value('preview_motion_default') != 'off':
            self.store.remove('preview_motion')
            self.store.setValue('preview_motion_default', 'off')
""")
    p.save()
    p = Patch(py / 'panel.py')
    p.rep("""            values=self._settings_values()
            if self.preview_engine.currentData()=='moonray' and values['denoising'].get('safe')""",
          """            values=self._settings_values()
            # MoonLightIPR's motion blur, off by default: each update then reads the shutter's ends too.
            if self.preview_engine.currentData()=='moonlightipr' and self.preferences.get('preview_motion'):words.append('Motion Blur')
            if self.preview_engine.currentData()=='moonray' and values['denoising'].get('safe')""")
    p.save()


def render_reading(kit):
    """The Render button greyed, with "Reading Modo's scene graph..." in the status line, while the scene is read and
    until the render is on its way (main repo panel.py, start_render, uncommitted 2026-10-27)."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("""        self._clear_first=True
        self.render_once()
""", """        self._clear_first=True
        # Reading the scene holds Modo's thread, for a minute on a heavy scene: it is said before it begins, and the
        # button is greyed until the render is on its way and Stop can stop it.
        reading=not self._live()
        if reading:
            self.status.setText("Reading Modo's scene graph...");self.status.setToolTip('')
            self.start.setEnabled(False)
            self.status.repaint();self.start.repaint()
        try:self.render_once()
        finally:
            # A render on its way has set the button already; one that never began gives it back.
            if reading and not self._rendering:self.start.setEnabled(True)
""")
    p.save()


def geometry_detail(kit):
    """Realtime previews' geometry by size on screen (the main kit, 2026-10-27; moonlightipr_detail.py, simplify.py):
    MoonLightIPR cuts its subdivided and displaced meshes and simplifies dense ones; MoonRay's realtime (XPU2's here)
    has MoonRay tessellate subdivided and displaced meshes to the cut's pixels from a held dicing camera, and is given
    dense meshes simplified; coarser after an edit, finer once the scene is left alone a second. The modules are taken
    from the repo whole; moonlightipr_scene.py, rdla.py and panel.py patched as the repo has them; the adapter with
    MR_simplify (meshoptimizer) built by tools/build_modo_bridge.py into build/modo-bridge-detail."""
    py = kit / 'python' / 'moonray_modo'
    for name in ('moonlightipr_detail.py', 'simplify.py'):
        shutil.copy2(REPO_KIT / name, py / name)
    shutil.copy2(REPO_KIT.parents[3] / 'build' / 'modo-bridge-detail' / 'MoonRayGeometry.lx', kit / 'bin' / 'MoonRayGeometry.lx')
    p = Patch(py / 'moonlightipr_scene.py')
    p.rep("""def mesh_payload(mesh, material_index, slots):""", """def mesh_payload(mesh, material_index, slots, simplified=None):""")
    p.rep("""                 tuple((slot, content(values)) for slot, values in sets), content(closing) if closing is not None else '')
    cached = _packed.get(signature)""", """                 tuple((slot, content(values)) for slot, values in sets), content(closing) if closing is not None else '',
                 (simplified['triangles'], round(simplified['error'], 6)) if simplified else None)
    cached = _packed.get(signature)""")
    p.rep("""    flags = ((MESH_HAS_NORMALS if normals else MESH_SMOOTH if smooth else 0) | (MESH_HAS_MATERIAL_IDS if face_materials is not None else 0)""",
          """    if simplified:
        # Lighter, as far as it is small on screen: its normals the full mesh's (simplify.py).
        from . import simplify
        made = simplify.arrays(positions, indices, shading if normals else None, moved if moved else None, uvs,
                               material_ids if face_materials is not None else None, smooth, simplified['triangles'], simplified['error'])
        if made is not None:
            positions, given, moved, indices, uvs, material_ids, _ = made
            if len(given):
                shading, normals = given, True
    flags = ((MESH_HAS_NORMALS if normals else MESH_SMOOTH if smooth else 0) | (MESH_HAS_MATERIAL_IDS if face_materials is not None else 0)""")
    p.rep("""            key, flags, payload, slot_map = mesh_payload(mesh, material_index, compiler.slots)""",
          """            key, flags, payload, slot_map = mesh_payload(mesh, material_index, compiler.slots, cut.get('simplify') if cut else None)""")
    p.rep("""DISPLACED_TRIANGLES = 400000
""", """DISPLACED_TRIANGLES = 400000
# Whether the last scene packed with coarse detail had a mesh cut coarser than at rest (moonlightipr_detail).
LAST_REFINABLE = False
""")
    p.rep("""def displaced(mesh, key, flags, payload, materials, material_index, warnings):""",
          """def displaced(mesh, key, flags, payload, materials, material_index, warnings, limit=None):""")
    p.rep("""    while levels < 8 and triangles * 4 ** (levels + 1) <= DISPLACED_TRIANGLES:
        levels += 1""", """    # As many times as its budget allows, and no more than its size on screen needs (limit, moonlightipr_detail).
    while levels < 8 and triangles * 4 ** (levels + 1) <= DISPLACED_TRIANGLES and (limit is None or levels < limit):
        levels += 1""")
    p.rep("""    used, meshes, instances, order = set(), [], [], {}
    for mesh in list(geometry.render_meshes(scene.get('meshes', []))) + [fog for fog, _, _ in fogs]:
        if not mesh['faces']:
            continue""", """    used, meshes, instances, order = set(), [], [], {}
    drawn = list(geometry.render_meshes(scene.get('meshes', []))) + [fog for fog, _, _ in fogs]
    # How finely each subdivided and displaced mesh is cut: by its size on screen (moonlightipr_detail).
    from . import moonlightipr_detail
    from .moonlightipr_materials import displacement as displacement_program

    def displaces(mesh):
        for tag in set(mesh.get('face_materials') or [mesh.get('material', '')]):
            held = materials.get(tag if tag in materials else '')
            if held and displacement_program(held):
                return True
        return False
    global LAST_REFINABLE
    # Following the scene, its meshes are cut from a camera held through its edits (_detail_keep, the preview window):
    # cut from the camera itself, a move of it simplified and packed dense meshes anew (a car's 26 s, Modo held meanwhile).
    cutting = moonlightipr_detail.held_dicing(camera, drawn, bool(scene.get('_detail_keep'))) if scene.get('_geometry_detail') else camera
    cuts, LAST_REFINABLE = moonlightipr_detail.plan(drawn, cutting, width, scene.get('_geometry_detail', 'fine'), displaces)
    # The dense meshes made lighter side by side first; the loop below then finds them packed.
    dense = [(mesh, cuts[id(mesh)]['simplify']) for mesh in drawn
             if mesh['faces'] and not mesh.get('subdivision') and (cuts.get(id(mesh)) or {}).get('simplify')]
    if len(dense) > 1:
        import os
        from concurrent.futures import ThreadPoolExecutor

        def ahead(pair):
            try:
                mesh_payload(pair[0], material_index, compiler.slots, pair[1])
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=min(len(dense), os.cpu_count() or 4)) as pool:
            list(pool.map(ahead, dense))
    for mesh in drawn:
        if not mesh['faces']:
            continue
        cut = cuts.get(id(mesh))
        if cut is not None and mesh.get('subdivision'):
            mesh = dict(mesh, subdivision_level=cut['subdivision'])""")
    p.rep("""        key, flags, payload = displaced(mesh, key, flags, payload, materials, material_index, warnings)""",
          """        key, flags, payload = displaced(mesh, key, flags, payload, materials, material_index, warnings, cut['splits'] if cut else None)""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("""from array import array as numbers
""", """from array import array as numbers

# Whether the last realtime preview's scene had a mesh cut coarser than at rest (moonlightipr_detail).
LAST_REFINABLE = False
""")
    p.rep("""    scene = entities.mesh_lights(scene)
""", """    scene = entities.mesh_lights(scene)
    # A realtime preview cuts its meshes by their size on screen, from a held dicing camera (moonlightipr_detail).
    detail = scene.get('_geometry_detail')
    if detail and not scene.get('dicing_camera') and scene.get('camera'):
        from .moonlightipr_detail import held_dicing
        scene = dict(scene, dicing_camera=held_dicing(scene['camera'], scene.get('meshes', []), bool(scene.get('_detail_keep'))))
""")
    p.rep("""    resolution_cap = held_render['max_geometry_resolution'] if held_render['enable_max_geometry_resolution'] else None
    for index, mesh in enumerate(render_meshes):
        vertices = mesh['vertices']
        faces = mesh['faces']""", """    resolution_cap = held_render['max_geometry_resolution'] if held_render['enable_max_geometry_resolution'] else None
    # A realtime preview's cut of each mesh (moonlightipr_detail, simplify.py).
    cuts = {}
    if detail:
        from . import moonlightipr_detail
        global LAST_REFINABLE
        cuts, LAST_REFINABLE = moonlightipr_detail.plan(render_meshes, scene.get('dicing_camera') or scene['camera'], width, detail,
            lambda held: bool(({held.get('material', '')} | set(held.get('face_materials') or [])) & displacement_tags))
    for index, mesh in enumerate(render_meshes):
        cut = cuts.get(id(mesh))
        if cut and cut.get('simplify'):
            from . import simplify
            lighter = simplify.polygon_mesh(mesh, cut['simplify']['triangles'], cut['simplify']['error'])
            if lighter is not None:
                mesh = lighter
        if cut and mesh.get('subdivision') and not mesh.get('adaptive_error') and 'instances' not in mesh:
            mesh = dict(mesh, adaptive_error=cut['pixels'])
        elif mesh.get('subdivision') and not mesh.get('adaptive_error') and 'instances' not in mesh and held_render['displacement_tessellation']:
            # A subdivision surface with no screen error of its own follows the scene's Displacement Tessellation too:
            # Modo's Displacement Rate (or the Adaptive Error Override), as for a displaced polygon mesh.
            mesh = dict(mesh, adaptive_error=held_render['displacement_adaptive_error'] or float(scene.get('displacement_rate', 1.0)))
        vertices = mesh['vertices']
        faces = mesh['faces']""")
    p.rep("""            if settings['adaptive_error']:error=settings['adaptive_error']
            elif 'instances' in mesh or held_render['displacement_tessellation']==0:error=0.0""", """            if settings['adaptive_error']:error=settings['adaptive_error']
            elif cut and 'instances' not in mesh:error=cut['pixels']
            elif 'instances' in mesh or held_render['displacement_tessellation']==0:error=0.0""")
    p.save()
    p = Patch(py / 'panel.py')
    p.rep("""        self.refine_timer.timeout.connect(self._refine_ipr)
""", """        self.refine_timer.timeout.connect(self._refine_ipr)
        # MoonLightIPR's meshes cut finer once the scene has been left alone this long (moonlightipr_detail).
        self.detail_timer=QtCore.QTimer(self);self.detail_timer.setSingleShot(True);self.detail_timer.setInterval(1000)
        self.detail_timer.timeout.connect(self._refine_detail)
""")
    p.rep("""        if not output:scene=dict(scene,_clay_preview=self._clay())
""", """        if not output:scene=dict(scene,_clay_preview=self._clay())
        # A preview following the scene (MoonLightIPR, or MoonRay's realtime) cuts its meshes coarser after an edit and
        # finer once left alone. An edit that leaves the meshes as they were (the camera, a light, a material or where
        # items stand) and a refinement keep the cut as it stands: cut anew, every mesh would be built again twice for
        # each move.
        fine,self._detail_fine=getattr(self,'_detail_fine',False),False
        kept,self._detail_kept=getattr(self,'_detail_kept',False) or refining,False
        # The camera the meshes are cut from is held through every edit, and taken anew only at rest where the view came
        # far enough (_refine_detail: _detail_recut).
        recut,self._detail_recut=getattr(self,'_detail_recut',False),False
        if not output and self._following and self._engine()=='moonray':
            # MoonRay's realtime: the fine cut from the first, no coarse one before it (a cut of other meshes is a reload
            # of the scene: the coarse pass cost a second preparation of a heavy scene); through a move of the camera
            # its dicing camera is held whatever the view (_detail_keep), and looked at again once the view rests
            # (_refine_detail), the meshes cut anew only where it came far enough.
            self._detail_now='fine'
            scene=dict(scene,_geometry_detail='fine',_detail_keep=not recut)
            self.detail_timer.start()
        elif not output and self._following:
            now=getattr(self,'_detail_now',None)
            detail=now if kept and now else 'fine' if fine else 'coarse'
            self._detail_now=detail
            scene=dict(scene,_geometry_detail=detail,_detail_keep=not recut)
            if not (kept and self.detail_timer.isActive()):self.detail_timer.start()
        else:self.detail_timer.stop()
""")
    p.rep("""            if not blurred:
                scene = self._capture(reuse,items if reuse=='transforms' else None)
""", """            if not blurred:
                scene = self._capture(reuse,items if reuse=='transforms' else None)
            # The meshes as they were: their cut is kept (_submit).
            self._detail_kept=not full and not changed and (reuse is True or reuse in ('materials','transforms'))
""")
    p.rep("""    def _refine_ipr(self):""", """    def _refine_detail(self):
        \"\"\"The scene left alone: its meshes cut finer, where the coarse cut left any coarser.\"\"\"
        if self.disposed or not self._following or self.preview_lock.isChecked() or self._output_busy():return
        from .interaction import dragging
        if dragging() or self.release_timer.isActive():self.detail_timer.start();return
        from . import moonlightipr_scene
        from .moonlightipr_detail import rested
        scene=self._ipr_refine_scene
        if scene is None:return
        # A scene still being made ready is waited for: what its cut left coarser is known once it is (LAST_REFINABLE),
        # and before, the refinement was missed until the next edit, then made in the middle of it.
        if getattr(getattr(self.renderer,'moonlightipr',None),'packer',None) is not None:self.detail_timer.start();return
        # At rest: the meshes cut anew, once, where the view came far enough from the camera held through its moves
        # (moonlightipr_detail.rested); MoonLightIPR's coarse cut made fine besides.
        moved=not scene.get('dicing_camera') and rested(scene.get('camera') or {},scene.get('meshes',[]),60.0 if self._engine()=='moonray' else None)
        refinable=self._engine()=='moonlightipr' and moonlightipr_scene.LAST_REFINABLE
        if not (moved or refinable):return
        self._detail_fine=True;self._detail_recut=moved
        try:self._submit(scene)
        except Exception as exc:self._failed(str(exc))

    def _refine_ipr(self):""")
    p.rep("""    def stop(self):
        self._draft_quiet=False
        self.refine_timer.stop()
""", """    def stop(self):
        self._draft_quiet=False
        self.refine_timer.stop()
        self.detail_timer.stop();self._detail_fine=False;self._detail_now=None
""")
    p.save()


def build(session, base):
    base, session = Path(base).resolve(), Path(session).resolve()
    if any((base / m).exists() for m in MARKERS):
        raise SystemExit('The base kit is an experimental one; give the release kit with --base')
    report = json.loads((session / 'build.json').read_text(encoding='utf-8'))
    if report.get('status') != 'passed':
        raise SystemExit('The session build did not pass')
    if KIT.exists():
        shutil.rmtree(KIT)
    shutil.copytree(base, KIT, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'runtime-xpu2'))
    shutil.copytree(session / 'runtime-xpu2', KIT / 'runtime-xpu2')
    py = KIT / 'python' / 'moonray_modo'
    denoiser(KIT)
    generations(KIT)
    realtime(KIT)
    denoised_buffer(KIT)
    engines(KIT)
    repo_fixes(KIT)
    displacement_choice(KIT)
    geometry_rows(KIT)
    xpu2_smart_render(KIT)
    # meter_switch(KIT): no longer; the meter is switched in Preview Preferences now (memory_in_picture).
    motion_skip(KIT)
    geometry_detail(KIT)
    render_reading(KIT)
    shutter_reading(KIT)
    edited_surfaces(KIT)
    motion_default(KIT)
    realtime_outputs(KIT)
    xpu2_sampling_words(KIT)
    withdrawn_frames(KIT)
    cache_folder(KIT)
    first_tick(KIT)
    scene_cache(KIT)
    third_party(KIT)
    xpu2_rt_no_denoise(KIT)
    lock_free_pack(KIT)
    reading_status(KIT)
    scene_read_cache(KIT)
    ear_clipping(KIT)
    kit_version(KIT)
    hdri_and_texture_shape(KIT)
    lights_seen_by_camera(KIT)
    flowing_toolbar(KIT)
    tools_ngons_clear_cache(KIT)

    p = Patch(py / 'preferences.py')
    p.rep("""    'auto_xpu': (True, bool),""", """    'auto_xpu': (True, bool),
    # XPU2: what renders on the graphics card, and where its adaptive passes are made.
    'xpu2/renderer': ('xpu2', str), 'xpu2/adaptive_chain': (False, bool), 'xpu2/devices': ('both', str), 'xpu2/upscale': (True, bool),""")
    p.save()

    p = Patch(py / 'native.py')
    p.rep("""        globals()[key]() for key in ('usd_specular', 'opaque_alpha', 'transmission_roughness', 'auto_xpu'))""",
          """        globals()[key]() for key in ('usd_specular', 'opaque_alpha', 'transmission_roughness', 'auto_xpu')) + (xpu2(), xpu2_chain(), xpu2_devices(), xpu2_upscale())""")
    p.rep("""        if 'falling back to cpu' in line: result='Vector active (CPU fallback — see Render Log)'""",
          """        if 'falling back to cpu' in line: result='Vector active (CPU fallback — see Render Log)'
        if 'xpu2 preview session' in line: result='XPU2 active (%s + %s)'%(_gpu() or 'NVIDIA GPU',_cpu() or 'CPU')""")
    p.text = p.text.rstrip('\n') + '\n' + NATIVE
    p.save()

    p = Patch(py / 'persistent.py')
    p.rep("""        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(self.root))
        self.process.setProgram(str(request['runtime']/'moonray.exe'))""",
          """        # XPU2 renders in MoonRay's place where it is chosen, the render's execution mode is
        # XPU or Auto, and XPU2 has not left this scene to MoonRay (@@MODO_XPU2 FALLBACK).
        program=request['runtime']/'moonray.exe'
        self.xpu2_active=bool(native.xpu2() and request['mode'] in ('xpu','auto') and not getattr(self,'xpu2_refused',None))
        if self.xpu2_active and native.xpu2_devices()!='cpu' and not native.xpu2_shaders_compiled():
            # XPU2 chosen, its shaders not compiled: nothing is rendered (MoonRay XPU in its place hid that it was not XPU2),
            # and the preview says where to compile them.
            self.sent=None;self.failed.emit(native.XPU2_NOT_COMPILED);return
        # The view again, now that it is known whose: Denoised Beauty is XPU2's own (realtime), the beauty MoonRay's.
        self.select_view(self.view)
        if self.xpu2_active:
            program=native.xpu2_runtime()/'moonray_xpu2.exe'
            for key,value in native.xpu2_environment(request['runtime']).items():env.insert(key,value)
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(self.root))
        self.process.setProgram(str(program))""")
    p.rep("""            if not line.startswith('@@MODO_SESSION '):continue""",
          """            if line.startswith('@@MODO_XPU2 FALLBACK'):
                # XPU2 does not render this scene: MoonRay is started in its place with the scene, as after a restart.
                self.xpu2_refused=line[len('@@MODO_XPU2 FALLBACK'):].strip() or 'it does not render this scene'
                self.status.emit('MoonRay XPU2: %s. MoonRay XPU renders this preview instead.'%self.xpu2_refused)
                if not self.closed:self.stopping=True
                continue
            if not line.startswith('@@MODO_SESSION '):continue""")
    p.rep("""        if self.running() and signature!=self.signature:
            self.stopping=True;self.process.kill();return""",
          """        if self.running() and signature!=self.signature:
            # A preference changed (XPU2's among them): XPU2 may try again.
            self.xpu2_refused=None
            self.stopping=True;self.process.kill();return""")
    p.rep("""    def _exit(self,code,status):
        self.idle.cancel()
        self._read()""",
          """    def _exit(self,code,status):
        self.idle.cancel()
        self._read()
        if getattr(self,'xpu2_active',False) and not self.stopping and not self.closed:
            # XPU2 ended without saying why: MoonRay takes the preview over.
            self.xpu2_refused='ended unexpectedly (0x%08X; see Render Log)'%(code&0xffffffff)
            self.status.emit('MoonRay XPU2 %s. MoonRay XPU renders this preview instead.'%self.xpu2_refused)
            self.stopping=True""")
    p.rep("""    def stop(self):
        self.idle.cancel()""",
          """    def stop(self):
        self.idle.cancel()
        self.xpu2_refused=None""")
    p.save()
    binary_memory(KIT)

    # Where XPU2 renders the preview: the snapshot says so.
    p = Patch(py / 'render.py')
    p.rep("""        snapshot = dict(request['snapshot'],preview_buffer=self.buffer_key)
""", """        snapshot = dict(request['snapshot'],preview_buffer=self.buffer_key)
        # XPU2 renders it where it is chosen, the mode is XPU or Auto and it has not left this
        # session to MoonRay.
        snapshot['_xpu2']=bool(request.get('persistent') and mode in ('xpu','auto') and native.xpu2() and not getattr(self.session,'xpu2_refused',None))
""")
    # XPU2 tells of the GPU's cells apart from the CPU's workers (@@MODO_GPU_WORKERS): they reach the view through the
    # workers' signal, marked 'gpu', and are drawn in a colour of their own beside the CPU's.
    p.rep("""                    (self.workers if line.startswith('@@MODO_WORKERS ') else self.buckets).emit(packet)""",
          """                    if line.startswith('@@MODO_GPU_WORKERS '):self.workers.emit(('gpu',)+tuple(packet))
                    else:(self.workers if line.startswith('@@MODO_WORKERS ') else self.buckets).emit(packet)""")
    p.save()
    p = Patch(py / 'buckets.py')
    p.rep("""parts[0] not in ('@@MODO_TILES','@@MODO_WORKERS')""", """parts[0] not in ('@@MODO_TILES','@@MODO_WORKERS','@@MODO_GPU_WORKERS')""")
    p.save()
    p = Patch(py / 'viewer.py')
    p.rep("""        self.workers=None
        self.show_workers=True""", """        self.workers=None
        self.gpu_workers=None  # XPU2: the cells the GPU's launches hold
        self.show_workers=True""")
    p.rep('''    def _note_outlines(self):
        """Take in what MoonRay last told of: each outline is new, or has been seen again just now."""
        told=self._outlines()''', '''    def _note_outlines(self,told=None,kind='cpu'):
        """Take in what MoonRay last told of: each outline is new, or has been seen again just now. XPU2's GPU cells
        (kind 'gpu') are told of apart from the rest."""
        told=told if told is not None else self._outlines()''')
    p.rep("""        hold=self.HOLD_WORKER if told is self.workers else self.HOLD
        for rectangle in rectangles:
            key=(width,height,rectangle)""", """        hold=self.HOLD_WORKER if told is self.workers or kind=='gpu' else self.HOLD
        for rectangle in rectangles:
            key=(width,height,rectangle,kind)""")
    p.rep("""        if packet is None:self.workers=None""", """        if packet is None:self.workers=self.gpu_workers=None""")
    p.rep("""    def set_workers(self,packet):
        self.workers=packet""", """    def set_workers(self,packet):
        if packet and packet[0]=='gpu':
            # XPU2: the GPU's cells, beside the CPU's workers.
            self.gpu_workers=packet[1:]
            if self.show_workers:self._note_outlines(self.gpu_workers,'gpu')
            return
        self.workers=packet""")
    p.rep("""drawn=[(width,height,rectangle,min(1.0,""", """drawn=[(width,height,rectangle,kind,min(1.0,""")
    p.rep("""for (width,height,rectangle),(first,last,hold) in self._fading.items()]""",
          """for (width,height,rectangle,kind),(first,last,hold) in self._fading.items()]""")
    p.rep("""for width,height,(x0,y0,x1,y1),strength in drawn:""", """for width,height,(x0,y0,x1,y1),kind,strength in drawn:""")
    # The GPU's a pastel violet beside the CPU's pastel blue.
    p.rep("""colour=QtGui.QColor('#a9cdf2');""", """colour=QtGui.QColor('#bfaef5' if kind=='gpu' else '#a9cdf2');""")
    p.save()
    # XPU2 renders the preview's outputs as MoonRay would (its AOVs), the denoiser's guides among them (albedo and normal
    # match MoonRay's), drains on Stop and Smart Render, and saves checkpoints for Safe Denoising: the kit's own
    # denoising.py, outputs.py and Stop paths serve it as they serve MoonRay.

    p = Patch(KIT / 'lxserv' / 'moonray_preferences.py')
    p.rep("""lx.bless(switch('auto_xpu'),'moonray.pref.autoXpu')""",
          """lx.bless(switch('auto_xpu'),'moonray.pref.autoXpu')
lx.bless(choice('xpu2/renderer',(('xpu','MoonRay XPU'),('xpu2','XPU2'))),'moonray.pref.xpu2')
lx.bless(switch('xpu2/adaptive_chain'),'moonray.pref.xpu2Chain')
lx.bless(choice('xpu2/devices',(('both','Processor and graphics card'),('gpu','Graphics card only'),('cpu','Processor only'))),'moonray.pref.xpu2Devices')
lx.bless(switch('xpu2/upscale'),'moonray.pref.xpu2Upscale')""")
    p.save()
    (KIT / 'lxserv' / 'moonray_xpu2.py').write_text(COMMANDS, encoding='utf-8')

    p = Patch(KIT / 'preferences.cfg')
    p.rep("""        <atom type="Hash">MoonRaySourceOverrides:sheet</atom>
      </list>
""", """        <atom type="Hash">MoonRaySourceOverrides:sheet</atom>
      </list>
""" + SHEET_ENTRY)
    p.rep("""    </hash>
  </atom>
  <atom type="Filters">""", """    </hash>
""" + SHEET + """  </atom>
  <atom type="Filters">""")
    p.save()
    preference_pages(KIT)
    memory_in_picture(KIT)
    details_line(KIT)
    engine_names(KIT)
    about_xpu2(KIT)
    partial_moves(KIT)
    xpu2_status_words(KIT)
    realtime_cap(KIT)

    version = 'unknown'
    for line in (KIT / 'index.cfg').read_text(encoding='utf-8').splitlines():
        if 'version="' in line:
            version = line.split('version="', 1)[1].split('"', 1)[0]
    (KIT / 'XPU2-README.txt').write_text(README.format(version=version), encoding='utf-8')
    manifest = {'built': time.strftime('%Y-%m-%d %H:%M:%S'), 'base': str(base), 'session': str(session),
                'runtime_xpu2': json.loads((KIT / 'runtime-xpu2' / 'xpu2-runtime.json').read_text(encoding='utf-8'))}
    (HERE / 'kit-build.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('XPU2 kit made in', KIT)


def partial_moves(kit):
    """A move after the scene was read is never a whole reading again (main repo, 2026-10-29): any item moved has every
    mesh that stands with its item asked where it now is, instances among them; only a replicator's copies, meshes a
    deformer bends and meshes textured by a projection are read again (incremental.classify, standing). A car with
    instances read whole at every move of a part, 28 s. incremental.py from the repo whole (the kit's had nothing of
    its own left), and panel._capture as the repo has it."""
    py = kit / 'python' / 'moonray_modo'
    shutil.copyfile(REPO_KIT / 'incremental.py', py / 'incremental.py')
    source = (REPO_KIT / 'panel.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    start, end = source.index('    def _capture(self,reuse=False'), source.index('    def _capture_output(self):')
    p = Patch(py / 'panel.py')
    begin, finish = p.text.index('    def _capture(self,reuse=False'), p.text.index('    def _capture_output(self):')
    p.text = p.text[:begin] + source[start:end] + p.text[finish:]
    p.save()
    # A mesh edit reads its meshes from the live render cache where it heard the edit (the adapter's MR_geometry_live_*,
    # built into build/modo-geometry-live with the Shader Tree's stacks, MR_shader_stacks), the stacks' materials kept
    # while the Shader Tree is as it was (host.snapshot shading_kept); the live cache let go when the preview stops.
    shutil.copy2(REPO_KIT.parents[3] / 'build' / 'modo-geometry-live' / 'MoonRayGeometry.lx', kit / 'bin' / 'MoonRayGeometry.lx')
    p = Patch(py / 'host.py')
    p.rep("def snapshot(evaluated_geometry=False,reuse_geometry=None,refresh_materials=False,dirty_meshes=None):",
          "def snapshot(evaluated_geometry=False,reuse_geometry=None,refresh_materials=False,dirty_meshes=None,shading_kept=False):")
    p.rep("            result['materials'] = evaluated.assign_materials(data, scene, warnings)",
          "            result['materials'] = evaluated.assign_materials(data, scene, warnings, shading_kept=shading_kept)")
    p.save()
    # The scene writer after a move (2026-10-29): its encoded lists kept by a limit from the machine's memory, and what
    # is small of them (fingerprints, rdlb.py; the corners' check, rdla.py) kept apart: 6.5 s for a car's write, 0.85 s.
    shutil.copyfile(REPO_KIT / 'serialization.py', py / 'serialization.py')
    p = Patch(py / 'rdla.py')
    p.rep("        _kept('checked:%d' % len(vertices), faces, checked)",
          "        from .serialization import small\n        small('checked:%d' % len(vertices), faces, checked)")
    p.save()
    p = Patch(py / 'panel.py')
    p.rep("""        self._following=False;self.timer.stop()
        # A draft's region is not gone on to once its lines have been stopped.""", """        self._following=False;self.timer.stop()
        # The live render cache let go: Modo no longer keeps it up to date for a preview that does not follow the scene.
        try:
            from . import evaluated
            evaluated.stop_live()
        except Exception:pass
        # A draft's region is not gone on to once its lines have been stopped.""")
    p.save()


def realtime_cap(kit):
    """Realtime Adaptive Sampling Cap (MoonRay XPU2 only), Preferences > MoonRay > Preview Window: None, 30 or 60 samples
    a pixel at most while realtime follows the scene under XPU2, its average and its most a pixel alike, beside the
    scene's own Realtime Samples (the fewer of the two). Read when the scene is written (xpu2_sampling.realtime_cap).
    Asked 2026-10-30."""
    py = kit / 'python' / 'moonray_modo'
    p = Patch(py / 'preferences.py')
    p.rep("""'xpu2/upscale': (True, bool),""", """'xpu2/upscale': (True, bool), 'xpu2/rt_cap': ('none', str),""")
    p.save()
    p = Patch(py / 'native.py')
    p.rep("""def xpu2_upscale():""", '''def xpu2_rt_cap():
    """Realtime Adaptive Sampling Cap (Preferences > MoonRay > Preview Window): the most samples a pixel realtime takes
    under XPU2, 0 for none (the default)."""
    value = str(_setting('xpu2/rt_cap', 'none'))
    return int(value) if value in ('30', '60') else 0


def xpu2_upscale():''')
    p.save()
    p = Patch(kit / 'lxserv' / 'moonray_preferences.py')
    p.rep("""lx.bless(switch('xpu2/upscale'),'moonray.pref.xpu2Upscale')""", """lx.bless(switch('xpu2/upscale'),'moonray.pref.xpu2Upscale')
lx.bless(choice('xpu2/rt_cap',(('none','None'),('30','30'),('60','60'))),'moonray.pref.xpu2RealtimeCap')""")
    p.save()
    p = Patch(kit / 'preferences.cfg')
    p.rep("""        <atom type="Tooltip">How many pixels wide a draft's first, quick picture is. The fine picture that follows it is as fine as the view</atom>
      </list>
""", """        <atom type="Tooltip">How many pixels wide a draft's first, quick picture is. The fine picture that follows it is as fine as the view</atom>
      </list>
      <list type="Control" val="cmd moonray.pref.xpu2RealtimeCap ?">
        <atom type="Label">Realtime Adaptive Sampling Cap (MoonRay XPU2 only)</atom>
        <atom type="Tooltip">The most samples a pixel MoonRay realtime takes under XPU2, on average and at most, so that it settles sooner: None, 30 or 60. A scene's Realtime Samples still applies; the fewer of the two is taken. It shows with the next update of the preview</atom>
      </list>
""")
    p.save()
    p = Patch(py / 'rdla.py')
    p.rep("int(generation.get('most',0)),1 if generation.get('cpu',True) else 0,",
          "xpu2_sampling.most_samples(generation,samples,bool(scene.get('_realtime'))),1 if generation.get('cpu',True) else 0,")
    p.save()


def xpu2_status_words(kit):
    """While XPU2 renders a preview, the status says how XPU2 samples it, in the words of the picture's line
    (xpu2_sampling.words): it said MoonRay's "Adaptive rendering, 4-4096 SPP, error 15", which XPU2 does not follow
    (asked 2026-10-29). Final renders, MoonRay's, say MoonRay's as before."""
    p = Patch(kit / 'python' / 'moonray_modo' / 'render.py')
    p.rep("""            if snapshot.get('_ipr'):label='LS · '+label""", """            if snapshot.get('_ipr'):label='LS · '+label
            # XPU2 renders the preview: how XPU2 samples it, as the picture's line says it.
            from . import xpu2_sampling
            if not request.get('output') and mode in ('xpu','auto') and xpu2_sampling.active():
                try:label='Rendering · '+xpu2_sampling.words({'samples':self.sample_grid,'render':sampling,'xpu2_generations':snapshot.get('xpu2_generations') or {}},realtime=bool(snapshot.get('_realtime')))
                except Exception:pass""")
    p.save()
    # Realtime Samples (2026-10-29): XPU2 is given realtime's share of Generation Samples while RT follows the scene.
    p = Patch(kit / 'python' / 'moonray_modo' / 'rdla.py')
    p.rep("values=[2,1,int(generation.get('samples',0)),", "values=[2,1,xpu2_sampling.budget(generation,samples,bool(scene.get('_realtime'))),")
    p.save()
    p = Patch(kit / 'python' / 'moonray_modo' / 'panel.py')
    p.rep("try:return xpu2_sampling.words(self._settings_values())", "try:return xpu2_sampling.words(self._settings_values(),realtime=bool(getattr(self,'_following',False)))")
    p.save()


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--session', type=Path, required=True)
    a.add_argument('--base', type=Path, default=INSTALLED)
    args = a.parse_args()
    build(args.session, args.base)

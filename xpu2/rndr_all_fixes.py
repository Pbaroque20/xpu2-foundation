"""Every exact renderer saving together: xpu2/rndr_cpu_fixes.py (shade handler and
integration, switches on by default) and the per-lane random-walk subsurface
wavefront (xpu2/rndr_sss_wavefront.py, off unless XPU2_SSS_WAVEFRONT=1)."""
from . import rndr_cpu_fixes as cpu
from . import rndr_sss_wavefront as sss

FILES = cpu.FILES + sss.FILES
HEADERS = tuple(dict.fromkeys(cpu.HEADERS + sss.HEADERS))
NEW_SOURCES = cpu.NEW_SOURCES

if set(cpu.FILES) & set(sss.FILES):
    raise ValueError('overlapping patch modules')


def patch(sources):
    out = cpu.patch({f: sources[f] for f in cpu.FILES})
    out.update(sss.patch({f: sources[f] for f in sss.FILES}))
    return out

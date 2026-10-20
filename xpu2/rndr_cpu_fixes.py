"""All exact CPU savings together: xpu2/rndr_shade_fixes.py and xpu2/rndr_integ_fixes.py."""
from . import rndr_shade_fixes as shade
from . import rndr_integ_fixes as integ

FILES = shade.FILES + integ.FILES
HEADERS = tuple(dict.fromkeys(shade.HEADERS + integ.HEADERS))
NEW_SOURCES = integ.NEW_SOURCES


def patch(sources):
    out = shade.patch({f: sources[f] for f in shade.FILES})
    out.update(integ.patch({f: sources[f] for f in integ.FILES}))
    return out

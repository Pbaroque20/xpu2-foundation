"""xpu2/rndr_shade_fixes.py with the stage marks of xpu2/rndr_shade_profile.py on top."""
from . import rndr_shade_fixes as fixes
from . import rndr_shade_profile as profile

FILES = fixes.FILES
HEADERS = profile.HEADERS


def patch(sources):
    return profile.patch(fixes.patch(sources))

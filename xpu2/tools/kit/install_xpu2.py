"""Install the private XPU2 test kit in place of the MoonRay kit in Modo's Kits folder, or put the saved kit back.

    python install_xpu2.py install    # backs up the current kit, then copies this folder's MoonRayForModo in
    python install_xpu2.py restore    # puts the most recent backup back

Refuses while Modo or a renderer is running. Backups go next to this script (backups/kit-<time>), never into a
repository. The kit's preferences live in the registry and are shared; XPU2's own keys are new and ignored by the
stock kit, so restoring needs nothing else.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE / 'MoonRayForModo'
TARGET = Path(os.environ['APPDATA']) / 'Luxology' / 'Kits' / 'MoonRayForModo'
BACKUPS = HERE / 'backups'


def running():
    """Modo, and anything running from the kit's folder (its MoonRay, XPU2, MoonLightIPR, tools). Renderers run from
    elsewhere (test runtimes) do not hold the kit's files."""
    listed = subprocess.run(['powershell', '-NoProfile', '-Command',
                             'Get-Process | ForEach-Object { $_.ProcessName + "|" + $_.Path }'],
                            capture_output=True, text=True).stdout.splitlines()
    kit = str(TARGET).lower()
    busy = []
    for line in listed:
        name, _, path = line.partition('|')
        if name.lower() in ('modo', 'modo_cl') or (path and path.lower().startswith(kit)):
            busy.append(name + (' (' + path + ')' if path else ''))
    return busy


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('install', 'restore'):
        raise SystemExit(__doc__)
    busy = running()
    if busy:
        raise SystemExit('Close these first: ' + ', '.join(busy))
    if sys.argv[1] == 'install':
        if not (KIT / 'index.cfg').is_file() or not any((KIT / name).is_file() for name in ('XPU2-README.txt', 'XPU2-EXPERIMENTAL.txt')):
            raise SystemExit('No XPU2 kit next to this script (build_xpu2_kit.py makes it)')
        BACKUPS.mkdir(exist_ok=True)
        backup = BACKUPS / ('kit-' + time.strftime('%Y%m%d-%H%M%S'))
        if TARGET.exists():
            shutil.copytree(TARGET, backup)
            print('Current kit saved in', backup)
            shutil.rmtree(TARGET)
        shutil.copytree(KIT, TARGET, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print('XPU2 kit installed in', TARGET)
        print('In Modo: System > Preferences > MoonRay > XPU2: Compile XPU2 Shaders once, then GPU renderer > XPU2')
    else:
        saved = sorted(BACKUPS.glob('kit-*')) if BACKUPS.is_dir() else []
        if not saved:
            raise SystemExit('No backup to restore')
        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.copytree(saved[-1], TARGET)
        print('Restored', saved[-1], 'into', TARGET)


if __name__ == '__main__':
    main()

"""Add the built MoonLightIPR GPU preview to the installed kit, with a backup of what it replaces.

Copies the session, its device program and the CUDA runtime into the kit's runtime/moonlightipr
folder, and every file of the kit in this repository that differs from the installed one: the
Python modules, the commands, the forms and the assets. Nothing is named here one by one, so a
new module is installed without being added to a list. The installed layout.cfg is edited, not
replaced, and the installed bin and runtime folders are left to the steps below. A module that
is no longer in the repository is moved into the backup. Close Modo first; a running Modo keeps
the old modules loaded and the old session open.
"""
import csv
from datetime import datetime
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
source = root / 'kit/MoonRayForModo/python/moonray_modo'
kit = Path(os.environ['APPDATA']) / 'Luxology/Kits/MoonRayForModo'
target = kit / 'python/moonray_modo'
repository = root / 'kit/MoonRayForModo'
# Left alone: the layout is edited in place below, bin holds the adapter built for this machine, and the two notes
# are what an install writes about itself.
SKIPPED = ('layout.cfg', 'runtime.json', 'development-install.json')


def kit_files():
    """Every file of the repository's kit that an installed kit should hold as it is here, relative to the kit."""
    for path in sorted(repository.rglob('*')):
        name = path.relative_to(repository)
        if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in name.parts and name.parts[0] != 'bin' and name.as_posix() not in SKIPPED:
            yield name


def keep(path, name):
    """Put an installed file into the backup before it is replaced or removed."""
    (backup / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup / name)


if not target.is_dir():
    raise SystemExit('MoonRayForModo is not installed in ' + str(kit))
processes = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq modo.exe', '/FO', 'CSV', '/NH'],
                                    text=True, creationflags=subprocess.CREATE_NO_WINDOW)
if any(row and row[0].lower() == 'modo.exe' for row in csv.reader(io.StringIO(processes))):
    raise SystemExit('Close Modo before installing MoonLightIPR; nothing was changed')

backup = root / 'backups' / ('before-moonlightipr-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
if (kit / 'runtime/moonlightipr').is_dir():
    shutil.copytree(kit / 'runtime/moonlightipr', backup / 'runtime/moonlightipr')
subprocess.run([sys.executable, str(root / 'tools/stage_moonlightipr.py'), '--destination', str(kit / 'runtime/moonlightipr')], check=True)
copied = 0
for name in kit_files():
    held = kit / name
    if held.is_file() and held.read_bytes() == (repository / name).read_bytes():
        continue
    if held.is_file():
        keep(held, name)
    held.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, held)
    copied += 1
# A module the repository no longer has would still be imported, or loaded as a server, if it were left.
removed = 0
for folder in ('python/moonray_modo', 'lxserv'):
    for old in sorted((kit / folder).glob('*.py')):
        if not (repository / folder / old.name).is_file():
            keep(old, Path('no-longer-in-the-kit') / folder / old.name)
            old.unlink()
            removed += 1
# What the engine was called before it was MoonLightIPR: its modules and its folder are no longer read, and are moved
# into the backup rather than left beside the ones that are.
for old in [kit / 'runtime/moonlight']:
    if old.exists():
        held = backup / 'before-the-rename' / old.name
        held.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(held))
# The native geometry adapter (tools/build_modo_bridge.py --output-dir build/modo-geometry-<name> --geometry-only): the
# newest build that has every function the adapter's source in native-modo exports, as a release is given it. One
# fixed folder, built on 2026-10-30, went into every install after the adapter had gained hair, projections and fur.
# Modo must be closed: it holds the file open.
def current_adapter():
    import hashlib
    import json
    import re
    wanted = set()
    for path in (root / 'native-modo').glob('*.cpp'):
        wanted.update(re.findall(r'dllexport\)[^(;{]*?\b(MR_\w+)\s*\(', path.read_text(encoding='utf-8', errors='replace')))
    found = []
    for build in (root / 'build').glob('modo-geometry*/build.json'):
        plugin = build.with_name('MoonRayGeometry.lx')
        try:
            noted = json.loads(build.read_text(encoding='utf-8'))
            data = plugin.read_bytes()
        except (OSError, ValueError):
            continue
        if noted.get('geometry_only') and hashlib.sha256(data).hexdigest() == noted.get('plugin_sha256') \
                and all(name.encode() + b'\0' in data for name in wanted):
            found.append((plugin.stat().st_mtime, plugin))
    if not found:
        print('No geometry adapter built from the current native-modo source: the installed one is left as it is')
        return None
    return max(found)[1]


adapter = current_adapter()
if adapter is not None and (kit / 'bin').is_dir():
    held = kit / 'bin' / adapter.name
    if held.is_file():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(held, backup / adapter.name)
    shutil.copyfile(adapter, held)
# The one renderer library rebuilt with more room for a GPU render's outputs (patches/native-windows/cl1-pool-scale.patch),
# where it has been built into runtime/steady-0350-pool. It goes only into a runtime of the build it was made for.
pool = root / 'runtime/steady-0350-pool/librendering_mcrt_common.dll'
made_for = root / 'runtime/steady-0349-candidate/moonray.exe'
if pool.is_file() and made_for.is_file() and (kit / 'runtime/moonray.exe').is_file()         and (kit / 'runtime/moonray.exe').read_bytes() == made_for.read_bytes():
    held = kit / 'runtime' / pool.name
    if held.is_file() and held.read_bytes() != pool.read_bytes():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(held, backup / pool.name)
        shutil.copyfile(pool, held)
        print('Installed the renderer library with the larger output store')
# The scene reader behind Import RDL Scene, where it has been built (tools/build_rdl_reader.py).
reader = root / 'build/native-avx/bin/modo_rdl_import.exe'
if reader.is_file() and (kit / 'runtime').is_dir():
    held = kit / 'runtime' / reader.name
    if held.is_file():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(held, backup / reader.name)
    shutil.copyfile(reader, held)
# What grows hair, where it has been built (tools/build_hair_grower.py).
grower = root / 'build/native-avx/bin/modo_hair_grow.exe'
if grower.is_file() and (kit / 'runtime').is_dir():
    shutil.copyfile(grower, kit / 'runtime' / grower.name)
# What reads a VDB file for the preview, where it has been built (tools/build_vdb_grid.py).
grids = root / 'build/native-avx/bin/modo_vdb_grid.exe'
if grids.is_file() and (kit / 'runtime').is_dir():
    shutil.copyfile(grids, kit / 'runtime' / grids.name)
# The MoonRay menu: its MoonRay items submenu, dividers between its groups, and plain characters.
# The installed layout.cfg is edited in place, not replaced, so nothing else in it changes.
sys.path.insert(0, str(root / 'tools'))
import moonray_menu
layout = kit / 'layout.cfg'
text = layout.read_bytes().decode('utf-8')
tidied = moonray_menu.context_menus(moonray_menu.viewport_title(moonray_menu.playblast_entry(moonray_menu.rdl_entry(moonray_menu.curve_controls(moonray_menu.tidy(text))))))
if 'MoonRayEntityMenu' not in tidied:
    print('Could not find where to add "Add MoonRay Item" in', layout)
if tidied != text:
    backup.mkdir(parents=True, exist_ok=True)
    (backup / 'layout.cfg').write_bytes(text.encode('utf-8'))
    layout.write_bytes(tidied.encode('utf-8'))
print('Installed MoonLightIPR into', kit)
print('Kit files replaced or added: %d; modules no longer in the kit removed: %d' % (copied, removed))
print('Backup:', backup)

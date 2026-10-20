"""Stage a test runtime: hard links to a base runtime, with named files replaced by copies.

xpu2_stage_runtime.py --base <runtime> --output <folder> [--replace name=path ...]

The base is never written: replaced files are copied in after their links are
removed, and the base's inventory is checked before and after. The output gets
runtime/ and a report.json with the base, the replacements and their hashes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from xpu2.reference import digest_file, inventory, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replace', action='append', default=[], help='name=path')
    a = p.parse_args()
    base, out = a.base.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    held = inventory(base)
    runtime = out / 'runtime'
    for f in base.rglob('*'):
        q = runtime / f.relative_to(base)
        if f.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(f, q)
    replaced = {}
    for spec in a.replace:
        name, path = spec.split('=', 1)
        path = Path(path).resolve()
        target = runtime / name
        if not target.exists():
            raise SystemExit('Not in the base runtime: ' + name)
        target.unlink()
        shutil.copyfile(path, target)
        replaced[name] = dict(source=str(path), sha256=digest_file(target), was=held.get(name) if isinstance(held, dict) else None)
    verify(base, held)
    (out / 'report.json').write_text(json.dumps(dict(status='staged', base=str(base), replaced=replaced), indent=2) + '\n')
    print('staged', out, len(replaced), 'replaced')


if __name__ == '__main__':
    main()

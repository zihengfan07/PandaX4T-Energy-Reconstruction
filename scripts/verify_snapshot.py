"""Verify the public snapshot without needing detector data or dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.venv', 'venv', 'data', 'local_results'}
MANIFEST = ROOT / 'docs' / 'PUBLIC_MANIFEST.json'


def inventory():
    return [
        {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p != MANIFEST
        and not (set(p.relative_to(ROOT).parts) & SKIP)
        and p.suffix not in {'.pyc', '.pyo'}
    ]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Maintainer: generate a new public manifest')
    args = parser.parse_args()
    actual = inventory()
    if args.write:
        MANIFEST.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'Wrote manifest for {len(actual)} files.')
        return
    expected = json.loads(MANIFEST.read_text(encoding='utf-8'))
    a = {r['path']: r for r in actual}
    e = {r['path']: r for r in expected}
    errors = [p for p in sorted(a.keys() | e.keys()) if a.get(p) != e.get(p)]
    if errors:
        for p in errors:
            print('Missing, added or modified:', p)
        raise SystemExit(1)
    print(f'OK: {len(actual)} files match the public SHA-256 manifest.')


if __name__ == '__main__':
    main()

"""Check immutable evidence, Python syntax and local Markdown links without dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
from evidence import ROOT, load_rows


def check(root=ROOT):
    root = Path(root)
    errors = []
    provenance = json.loads((root/'results/provenance.json').read_text('utf-8'))
    for item in provenance:
        p = root/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Changed or missing original evidence: ' + item['path'])
    skip = {'.git', '__pycache__', '.venv', 'venv', 'local_results', 'data'}
    for p in root.rglob('*'):
        if not p.is_file() or set(p.relative_to(root).parts) & skip:
            continue
        if p.suffix == '.py':
            try:
                ast.parse(p.read_text('utf-8-sig'), filename=str(p))
            except SyntaxError as exc:
                errors.append(f'{p.relative_to(root)}: {exc}')
        if p.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', p.read_text('utf-8')):
                if '://' in target or target.startswith('#'):
                    continue
                q = p.parent/unquote(target.split('#')[0])
                if not q.exists():
                    errors.append(f'{p.relative_to(root)}: broken local link {target}')
    try:
        rows, periods = load_rows(root)
        if {r['sample'] for r in rows} != {'9563','9566','9698','all'}:
            errors.append('Unexpected archived run coverage')
        for row in rows:
            if not 0 < row['retained_bootstrap_iterations'] <= 150:
                errors.append('Invalid archived bootstrap count')
        for period in periods:
            if not 0 <= period['support']['s2_clip_fraction'] <= 1:
                errors.append('Invalid boundary fraction')
    except (KeyError, ValueError) as exc:
        errors.append('Invalid evidence schema: ' + str(exc))
    return errors, len(provenance)


if __name__ == '__main__':
    errors, count = check()
    for error in errors:
        print('ERROR:', error)
    if errors:
        raise SystemExit(1)
    print(f'OK: {count} original source/evidence hashes, result consistency, Python syntax and local document links.')

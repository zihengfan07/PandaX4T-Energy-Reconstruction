"""Run the preserved time-matched study with checked inputs and a fresh output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from check_inputs import InputError, inspect_file

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(kr, th, output, iterations=150, synthetic=False):
    kr, th, output = Path(kr).resolve(), Path(th).resolve(), Path(output).resolve()
    if iterations < 2:
        raise InputError('Bootstrap iterations must be at least 2')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InputError('Output must be absent or empty; existing results are never overwritten')
    inputs = {'kr': inspect_file(kr, 'kr'), 'th2615': inspect_file(th, 'th2615')}
    for key, path in [('kr', kr), ('th2615', th)]:
        inputs[key].update({'filename': path.name, 'sha256': sha256(path)})
    output.mkdir(parents=True, exist_ok=True)
    scripts = ROOT / 'analysis/v21'
    record = {
        'kind': 'synthetic_demonstration' if synthetic else 'user_supplied_data_analysis',
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'bootstrap_iterations': iterations, 'seed': 20260822,
        'python': sys.version.split()[0],
        'packages': {n: importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','uproot']},
        'source_sha256': {p.name: sha256(p) for p in sorted(scripts.glob('*.py'))},
        'steps': [],
    }
    record_path = output / 'run_record.json'
    def save():
        record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    save()
    steps = [
        ('time_matched_kr_v21.py', ['--kr', str(kr), '--th2615', str(th), '--output', str(output)]),
        ('bootstrap_time_matched_v21.py', ['--input', str(output/'2615_time_matched_output.txt'),
                                        '--output', str(output), '--iterations', str(iterations)]),
    ]
    try:
        for filename, flags in steps:
            start = time.monotonic()
            logfile = output / (Path(filename).stem + '.log')
            print(f'Running {filename} ...', flush=True)
            with logfile.open('w', encoding='utf-8') as log:
                completed = subprocess.run([sys.executable, str(scripts/filename), *flags], stdout=log, stderr=subprocess.STDOUT)
            record['steps'].append({'script': filename, 'returncode': completed.returncode,
                                    'elapsed_seconds': round(time.monotonic()-start, 3), 'log': logfile.name})
            save()
            if completed.returncode:
                raise RuntimeError(f'{filename} failed; see {logfile}')
        record['status'] = 'complete'
    except BaseException:
        record['status'] = 'failed'
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(f'Complete: {output}')
    if synthetic:
        print('SYNTHETIC DATA ONLY: these outputs are not experimental evidence.')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kr', required=True, type=Path)
    p.add_argument('--th2615', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--iterations', type=int, default=150)
    args = p.parse_args()
    try:
        run(args.kr, args.th2615, args.output, args.iterations)
    except (InputError, ImportError, OSError, RuntimeError) as exc:
        p.exit(2, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()

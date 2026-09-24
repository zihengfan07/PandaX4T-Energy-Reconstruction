"""Validate the input contract of the archived time-matched v21 workflow."""
import argparse
import json
from pathlib import Path

COLUMNS = ['qS1_PCs', 'qS2Bdes_PCs', 'dt', 'wS2CDF_max',
           'xS2max_desImageMCPAF_firstS2', 'yS2max_desImageMCPAF_firstS2', 'runNumber']
KR_RUNS = {9568, 9685, 9695}
TH_RUNS = {9563, 9566, 9698}


class InputError(ValueError):
    pass


def validate_arrays(arrays, kind):
    import numpy as np
    if kind not in ('kr', 'th2615'):
        raise InputError('Unknown dataset kind: ' + kind)
    missing = sorted(set(COLUMNS) - arrays.keys())
    if missing:
        raise InputError(f'{kind}: missing branches: {missing}')
    lengths = set()
    for name in COLUMNS:
        a = np.asarray(arrays[name])
        if a.ndim != 1 or a.dtype.kind not in 'iuf':
            raise InputError(f'{kind}: {name} must be a one-dimensional numeric branch')
        if not np.isfinite(a).all():
            raise InputError(f'{kind}: {name} contains non-finite values; review selection instead of silently dropping them')
        lengths.add(len(a))
    if len(lengths) != 1 or 0 in lengths:
        raise InputError(f'{kind}: branches must have equal, nonzero lengths')
    for name in ['qS1_PCs', 'qS2Bdes_PCs']:
        if (arrays[name] <= 0).any():
            raise InputError(f'{kind}: {name} must be positive')
    r = arrays['runNumber']
    if not np.equal(r, np.floor(r)).all():
        raise InputError(f'{kind}: runNumber must contain integers')
    runs = set(int(x) for x in np.unique(r))
    required = KR_RUNS if kind == 'kr' else TH_RUNS
    if required - runs:
        raise InputError(f'{kind}: missing fixed-pairing runs {sorted(required-runs)}; new periods require a new study design')
    if kind == 'th2615' and runs - required:
        raise InputError(f'{kind}: unmapped runs {sorted(runs-required)} would remain uncorrected in the historical script')
    counts = {str(run): int((r == run).sum()) for run in sorted(runs)}
    result = {'events': len(r), 'run_counts': counts}
    if kind == 'kr':
        e = .0137 * (arrays['qS1_PCs'] / .128755 + arrays['qS2Bdes_PCs'] / 8.6125)
        peak = (e >= 37.75) & (e <= 44.75)
        peak_counts = {str(run): int(((r == run) & peak).sum()) for run in sorted(required)}
        period_counts = {'P1': peak_counts['9568'], 'P2': peak_counts['9685'] + peak_counts['9695']}
        if min(peak_counts.values()) < 100 or min(period_counts.values()) < 2000:
            raise InputError(f'kr: insufficient peak-window statistics for normalization/local binning: {peak_counts}')
        result['used_run_peak_counts'] = peak_counts
        result['used_period_peak_counts'] = period_counts
    return result


def inspect_file(path, kind):
    import uproot
    p = Path(path)
    if not p.is_file():
        raise InputError(f'{kind}: input file does not exist: {p}')
    with uproot.open(p) as root:
        if 'out_tree' not in root:
            raise InputError(f'{kind}: missing TTree out_tree')
        tree = root['out_tree']
        if tree.classname != 'TTree':
            raise InputError(f'{kind}: out_tree must be a TTree, found {tree.classname}')
        missing = set(COLUMNS) - set(tree.keys())
        if missing:
            raise InputError(f'{kind}: missing branches: {sorted(missing)}')
        return validate_arrays(tree.arrays(COLUMNS, library='np'), kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kr', required=True, type=Path)
    p.add_argument('--th2615', required=True, type=Path)
    args = p.parse_args()
    try:
        print(json.dumps({'kr': inspect_file(args.kr, 'kr'), 'th2615': inspect_file(args.th2615, 'th2615')}, indent=2))
    except (InputError, ImportError, OSError) as exc:
        p.exit(2, f'Input check failed: {exc}\n')


if __name__ == '__main__':
    main()

"""Summarize archived experimental evidence; never runs or simulates an experiment."""
import argparse
import json
from evidence import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable values')
    args = parser.parse_args()
    rows, periods = load_rows()
    if args.json:
        print(json.dumps({'kind': 'archived_experimental_summary', 'rows': rows,
                          'boundary_fractions': {p['name']: p['support']['s2_clip_fraction'] for p in periods}}, indent=2))
        return
    print('ARCHIVED EXPERIMENTAL RESULTS | point estimates, not bootstrap medians\n')
    print('| Sample | sigma/mu before -> after | Relative gain | 95% bootstrap interval | Retained |')
    print('|---|---:|---:|---:|---:|')
    for r in rows:
        lo, hi = r['gain_interval_percent']
        print(f"| {r['sample']} | {r['resolution_before_percent']:.3f}% -> {r['resolution_after_percent']:.3f}% "
              f"| {r['relative_gain_percent']:.2f}% | [{lo:.2f}%, {hi:.2f}%] "
              f"| {r['retained_bootstrap_iterations']}/150 |")
    print('\nPeak positions and absolute widths (keV):')
    for r in rows:
        print(f"{r['sample']:>5}: mu {r['mu_before_kev']:.2f} -> {r['mu_after_kev']:.2f}; "
              f"sigma {r['sigma_before_kev']:.3f} -> {r['sigma_after_kev']:.3f}")
    for p in periods:
        print(f"{p['name']}: S2 either-boundary hits = {100*p['support']['s2_clip_fraction']:.2f}%")
    print('\nIntervals condition on retained fits and fixed maps. They do not include calibration/model-selection uncertainty.')
    print('See docs/RESULTS.md and docs/VALIDATION.md. This is not a new data analysis.')


if __name__ == '__main__':
    main()

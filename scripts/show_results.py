"""Print archived v21 evidence; does not rerun the experiment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'energy_resolution_analysis' / 'results' / 'v21_kr_residual_correction_20260822'


def main():
    data = json.loads((RESULTS / 'time_matched_bootstrap.json').read_text(encoding='utf-8'))
    print('Archived v21 bootstrap results (not a new experiment)')
    for run, row in data.items():
        gain = row['relative_gain']
        print(f"{run:12s} successful fits={row['successful_iterations']:3d}/150; "
              f"median relative gain={100*gain['median']:.2f}%; "
              f"95% interval=[{100*gain['p025']:.2f}%, {100*gain['p975']:.2f}%]")
    print('Candidate only: P1 has 22.8% S2 lower-bound hits; run 9698 interval includes zero.')
    print('Raw inputs are not included. See docs/REPRODUCIBILITY.md.')


if __name__ == '__main__':
    main()

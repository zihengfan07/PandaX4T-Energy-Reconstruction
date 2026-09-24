"""Generate toy ROOT inputs and run v21; never represents PandaX detector simulation."""
import argparse
import json
from pathlib import Path
from run_analysis import run


def generate(path, runs, energy, count, rng):
    import numpy as np
    import uproot
    pieces = []
    for run_id in runs:
        dt = rng.uniform(10000, 850000, count)
        z = (dt - 430000) / 420000
        latent_e = energy * (1 + rng.normal(0, .017, count))
        split = np.clip(.5 + rng.normal(0, .035, count), .2, .8)
        s1 = .128755 * latent_e / .0137 * split * (1 + .008*z)
        s2 = 8.6125 * latent_e / .0137 * (1-split) * (1 - .05*z)
        pieces.append({
            'qS1_PCs': s1, 'qS2Bdes_PCs': s2, 'dt': dt,
            'wS2CDF_max': 3000 + 100*z + rng.normal(0, 100, count),
            'xS2max_desImageMCPAF_firstS2': rng.normal(0, 100, count),
            'yS2max_desImageMCPAF_firstS2': rng.normal(0, 100, count),
            'runNumber': np.full(count, run_id, dtype=np.int32),
        })
    arrays = {k: np.concatenate([a[k] for a in pieces]) for k in pieces[0]}
    with uproot.recreate(path) as root:
        root.mktree('out_tree', {k: v.dtype for k, v in arrays.items()})
        root['out_tree'].extend(arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('local_results/demo'))
    p.add_argument('--iterations', type=int, default=12, help='Smoke-test bootstrap count, not an uncertainty study')
    args = p.parse_args()
    output = args.output.resolve()
    if args.iterations < 2:
        p.error('--iterations must be at least 2')
    if output.exists():
        p.error('Demo output already exists. Use a fresh directory; no data will be overwritten.')
    import numpy as np
    output.mkdir(parents=True)
    note = {'kind': 'synthetic_demonstration', 'seed': 20260924,
            'warning': 'Toy arrays only; not detector simulation, experimental data, or evidence of performance.',
            'run_numbers': 'Historical labels are reused only to exercise the fixed workflow.'}
    (output/'SYNTHETIC_DATA.json').write_text(json.dumps(note, indent=2)+'\n', encoding='utf-8')
    rng = np.random.default_rng(note['seed'])
    kr, th = output/'synthetic_kr.root', output/'synthetic_2615.root'
    generate(kr, [9568, 9685, 9695], 41.5, 6000, rng)
    generate(th, [9563, 9566, 9698], 2614.5, 2500, rng)
    run(kr, th, output/'analysis', args.iterations, synthetic=True)


if __name__ == '__main__':
    main()

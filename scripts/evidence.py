"""Read archived aggregate evidence, without scientific Python dependencies."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(root=ROOT):
    base = Path(root) / 'results/v21'
    summary = json.loads((base / 'time_matched_summary.json').read_text('utf-8'))
    bootstrap = json.loads((base / 'time_matched_bootstrap.json').read_text('utf-8'))
    fits = summary['2615_fits']
    rows = []
    for label, pair in [*fits['per_run'].items(), ('all', fits['all'])]:
        before, after = pair['base'], pair['corrected']
        if not (before['success'] and after['success']):
            raise ValueError(f'{label}: unsuccessful archived fit')
        for fit in [before, after]:
            if not all(math.isfinite(fit[k]) and fit[k] > 0 for k in ['mu_kev', 'sigma_kev', 'sigma_over_mu']):
                raise ValueError(f'{label}: invalid archived fit parameter')
            if not math.isclose(fit['sigma_kev'] / fit['mu_kev'], fit['sigma_over_mu'], rel_tol=1e-9):
                raise ValueError(f'{label}: inconsistent sigma/mu')
        boot = bootstrap['all' if label == 'all' else 'run_' + label]
        rows.append({
            'sample': label,
            'mu_before_kev': before['mu_kev'], 'mu_after_kev': after['mu_kev'],
            'sigma_before_kev': before['sigma_kev'], 'sigma_after_kev': after['sigma_kev'],
            'resolution_before_percent': 100 * before['sigma_over_mu'],
            'resolution_after_percent': 100 * after['sigma_over_mu'],
            'relative_gain_percent': 100 * (1 - after['sigma_over_mu'] / before['sigma_over_mu']),
            'gain_interval_percent': [100 * boot['relative_gain'][k] for k in ['p025', 'p975']],
            'retained_bootstrap_iterations': boot['successful_iterations'],
            'requested_bootstrap_iterations': 150,
            'absolute_sigma_decreased': after['sigma_kev'] < before['sigma_kev'],
        })
    return rows, summary['periods']

"""Guard against misleading summaries and unsupported input silently entering v21."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence import load_rows
from check_inputs import InputError, inspect_file, validate_arrays
from run_analysis import run


class EvidenceTests(unittest.TestCase):
    def test_point_estimate_is_not_bootstrap_median(self):
        rows, _ = load_rows()
        all_row = next(r for r in rows if r['sample'] == 'all')
        self.assertEqual(round(all_row['relative_gain_percent'], 2), 14.54)

    def test_ratio_improvement_does_not_imply_narrower_absolute_peak(self):
        rows, periods = load_rows()
        row = next(r for r in rows if r['sample'] == '9698')
        self.assertGreater(row['relative_gain_percent'], 0)
        self.assertFalse(row['absolute_sigma_decreased'])
        p1 = next(p for p in periods if p['name'] == 'P1')
        self.assertAlmostEqual(p1['support']['s2_clip_fraction'], .22762667870443518)


class InputTests(unittest.TestCase):
    def arrays(self):
        import numpy as np
        from check_inputs import COLUMNS
        a = {k: np.ones(6) for k in COLUMNS}
        a['runNumber'] = np.array([9563,9563,9566,9566,9698,9698])
        return a

    def test_unmapped_run_is_rejected(self):
        a = self.arrays()
        a['runNumber'][0] = 9999
        with self.assertRaisesRegex(InputError, 'unmapped'):
            validate_arrays(a, 'th2615')

    def test_nonfinite_signal_is_not_silently_dropped(self):
        a = self.arrays()
        a['qS1_PCs'][0] = float('nan')
        with self.assertRaisesRegex(InputError, 'non-finite'):
            validate_arrays(a, 'th2615')

    def test_missing_tree_is_actionable(self):
        import uproot
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'empty.root'
            with uproot.recreate(p):
                pass
            with self.assertRaisesRegex(InputError, 'out_tree'):
                inspect_file(p, 'kr')

    def test_small_run_is_valid_when_its_combined_period_has_enough_events(self):
        import numpy as np
        from check_inputs import COLUMNS
        runs = np.repeat([9568, 9685, 9695], [2200, 1500, 700])
        a = {k: np.ones(len(runs)) for k in COLUMNS}
        a['runNumber'] = runs
        a['qS1_PCs'][:] = .128755 * 41.5 / .0137 / 2
        a['qS2Bdes_PCs'][:] = 8.6125 * 41.5 / .0137 / 2
        result = validate_arrays(a, 'kr')
        self.assertEqual(result['used_period_peak_counts'], {'P1': 2200, 'P2': 2200})

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output/'existing.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(InputError, 'never overwritten'):
                run('missing.root', 'missing.root', output)
            self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

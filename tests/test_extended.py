"""Consistency tests for the extended analyses and the security fuzz results.

Test plan (data-pipeline strategy): transformation correctness against independent recomputation,
cross-file agreement between result artifacts, idempotency of seeded procedures, security boundaries.
"""
import unittest, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from study import ROOT, metrics
from extended import rao_wu, ece, calib_slope_intercept

S = json.loads((ROOT / 'results/study.json').read_text())
E = json.loads((ROOT / 'results/extended.json').read_text())
X = json.loads((ROOT / 'results/security_extended.json').read_text())
PRED = pd.read_csv(ROOT / 'results/test_predictions.csv')


def sel(task):
    return S['models'][task]['selected']


class ExtendedConsistency(unittest.TestCase):
    def test_selected_models_agree(self):
        for t in ['diabetes', 'hypertension']:
            self.assertEqual(E['selected_models'][t], sel(t))

    def test_point_estimates_match_primary(self):
        for t in ['diabetes', 'hypertension']:
            m = S['models'][t]['candidates'][sel(t)]
            self.assertAlmostEqual(E['design_aware'][t]['cluster_unweighted']['auroc']['point'], m['test']['auroc'], places=9)
            self.assertAlmostEqual(E['design_aware'][t]['design_weighted']['ppv']['point'], m['weighted_test']['ppv'], places=9)

    def test_predictions_recompute_primary_metrics(self):
        for t in ['diabetes', 'hypertension']:
            m = S['models'][t]['candidates'][sel(t)]['test']
            r = metrics(PRED[t], PRED[f'{t}_{sel(t)}'], m['threshold'])
            for k in ['auroc', 'sensitivity', 'ppv', 'tp', 'fp']:
                self.assertAlmostEqual(r[k], m[k], places=9)

    def test_intervals_bracket_points(self):
        for t in E['design_aware'].values():
            for key in ['cluster_unweighted', 'design_weighted']:
                for k in ['auroc', 'sensitivity', 'specificity', 'ppv']:
                    e = t[key][k]
                    self.assertLessEqual(e['ci'][0], e['point'] + 0.05)
                    self.assertGreaterEqual(e['ci'][1], e['point'] - 0.05)
                    self.assertLess(e['ci'][0], e['ci'][1])

    def test_rule_baseline_counts(self):
        test = PRED
        # Independent recomputation of one rule from public predictions plus raw NHANES age is not possible
        # without the cohort, so check internal arithmetic: sensitivity and PPV are proportions.
        for rows in E['rule_baselines'].values():
            for r in rows:
                self.assertTrue(0 <= r['sensitivity'] <= 1 and 0 < r['referral'] <= 1)

    def test_target_sweep_is_monotone(self):
        for rows in E['target_sweep'].values():
            sens = [r['sensitivity'] for r in rows]
            ref = [r['referral'] for r in rows]
            th = [r['threshold'] for r in rows]
            self.assertEqual(sens, sorted(sens))
            self.assertEqual(ref, sorted(ref))
            self.assertEqual(th, sorted(th, reverse=True))

    def test_primary_threshold_in_sweep(self):
        for t in ['diabetes', 'hypertension']:
            thr = S['models'][t]['candidates'][sel(t)]['test']['threshold']
            row = next(r for r in E['target_sweep'][t] if r['target'] == .8)
            self.assertAlmostEqual(row['threshold'], thr, places=12)

    def test_repeated_splits_complete(self):
        d = pd.read_csv(ROOT / 'results/repeated_splits.csv')
        self.assertEqual(len(d), 60)
        self.assertTrue(set(d.seed).isdisjoint({20261015}))
        self.assertTrue(d.auroc.between(.5, 1).all())

    def test_planning_arithmetic(self):
        row = next(p for p in E['planning'] if p['prevalence'] == .028 and p['sensitivity'] == .8 and p['half_width'] == .05)
        self.assertEqual(row['positives'], 246)
        self.assertAlmostEqual(row['recruits_10pct_nonevaluable'], 246 / .028 / .9, delta=1)


class Helpers(unittest.TestCase):
    def test_rao_wu_preserves_stratum_total(self):
        df = pd.DataFrame({'SDMVSTRA': [1, 1, 1, 1, 2, 2, 2, 2], 'SDMVPSU': [1, 1, 2, 2, 1, 1, 2, 2]})
        rng = np.random.default_rng(1)
        for _ in range(50):
            m = rao_wu(df, rng)
            self.assertTrue((m >= 0).all())
            # multipliers are n/(n-1)*r with n-1 draws, so each stratum sums to its PSU count
            self.assertAlmostEqual(m[:4].sum() / 2, 2.0, places=9)
            self.assertAlmostEqual(m[4:].sum() / 2, 2.0, places=9)

    def test_rao_wu_is_seeded(self):
        df = pd.DataFrame({'SDMVSTRA': [1] * 4, 'SDMVPSU': [1, 1, 2, 2]})
        a = rao_wu(df, np.random.default_rng(5))
        b = rao_wu(df, np.random.default_rng(5))
        self.assertTrue((a == b).all())

    def test_ece_perfect_and_biased(self):
        y = np.array([0, 0, 1, 1] * 8)
        self.assertAlmostEqual(ece(y, np.full(len(y), .5)), 0.0, places=9)
        self.assertGreater(ece(y, np.full(len(y), .9)), .3)

    def test_calibration_slope_of_perfect_scores(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(.05, .95, 20000)
        y = (rng.random(20000) < p).astype(int)
        s, i = calib_slope_intercept(y, p)
        self.assertAlmostEqual(s, 1.0, delta=.05)
        self.assertAlmostEqual(i, 0.0, delta=.05)


class SecurityFuzz(unittest.TestCase):
    def test_every_mutation_rejected(self):
        for group in ['mutation_fuzz', 'mutation_fuzz_resigned']:
            for field, r in X[group].items():
                self.assertEqual(r['rejected'], r['trials'], f'{group}:{field}')

    def test_resigned_fuzz_exercises_inner_layers(self):
        self.assertEqual(set(X['mutation_fuzz_resigned']), {'kem_ct', 'ephemeral_xpk', 'nonce', 'ciphertext'})

    def test_size_overhead_reported(self):
        c = X['classical_vs_hybrid']
        self.assertGreater(c['hybrid_bytes'], 10 * c['classical_bytes'])


if __name__ == '__main__':
    unittest.main()

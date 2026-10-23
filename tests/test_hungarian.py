"""Existing CPU algorithm regressions, relocated to the package import path.
Tests do not execute pretrained models or claim hidden accuracy.
"""
import itertools
import json
from pathlib import Path
import time
import unittest
import numpy as np
from ntl.postprocessing.hungarian import selective_hungarian, _assignment_with_gap
ROOT=Path(__file__).resolve().parents[1]/'artifacts'/'test-results'
ROOT.mkdir(parents=True,exist_ok=True)

def collision_fixture(rows=18, classes=18):
    probability = np.full((rows, classes), 1e-05)
    probability[np.arange(rows), np.arange(rows)] = 0.99
    probability[0, :2] = [0.51, 0.48]
    probability[1, :2] = [0.9, 0.09]
    probability /= probability.sum(1, keepdims=True)
    return probability

class SelectiveAssignmentTests(unittest.TestCase):

    def test_supported_collision_corrected_without_modifying_probability(self):
        probability = collision_fixture()
        original = probability.copy()
        selected, report = selective_hungarian(probability, np.zeros(18, int), np.full(18, 91))
        np.testing.assert_array_equal(selected, np.r_[1, 0, np.arange(2, 18)])
        np.testing.assert_array_equal(probability, original)
        self.assertEqual(report['accepted_groups'], 1)
        self.assertEqual(report['changed_rows'], 1)

    def test_incomplete_prompt_does_not_force_missing_classes(self):
        probability = collision_fixture(rows=12)
        selected, report = selective_hungarian(probability, np.zeros(12, int), np.ones(12, int))
        self.assertEqual(report['accepted_groups'], 1)
        self.assertEqual(len(set(selected)), 12)
        self.assertLess(int(selected.max()), 12)

    def test_consensus_rejects_splits_and_extra_members(self):
        probability = np.vstack([collision_fixture(), np.full((1, 18), 1 / 18)])
        first = np.r_[np.zeros(18, int), 999]
        for second in [np.r_[np.zeros(17, int), 1, 2], np.zeros(19, int)]:
            selected, report = selective_hungarian(probability, first, second)
            np.testing.assert_array_equal(selected, probability.argmax(1))
            self.assertEqual(report['group_decisions'][0]['reason'], 'grouping_disagreement')

    def test_ambiguous_optimum_keeps_whole_group_baseline(self):
        probability = collision_fixture()
        probability[1] = probability[0]
        selected, report = selective_hungarian(probability, np.zeros(18, int), np.ones(18, int))
        np.testing.assert_array_equal(selected, probability.argmax(1))
        self.assertEqual(report['group_decisions'][0]['reason'], 'unstable_assignment')

    def test_forced_low_probability_label_is_rejected(self):
        probability = collision_fixture()
        probability[0, :2] = [0.98, 0.01]
        probability[0] /= probability[0].sum()
        selected, report = selective_hungarian(probability, np.zeros(18, int), np.ones(18, int))
        np.testing.assert_array_equal(selected, probability.argmax(1))
        self.assertEqual(report['accepted_groups'], 0)

    def test_public_sample_and_no_collision_are_exact_fallbacks(self):
        for probability in [np.full((3, 18), 1 / 18), np.eye(18)]:
            count = len(probability)
            selected, report = selective_hungarian(probability, np.zeros(count, int), np.ones(count, int))
            np.testing.assert_array_equal(selected, probability.argmax(1))
            self.assertEqual(report['changed_rows'], 0)

    def test_permuting_rows_classes_and_group_ids_preserves_decoding(self):
        rng = np.random.default_rng(19)
        probability = collision_fixture()
        expected, _ = selective_hungarian(probability, np.zeros(18, int), np.ones(18, int))
        rows, columns = (rng.permutation(18), rng.permutation(18))
        result, _ = selective_hungarian(probability[rows][:, columns], np.full(18, -6), np.full(18, 830))
        np.testing.assert_array_equal(columns[result], expected[rows])

    def test_second_best_gap_matches_exhaustive_rectangular_solution(self):
        rng = np.random.default_rng(23)
        for rows, columns in [(2, 3), (3, 3), (3, 4)]:
            cost = rng.uniform(0.1, 4.0, size=(rows, columns))
            possibilities = sorted((float(cost[np.arange(rows), permutation].sum()) for permutation in itertools.permutations(range(columns), rows)))
            selected, gap = _assignment_with_gap(cost)
            self.assertAlmostEqual(float(cost[np.arange(rows), selected].sum()), possibilities[0])
            self.assertAlmostEqual(gap, possibilities[1] - possibilities[0])

    def test_invalid_probabilities_or_misaligned_groups_stop(self):
        for probability in [np.full((18, 18), np.nan), np.zeros((18, 18)), np.full((18, 18), -1), np.full((18, 18), 0.5)]:
            with self.assertRaises(ValueError):
                selective_hungarian(probability, np.zeros(18, int), np.ones(18, int))
        with self.assertRaises(ValueError):
            selective_hungarian(collision_fixture(), np.zeros(17, int), np.ones(18, int))
        with self.assertRaises(ValueError):
            selective_hungarian(collision_fixture(), np.zeros(18, int), np.ones(18, int), {'typo': 1})

    def test_4000_row_cpu_benchmark(self):
        probability = np.vstack([collision_fixture() for _ in range(222)] + [np.full((4, 18), 1 / 18)])
        groups = np.r_[np.repeat(np.arange(222), 18), np.arange(222, 226)]
        started = time.perf_counter()
        selected, report = selective_hungarian(probability, groups, groups + 1000)
        elapsed = time.perf_counter() - started
        self.assertEqual(report['accepted_groups'], 222)
        self.assertEqual(report['changed_rows'], 222)
        self.assertEqual(selected.shape, (4000,))
        (ROOT / 'cpu_benchmark.json').write_text(json.dumps({'rows': 4000, 'dense_groups': 222, 'seconds': elapsed, 'synthetic_probabilities': True, 'includes_qwen_inference': False}, indent=2))

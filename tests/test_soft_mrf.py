"""Existing CPU algorithm regressions, relocated to the package import path.
Tests do not execute pretrained models or claim hidden accuracy.
"""
import itertools
import json
from pathlib import Path
import time
import unittest
import numpy as np
from ntl.postprocessing.soft_mrf import soft_constraint_decode, _soft_energy, _solve_soft_clique
ROOT=Path(__file__).resolve().parents[1]/'artifacts'/'test-results'
ROOT.mkdir(parents=True,exist_ok=True)

def two_way_fixture():
    return np.asarray([[0.51, 0.49], [0.9, 0.1]])

class SoftConstraintTests(unittest.TestCase):

    def test_exact_map_and_configuration_gap_against_enumeration(self):
        rng = np.random.default_rng(101)
        for rows, classes in [(2, 2), (3, 2), (3, 3), (4, 3)]:
            unary = rng.uniform(0.1, 4.0, size=(rows, classes))
            for penalty in [0.0, 0.4, 3.0]:
                possibilities = sorted((_soft_energy(unary, np.asarray(labels), penalty) for labels in itertools.product(range(classes), repeat=rows)))
                chosen, energy, gap = _solve_soft_clique(unary, penalty)
                self.assertAlmostEqual(energy, possibilities[0])
                self.assertAlmostEqual(gap, possibilities[1] - possibilities[0])
                self.assertAlmostEqual(_soft_energy(unary, chosen, penalty), energy)

    def test_supported_collision_changes_one_label_without_probability_mutation(self):
        probability = two_way_fixture()
        saved = probability.copy()
        chosen, report = soft_constraint_decode(probability, np.zeros(2, int), np.ones(2, int))
        np.testing.assert_array_equal(chosen, [1, 0])
        np.testing.assert_array_equal(probability, saved)
        self.assertEqual(report['accepted_blocks'], 1)
        self.assertEqual(report['changed_rows'], 1)

    def test_strong_duplicate_prediction_remains_legal(self):
        probability = np.asarray([[0.99, 0.01], [0.98, 0.02]])
        chosen, report = soft_constraint_decode(probability, np.zeros(2, int), np.ones(2, int))
        np.testing.assert_array_equal(chosen, [0, 0])
        self.assertEqual(report['block_decisions'][0]['reason'], 'baseline_is_soft_optimum')
        self.assertGreater(report['block_decisions'][0]['configuration_gap'], 0)

    def test_label_configuration_tie_does_not_count_slot_permutations(self):
        probability = np.asarray([[0.51, 0.49], [0.51, 0.49]])
        chosen, report = soft_constraint_decode(probability, np.zeros(2, int), np.ones(2, int))
        np.testing.assert_array_equal(chosen, [0, 0])
        self.assertEqual(report['block_decisions'][0]['reason'], 'ambiguous_label_configuration')

    def test_lower_grouping_confidence_reduces_constraint_strength(self):
        pair = np.asarray([[0.56, 0.44], [0.95, 0.05]])
        strong, strong_report = soft_constraint_decode(pair, np.zeros(2, int), np.ones(2, int))
        np.testing.assert_array_equal(strong, [1, 0])
        probability = np.vstack([pair, [0.01, 0.99], [0.01, 0.99]])
        weak, weak_report = soft_constraint_decode(probability, np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]))
        np.testing.assert_array_equal(weak[:2], [0, 0])
        self.assertAlmostEqual(strong_report['block_decisions'][0]['confidence'], 1)
        self.assertAlmostEqual(weak_report['block_decisions'][0]['confidence'], 0.5)

    def test_zero_penalty_is_noop(self):
        chosen, report = soft_constraint_decode(two_way_fixture(), np.zeros(2, int), np.ones(2, int), {'penalty_strength': 0})
        np.testing.assert_array_equal(chosen, [0, 0])
        self.assertEqual(report['changed_rows'], 0)

    def test_unreasonably_weak_label_is_guarded_even_with_large_penalty(self):
        probability = np.asarray([[0.99, 0.01], [0.98, 0.02]])
        chosen, report = soft_constraint_decode(probability, np.zeros(2, int), np.ones(2, int), {'penalty_strength': 10})
        np.testing.assert_array_equal(chosen, [0, 0])
        self.assertEqual(report['block_decisions'][0]['reason'], 'individual_probability_loss')

    def test_large_parent_groups_are_not_treated_as_one_prompt(self):
        probability = np.tile(two_way_fixture(), (10, 1))
        chosen, report = soft_constraint_decode(probability, np.zeros(20, int), np.ones(20, int))
        np.testing.assert_array_equal(chosen, probability.argmax(1))
        self.assertEqual(report['block_reasons'], {'oversized_parent_group': 1})

    def test_singletons_and_already_unique_labels_are_unchanged(self):
        for probability, groups in [(np.eye(3), np.zeros(3, int)), (two_way_fixture(), np.arange(2))]:
            chosen, report = soft_constraint_decode(probability, groups, groups + 400)
            np.testing.assert_array_equal(chosen, probability.argmax(1))
            self.assertEqual(report['changed_rows'], 0)

    def test_row_class_and_group_id_permutation_invariance(self):
        probability = np.asarray([[0.51, 0.48, 0.01], [0.9, 0.09, 0.01], [0.01, 0.01, 0.98]])
        expected, _ = soft_constraint_decode(probability, np.zeros(3, int), np.ones(3, int))
        rows = np.array([2, 0, 1])
        columns = np.array([2, 1, 0])
        chosen, _ = soft_constraint_decode(probability[rows][:, columns], np.full(3, -8), np.full(3, 901))
        np.testing.assert_array_equal(columns[chosen], expected[rows])

    def test_invalid_inputs_stop_instead_of_silently_decoding(self):
        for probability in [np.zeros((2, 2)), np.full((2, 2), np.nan), np.full((2, 2), -1), np.ones((2, 2))]:
            with self.assertRaises(ValueError):
                soft_constraint_decode(probability, np.zeros(2, int), np.ones(2, int))
        with self.assertRaises(ValueError):
            soft_constraint_decode(two_way_fixture(), np.zeros(1, int), np.ones(2, int))
        with self.assertRaises(ValueError):
            soft_constraint_decode(two_way_fixture(), np.zeros(2, int), np.ones(2, int), {'unknown': 3})

    def test_4000_row_cpu_cost_is_bounded(self):
        dense = np.full((18, 18), 1e-05)
        dense[np.arange(18), np.arange(18)] = 0.99
        dense[0, :2] = [0.51, 0.48]
        dense[1, :2] = [0.9, 0.09]
        dense /= dense.sum(1, keepdims=True)
        probability = np.vstack([dense for _ in range(222)] + [np.full((4, 18), 1 / 18)])
        groups = np.r_[np.repeat(np.arange(222), 18), np.arange(222, 226)]
        start = time.perf_counter()
        chosen, report = soft_constraint_decode(probability, groups, groups + 1000)
        elapsed = time.perf_counter() - start
        self.assertEqual(len(chosen), 4000)
        self.assertEqual(report['accepted_blocks'], 222)
        (ROOT / 'cpu_benchmark.json').write_text(json.dumps({'rows': 4000, 'seconds': elapsed, 'synthetic_probabilities': True, 'includes_gpu_inference': False}, indent=2))

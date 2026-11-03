"""Tests for analytic planning fixtures, separate from the Adict research suite."""
import unittest
import numpy as np
from sandbox_gate_probe import (
    BASES, P, TARGET, all_states, chars, coefficients, native_use_fixture,
    read_product, run, worst_phase_budget,
)


class PlanningGateTests(unittest.TestCase):
    def test_basis_solution(self):
        expected = {"small": [2, 1], "medium": [7, 1], "large": [15, 15]}
        for name, basis in BASES.items():
            c = coefficients(basis)
            self.assertEqual(c.tolist(), expected[name])
            self.assertTrue(np.array_equal(c @ np.array(basis) % P, TARGET))

    def test_exact_all_states(self):
        x = all_states()
        for basis in BASES.values():
            self.assertTrue(np.array_equal(read_product(chars(x, basis), coefficients(basis)), x @ TARGET % P))

    def test_bounded_noise_certificate(self):
        x = all_states()
        rng = np.random.default_rng(18)
        for basis in BASES.values():
            c = coefficients(basis)
            bound = 0.49 * (np.pi / P) / np.abs(c).sum()
            errors = rng.uniform(-bound, bound, (len(x), 2))
            self.assertLess(worst_phase_budget(c, [bound, bound]), np.pi/P)
            pred = read_product(chars(x, basis)*np.exp(1j*errors), c)
            self.assertTrue(np.array_equal(pred, x @ TARGET % P))

    def test_outside_margin_counterexample(self):
        x = all_states()
        c = coefficients(BASES["large"])
        errors = np.full((len(x), 2), 1.1 * (np.pi/P) / np.abs(c).sum())
        pred = read_product(chars(x, BASES["large"])*np.exp(1j*errors), c)
        self.assertTrue(np.all(pred != x @ TARGET % P))

    def test_native_use_vs_bypass(self):
        d = native_use_fixture()
        self.assertTrue(d["original_outputs_identical"])
        self.assertEqual(d["external_phase_probe_accuracy_both"], 1)
        self.assertEqual(d["native_patch_counterfactual_accuracy_live"], 1)
        self.assertEqual(d["native_patch_counterfactual_accuracy_bypass"], 0)

    def test_deterministic_replay(self):
        self.assertEqual(run(repeats=1), run(repeats=1))

    def test_validation(self):
        with self.assertRaises(ValueError): coefficients(((0, 0), (0, 0)))
        with self.assertRaises(ValueError): all_states(1)
        with self.assertRaises(ValueError): run(repeats=0)
        with self.assertRaises(ValueError): read_product(np.zeros((2, 2)), np.ones(2))
        with self.assertRaises(ValueError): worst_phase_budget([1, 1], [-1, 1])


if __name__ == "__main__":
    unittest.main()

"""Blind Fourier discovery: exact-character recovery, label blindness, controls."""
import inspect
import math
import unittest

import torch

from research.accessible_causal_information.blind_fourier import (
    TARGET, complex_features, discover_characters, generic_features, group_decode,
    random_rotation_controls, shift_pairs, solve_exponents, blind_method, split_pairs)
from research.accessible_causal_information.pilot import states

P = 31


def characters(freqs, noise=0.0, seed=0):
    pairs, labels = states(P)
    cols = []
    for k, l in freqs:
        ph = 2 * math.pi * ((k * pairs[:, 0] + l * pairs[:, 1]) % P).double() / P
        cols += [torch.cos(ph), torch.sin(ph)]
    z = torch.stack(cols, 1)
    if noise:
        z = z + noise * torch.randn(z.shape, generator=torch.Generator().manual_seed(seed), dtype=z.dtype)
    return z, pairs, labels


class BlindFourierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recovers_exact_frequencies_for_all_estimators(self):
        for freqs in ([(1, 0), (0, 30)], [(5, 17), (29, 3)], [(1, 1), (1, 0)]):
            z, _, _ = characters(freqs)
            for est in ('ls', 'angle'):
                found = discover_characters(complex_features(z), shift_pairs(P, None, None), P, est)
                self.assertEqual([(d['k'], d['l']) for d in found], freqs)
                self.assertLess(max(d['group_law_residual'] for d in found), 1e-20)

    def test_discovery_receives_no_labels_or_operands(self):
        names = set(inspect.signature(discover_characters).parameters)
        self.assertEqual(names, {'zc', 'shifts', 'p', 'estimator'})
        shifts = shift_pairs(P, None, None)
        self.assertEqual(set(shifts), {'a', 'b', 'ab'})
        for i, j in shifts.values():
            self.assertEqual(i.dtype, torch.long)

    def test_relabeling_states_leaves_frequencies_unchanged(self):
        z, _, _ = characters([(3, 4), (7, 1)])
        shifts = shift_pairs(P, None, None)
        perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(5))
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(P * P)
        zp = z[perm]
        relabeled = {k: (inv[i], inv[j]) for k, (i, j) in shifts.items()}
        found = discover_characters(complex_features(zp), relabeled, P)
        self.assertEqual([(d['k'], d['l']) for d in found], [(3, 4), (7, 1)])

    def test_noise_tolerance_and_residual_grows(self):
        z, _, _ = characters([(1, 0), (0, 30)], noise=0.05)
        found = discover_characters(complex_features(z), shift_pairs(P, None, None), P)
        self.assertEqual([(d['k'], d['l']) for d in found], [(1, 0), (0, 30)])
        self.assertGreater(max(d['group_law_residual'] for d in found), 1e-4)

    def test_pool_restriction_keeps_endpoints_inside_pool(self):
        pool = torch.arange(0, P * P, 2)
        s = shift_pairs(P, pool, None)
        inpool = torch.zeros(P * P, dtype=torch.bool)
        inpool[pool] = True
        for i, j in s.values():
            self.assertTrue(bool(inpool[i].all() and inpool[j].all()))
        capped = shift_pairs(P, pool, 7)
        self.assertTrue(all(len(i) == 7 for i, _ in capped.values()))

    def test_exponent_solution_matches_target_for_all_independent_pairs(self):
        for k1, l1, k2, l2 in [(1, 0, 0, 1), (1, 0, 0, 30), (5, 17, 29, 3), (2, 7, 11, 13)]:
            c = solve_exponents([(k1, l1), (k2, l2)], TARGET, P)
            self.assertEqual(((c[0] * k1 + c[1] * k2) % P, (c[0] * l1 + c[1] * l2) % P), TARGET)
        self.assertIsNone(solve_exponents([(1, 2), (2, 4)], TARGET, P))

    def test_exact_features_decode_shear_perfectly_including_signed_lift(self):
        z, pairs, labels = characters([(1, 0), (0, 30)])
        zc = complex_features(z)
        idx = torch.arange(P * P)
        out = group_decode(zc, [(1, 0), (0, 30)], labels['shear'], idx[::2], idx[1::2], P, 'unit', 'signed')
        self.assertEqual(out['accuracy'], 1.0)
        self.assertEqual(out['exponents'], [2, -1])

    def test_dependent_frequencies_are_reported_as_failure(self):
        z, pairs, labels = characters([(1, 2), (2, 4)])
        idx = torch.arange(P * P)
        out = blind_method(z, labels['shear'], idx[::2], idx[1::2], idx[::2],
                           {'estimator': 'ls', 'normalize': 'unit', 'lift': 'signed', 'n_shift': None}, P)
        self.assertIsNone(out['accuracy'])
        self.assertFalse(out['decoder']['recoverable'])

    def test_random_rotation_destroys_discovery(self):
        z, pairs, labels = characters([(1, 0), (0, 30)])
        idx = torch.arange(P * P)
        cfg = {'estimator': 'ls', 'normalize': 'unit', 'lift': 'signed', 'n_shift': None}
        out = random_rotation_controls(z, labels['shear'], idx[::2], idx[1::2], idx[::2], cfg, P, 12, 1)
        self.assertGreater(out['group_law_residual_median'], 0.01)
        self.assertLess(out['accuracy_mean'], 0.5)

    def test_generic_feature_shapes(self):
        z = torch.randn(10, 32)
        self.assertEqual(generic_features(z, 'affine4').shape, (10, 4))
        self.assertEqual(generic_features(z, 'bilinear4').shape, (10, 14))
        self.assertEqual(generic_features(z, 'affine32').shape, (10, 32))
        self.assertEqual(generic_features(z, 'bilinear32').shape, (10, 32 + 32 * 33 // 2))

    def test_split_is_disjoint_and_sized(self):
        test, pool, fit, val = split_pairs({'split_seed': 20261103}, 961)
        sets = [set(t.tolist()) for t in (test, fit, val)]
        self.assertEqual((len(test), len(pool), len(fit), len(val)), (481, 480, 240, 240))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sets[1] | sets[2], set(pool.tolist()))


if __name__ == '__main__':
    unittest.main()

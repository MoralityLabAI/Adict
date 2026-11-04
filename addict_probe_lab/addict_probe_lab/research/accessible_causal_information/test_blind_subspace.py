"""Coordinate-free shift-representation discovery: synthetic exact codes, invariance, controls."""
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import torch

from research.accessible_causal_information.blind_fourier import shift_pairs, split_pairs
from research.accessible_causal_information.blind_subspace import (
    canon_p, discover_from_shifts, discover_subspace, exponent_search_decoder, fit_operators, load_checkpoint,
    monomial_features, pick_pair, rotation_matrix, state_hash, subspace_vs_truth,
    blind_subspace_method, tensor_hash, save_checkpoint)
from research.accessible_causal_information.pilot import Encoder, states

P = 31
CFG = {'ridge': 1e-4, 'mix': 0.7371, 'tau': 0.2, 'nu': 0.01, 'n_shift': None}


def code(freqs, nuisance=28, noise=0.0, seed=0):
    """32-d code: 4 phase coords then nuisance dims that do not follow the shift action."""
    pairs, labels = states(P)
    g = torch.Generator().manual_seed(seed)
    cols = []
    for k, l in freqs:
        ph = 2 * math.pi * ((k * pairs[:, 0] + l * pairs[:, 1]) % P).double() / P
        cols += [torch.cos(ph), torch.sin(ph)]
    z = torch.stack(cols, 1)
    z = z + noise * torch.randn(z.shape, generator=g, dtype=z.dtype)
    junk = torch.randn(len(pairs), nuisance, generator=g, dtype=torch.double)
    return torch.cat((z, junk), 1), labels


def modes(res):
    return {canon_p(tuple(f), P) for f in res['frequencies']}


class BlindSubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.idx = torch.arange(P * P)
        perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(77))
        cls.fit, cls.test = perm[:480].sort().values, perm[480:].sort().values

    def run_method(self, z, labels, cfg=CFG):
        return blind_subspace_method(z, labels['shear'], self.fit, self.test, self.fit, cfg, P)

    def test_recovers_frequencies_and_decodes_shear_without_coordinate_pairing(self):
        for freqs in ([(1, 0), (0, 30)], [(2, 0), (0, 1)], [(1, 1), (1, 0)]):
            z, labels = code(freqs)
            res = self.run_method(z, labels)
            self.assertEqual(modes(res), {canon_p(f, P) for f in freqs})
            self.assertEqual(res['accuracy'], 1.0)
            self.assertLess(res['group_law_residual'], 1e-3)
            self.assertGreater(res['subspace']['overlap_with_phase_coords'], 0.999)

    def test_discovery_is_invariant_to_orthogonal_rotation_of_the_code(self):
        z, labels = code([(1, 0), (0, 30)])
        base = self.run_method(z, labels)
        for seed in (1, 2, 3):
            res = self.run_method(z @ rotation_matrix(32, seed), labels)
            self.assertEqual(modes(res), modes(base))
            self.assertEqual(res['accuracy'], 1.0)
        pre = exponent_search_decoder(z, labels['shear'], self.fit, self.test, P)['accuracy']
        post = exponent_search_decoder(z @ rotation_matrix(32, 1), labels['shear'], self.fit, self.test, P)['accuracy']
        self.assertEqual(pre, 1.0)
        self.assertLess(post, 0.3)

    def test_discovery_signature_has_no_labels_pairing_or_targets(self):
        self.assertEqual(set(inspect.signature(discover_subspace).parameters), {'z', 'pool', 'cfg', 'p'})
        self.assertEqual(set(inspect.signature(fit_operators).parameters), {'z', 'shifts', 'pool', 'ridge'})

    def test_relabeling_states_leaves_frequencies_unchanged(self):
        z, labels = code([(3, 4), (7, 1)])
        shifts = shift_pairs(P, self.fit, None)
        perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(9))
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(P * P)
        d0 = discover_from_shifts(z, shifts, self.fit, CFG, P)
        relabeled = {k: (inv[i], inv[j]) for k, (i, j) in shifts.items()}
        d1 = discover_from_shifts(z[perm], relabeled, inv[self.fit], CFG, P)
        self.assertEqual({canon_p(c['freq'], P) for c in d0['features']},
                         {canon_p(c['freq'], P) for c in d1['features']})

    def test_shuffled_shift_graph_finds_nothing(self):
        z, labels = code([(1, 0), (0, 30)])
        g = torch.Generator().manual_seed(4)
        shifts = shift_pairs(P, self.fit, None)
        bad = {k: (i, j[torch.randperm(len(j), generator=g)]) for k, (i, j) in shifts.items()}
        _, _, rel = fit_operators(z, bad, self.fit, 1e-4)
        _, _, good = fit_operators(z, shifts, self.fit, 1e-4)
        self.assertGreater(min(rel.values()), 0.5)
        self.assertLess(max(good.values()), 0.9)

    def test_nuisance_only_code_yields_no_features_and_reports_failure(self):
        z, labels = code([(1, 0), (0, 30)])
        res = self.run_method(z[:, 4:], labels)
        self.assertEqual(res['n_features'], 0)
        self.assertIsNone(res['accuracy'])
        self.assertFalse(res['decoder']['recoverable'])

    def test_single_dependent_frequencies_report_no_independent_pair(self):
        z, labels = code([(1, 2), (2, 4)])
        res = self.run_method(z, labels)
        self.assertIsNone(res['accuracy'])

    def test_pick_pair_prefers_small_exponents(self):
        f = lambda k, l, r: {'freq': (k, l), 'residual': r}
        _, a, b = pick_pair([f(1, 0, .1), f(0, 1, .1), f(5, 17, .0)], P)
        self.assertEqual({a['freq'], b['freq']}, {(1, 0), (0, 1)})

    def test_subspace_overlap_metric(self):
        w = torch.zeros(32, dtype=torch.complex128)
        w[0], w[1] = 1, 1j
        self.assertAlmostEqual(subspace_vs_truth([{'w': w}])['overlap_with_phase_coords'], 0.5)
        w2 = torch.zeros(32, dtype=torch.complex128)
        w2[10], w2[11] = 1, 1j
        self.assertAlmostEqual(subspace_vs_truth([{'w': w2}])['leakage_outside_phase_coords'], 1.0)

    def test_monomial_features_shape(self):
        z, _ = code([(1, 0), (0, 30)])
        self.assertEqual(monomial_features(z).shape, (P * P, 48))

    def test_checkpoint_roundtrip_and_hash_guard(self):
        base = {'modulus': P, 'embedding_width': 16, 'hidden_width': 128, 'code_width': 32}
        torch.manual_seed(0)
        model = Encoder(base)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm.pt'
            info = save_checkpoint(model, path)
            again = load_checkpoint(base, path, info)
            self.assertEqual(state_hash(again), state_hash(model))
            bad = dict(info, file_sha256='0' * 64)
            with self.assertRaises(ValueError):
                load_checkpoint(base, path, bad)
        self.assertEqual(tensor_hash([('z', torch.ones(3))]), tensor_hash([('z', torch.ones(3))]))
        self.assertNotEqual(tensor_hash([('z', torch.ones(3))]), tensor_hash([('z', torch.zeros(3))]))

    def test_split_matches_bf01(self):
        test, pool, fit, val = split_pairs({'split_seed': 20261103}, 961)
        self.assertEqual((len(test), len(pool)), (481, 480))


if __name__ == '__main__':
    unittest.main()

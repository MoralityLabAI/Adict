"""C1 closeout: target decoding, held-out residual, interchange mechanics and artifact lock."""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from research.accessible_causal_information.blind_fourier import split_pairs
from research.accessible_causal_information.blind_subspace import discover_subspace
from research.accessible_causal_information.c1_closeout import (
    HERE, decode_target, gate_row, heldout_group_residual, interchange_pairs, native_interchange,
    patch, subspace_basis, verify_bf02_artifacts)
from research.accessible_causal_information.pilot import states
from research.accessible_causal_information.test_blind_subspace import CFG, code

P = 31


class MockModel(torch.nn.Module):
    """Head reading sum from a chosen pair of columns; stands in for read_sum."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.read_sum = self.head

    def head(self, z):
        if self.mode == 'product':           # chi(a) * chi(b) from columns 0-3
            f1, f2 = torch.complex(z[:, 0], z[:, 1]), torch.complex(z[:, 2], z[:, 3])
            prod = f1 * f2
        else:                                # bypass: chi(a+b) from columns 4-5
            prod = torch.complex(z[:, 4], z[:, 5])
        s = torch.arange(P, dtype=torch.double)
        centers = torch.exp(2j * math.pi * s / P)
        return (prod.to(torch.complex128)[:, None] * centers[None, :].conj()).real.float()


def synthetic(mode):
    pairs, labels = states(P)
    a, b = pairs[:, 0].double(), pairs[:, 1].double()
    ph = lambda k: 2 * math.pi * k / P
    z = torch.stack((torch.cos(ph(a)), torch.sin(ph(a)), torch.cos(ph(b)), torch.sin(ph(b)),
                     torch.cos(ph(a + b)), torch.sin(ph(a + b))), 1)
    z = torch.cat((z, torch.zeros(len(pairs), 26, dtype=torch.double)), 1)
    return pairs, labels, z, MockModel(mode)


class C1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(77))
        cls.fit, cls.test = perm[:480].sort().values, perm[480:].sort().values

    def feats(self, z):
        return discover_subspace(z, self.fit, CFG, P)['features']

    def test_both_small_exponent_targets_decode_exactly(self):
        z, labels = code([(1, 0), (0, 30)])
        pairs, _ = states(P)
        a, b = pairs[:, 0], pairs[:, 1]
        feats = self.feats(z)
        for target, lab in (((2, 1), (2 * a + b) % P), ((1, 2), (a + 2 * b) % P)):
            out = decode_target(feats, target, lab, self.fit, self.test, P)
            self.assertGreater(out['accuracy'], 0.95)
            self.assertLessEqual(out['max_abs_exponent'], 3)

    def test_abstention_counts_as_failure_in_gate(self):
        z, labels = code([(1, 2), (2, 4)])
        out = decode_target(self.feats(z), (2, 1), labels['shear'], self.fit, self.test, P)
        self.assertTrue(out['abstained'])
        self.assertIsNone(out['accuracy'])

    def test_heldout_residual_small_on_exact_code_and_large_on_noise(self):
        z, _ = code([(1, 0), (0, 30)])
        feats = self.feats(z)
        self.assertLess(heldout_group_residual(feats, P, self.test), 1e-3)
        for f in feats:
            f['feature'] = torch.randn(f['feature'].shape, generator=torch.Generator().manual_seed(3),
                                       dtype=torch.double).to(torch.complex128)
        self.assertGreater(heldout_group_residual(feats, P, self.test), 0.5)
        self.assertIsNone(heldout_group_residual([], P, self.test))

    def test_interchange_pairs_are_deterministic_inside_test_and_hold_b_fixed(self):
        pairs, _ = states(P)
        test, pool, _, _ = split_pairs({'split_seed': 2026110301}, 961)
        r, d = interchange_pairs(pairs, test, 256, 2026110302, P)
        r2, d2 = interchange_pairs(pairs, test, 256, 2026110302, P)
        self.assertTrue(torch.equal(r, r2) and torch.equal(d, d2))
        self.assertEqual(len(r), 256)
        inside = set(test.tolist())
        self.assertTrue(all(int(x) in inside for x in r) and all(int(x) in inside for x in d))
        self.assertTrue(torch.equal(pairs[r, 1], pairs[d, 1]))
        self.assertTrue(bool((pairs[r, 0] != pairs[d, 0]).all()))

    def test_zero_edit_full_donor_and_patch_algebra(self):
        pairs, labels, z, model = synthetic('product')
        feats = self.feats(z)
        r, d = interchange_pairs(pairs, self.test, 100, 1, P)
        res = native_interchange(model, z, feats, r, d, 4, 5)
        self.assertEqual(res['zero_edit_identity']['accuracy'], 1.0)
        self.assertEqual(res['full_code_donor']['accuracy'], 1.0)
        self.assertEqual(res['no_edit']['accuracy'], 0.0)
        q = subspace_basis(feats)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=torch.double), atol=1e-9))
        zr, zd = z[r], z[d]
        self.assertTrue(torch.allclose(patch(zr, zd, q) + (zd - zr) @ (torch.eye(32, dtype=torch.double) - q @ q.T),
                                       zd, atol=1e-9))

    def test_native_readout_that_uses_the_subspace_transfers_and_bypass_does_not(self):
        pairs, labels, z, product = synthetic('product')
        feats = self.feats(z[:, :6])  # discovery sees chi(a), chi(b), chi(a+b); keep the first two blocks
        zz = z
        feats_ab = [f for f in discover_subspace(zz, self.fit, CFG, P)['features']
                    if tuple(f['freq']) in {(1, 0), (0, 1)}]
        self.assertEqual(len(feats_ab), 2)
        r, d = interchange_pairs(pairs, self.test, 100, 1, P)
        used = native_interchange(product, zz, feats_ab, r, d, 4, 5)
        bypass = native_interchange(MockModel('bypass'), zz, feats_ab, r, d, 4, 5)
        self.assertGreater(used['discovered']['accuracy'], 0.95)
        self.assertEqual(used['lift_over_best_random'] > 0.5, True)
        self.assertEqual(bypass['discovered']['accuracy'], 0.0)
        self.assertEqual(bypass['complement_replacement']['accuracy'], 1.0)

    def test_empty_subspace_is_a_no_edit_failure(self):
        pairs, labels, z, model = synthetic('product')
        r, d = interchange_pairs(pairs, self.test, 50, 1, P)
        res = native_interchange(model, z, [], r, d, 3, 5)
        self.assertEqual(res['rank'], 0)
        self.assertEqual(res['discovered']['accuracy'], res['no_edit']['accuracy'])
        self.assertEqual(res['lift_over_best_random'], 0.0)

    def test_random_controls_are_norm_matched(self):
        pairs, labels, z, model = synthetic('product')
        feats = [f for f in discover_subspace(z, self.fit, CFG, P)['features'] if tuple(f['freq']) in {(1, 0), (0, 1)}]
        r, d = interchange_pairs(pairs, self.test, 80, 1, P)
        res = native_interchange(model, z, feats, r, d, 5, 5)
        disc = res['discovered']['mean_patch_norm']
        for c in res['random_norm_matched']:
            self.assertAlmostEqual(c['mean_patch_norm'], disc, places=5)

    def test_gate_row_requires_every_condition(self):
        g = json.loads((HERE / 'c1_manifest.json').read_text())['g1_gates']
        ok = {'seed': 1, 'present_task_accuracy': 1.0, 'rotation_accuracy_change': {'a': 0.0},
              'blind': {'targets': {'a': {'accuracy': 0.99}}, 'heldout_group_residual': 1e-4},
              'native_interchange': {'discovered': {'accuracy': 0.9}, 'lift_over_best_random': 0.5}}
        self.assertTrue(gate_row(ok, g)['all'])
        bad = json.loads(json.dumps(ok))
        bad['blind']['targets']['a']['accuracy'] = None
        self.assertFalse(gate_row(bad, g)['small_exponent_decoder'])
        bad = json.loads(json.dumps(ok))
        bad['native_interchange']['lift_over_best_random'] = 0.1
        self.assertFalse(gate_row(bad, g)['all'])

    def test_artifact_lock_passes_and_detects_tampering(self):
        base = json.loads((HERE / 'anti_collapse_manifest.json').read_text())
        self.assertTrue(verify_bf02_artifacts(base)['all_checkpoints_ok'])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for f in HERE.glob('blind_subspace_*'):
                if f.is_file():
                    shutil.copy(f, root / f.name)
            shutil.copytree(HERE / 'bf02_checkpoints', root / 'bf02_checkpoints')
            (root / 'bf02_checkpoints' / 'seed311_eq_norm_cov.pt').write_bytes(b'tampered')
            (root / 'bf02_checkpoints' / 'seed303_eq_norm.pt').unlink()
            lock = verify_bf02_artifacts(base, root)
            self.assertFalse(lock['all_checkpoints_ok'])
            self.assertEqual(lock['checkpoints']['311/eq_norm_cov'], 'hash_mismatch')
            self.assertEqual(lock['checkpoints']['303/eq_norm'], 'missing')


if __name__ == '__main__':
    unittest.main()

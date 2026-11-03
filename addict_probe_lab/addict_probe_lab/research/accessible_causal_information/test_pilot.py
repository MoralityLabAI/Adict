"""Exact sampler bridge, finite precision and held-out probe checks."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import unittest

import torch
from research.accessible_causal_information.pilot import (
    Encoder, exact_risk, quantize_nested, ridge_scores, states,
)


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())

    def test_shear_witness(self):
        pairs, labels = states(31)
        self.assertEqual(pairs[0].tolist(), [0, 0])
        self.assertEqual(pairs[61].tolist(), [1, 30])
        self.assertEqual(labels["sum"][0], labels["sum"][61])
        self.assertNotEqual(labels["shear"][0], labels["shear"][61])

    def test_exact_full_and_sum_code(self):
        pairs, labels = states(31)
        sum_code = [[s] for s in labels["sum"].tolist()]
        self.assertEqual(exact_risk(sum_code, labels["sum"]), 0)
        self.assertEqual(exact_risk(sum_code, labels["a"]), Fraction(30,31))
        self.assertEqual(exact_risk(sum_code, labels["shear"]), Fraction(30,31))
        self.assertEqual(exact_risk(pairs.tolist(), labels["shear"]), 0)

    def test_sampler_identity_by_direct_pair_enumeration(self):
        pairs, labels = states(5)
        code = [(int(a)%2, int(b)%3) for a, b in pairs]
        for target in labels.values():
            disagreement = Fraction()
            for i in range(len(code)):
                fiber = [j for j in range(len(code)) if code[j] == code[i]]
                disagreement += Fraction(sum(int(target[i] != target[j]) for j in fiber), len(code)*len(fiber))
            self.assertEqual(disagreement, exact_risk(code, target))

    def test_four_action_mixture(self):
        _, labels = states(31)
        code = [[s] for s in labels["sum"].tolist()]
        total = (exact_risk(code, labels["sum"]) +
                 2*exact_risk(code, (labels["sum"]+1)%31) +
                 exact_risk(code, labels["shear"]))/4
        self.assertEqual(total, Fraction(15,62))

    def test_finite_precision_nested_risks(self):
        _, labels = states(7)
        z = torch.randn(49, 4, generator=torch.Generator().manual_seed(14))
        codes = quantize_nested(z, torch.arange(30), [4,2,1,0], 2)
        previous = {target: Fraction() for target in labels}
        for k, code in codes.items():
            self.assertLessEqual(len(set(map(tuple, code.tolist()))), 2**(2*k))
            for target, y in labels.items():
                current = exact_risk(code.tolist(), y)
                self.assertGreaterEqual(current, previous[target])
                previous[target] = current
        for fine, coarse in ((4,2),(2,1),(1,0)):
            self.assertTrue(torch.equal(codes[fine][:,:coarse], codes[coarse]))

    def test_quantizer_fits_only_training_rows(self):
        g = torch.Generator().manual_seed(3)
        z = torch.randn(30,4,generator=g)
        train = torch.arange(20)
        first = quantize_nested(z,train,[4,2],3)
        changed = z.clone()
        changed[20:] *= 1000
        second = quantize_nested(changed,train,[4,2],3)
        for k in first:
            self.assertTrue(torch.equal(first[k][train],second[k][train]))

    def test_affine_decoder_recovers_linearly_available_labels(self):
        labels = torch.arange(3).repeat(10)
        z = torch.nn.functional.one_hot(labels,3).float()
        scores = ridge_scores(z,labels,torch.arange(21),torch.arange(21,30),3,.01)
        self.assertEqual(scores["accuracy"],1)
        self.assertLess(scores["onehot_mse"],1e-6)

    def test_model_shapes_and_identical_start(self):
        torch.manual_seed(1)
        model = Encoder(self.manifest)
        clone = deepcopy(model)
        pairs,_ = states(31)
        for left,right in zip(model(pairs),clone(pairs)):
            self.assertTrue(torch.equal(left,right))
        task, hidden, z = model(pairs)
        self.assertEqual(task.shape,(961,31))
        self.assertEqual(hidden.shape,(961,31))
        self.assertEqual(z.shape,(961,32))

    def test_training_step(self):
        model = Encoder(self.manifest)
        pairs, labels = states(31)
        opt = torch.optim.AdamW(model.parameters(),lr=.001)
        logits,_,_ = model(pairs)
        loss = torch.nn.functional.cross_entropy(logits,labels["sum"])
        loss.backward()
        self.assertIsNotNone(model.body[0].weight.grad)
        self.assertIsNone(model.read_a.weight.grad)
        opt.step()

    def test_invalid_input(self):
        with self.assertRaises(ValueError): states(1)
        with self.assertRaises(ValueError): exact_risk([],[])
        with self.assertRaises(ValueError): exact_risk([[1]],[1,2])
        with self.assertRaises(ValueError): quantize_nested(torch.ones(5,2),torch.arange(4),[3],2)
        with self.assertRaises(ValueError): quantize_nested(torch.ones(5,2),torch.arange(4),[1],0)


if __name__ == "__main__":
    unittest.main()

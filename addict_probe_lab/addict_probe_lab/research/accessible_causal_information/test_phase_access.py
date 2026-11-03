"""Analytic readout separation with held-out input keys."""
import math
import random
import unittest

from research.accessible_causal_information.phase_access import (
    phase_code, label_offset, decode_phase, affine_onehot_risk,
)

class PhaseAccessTests(unittest.TestCase):
    def test_known_product_oracle(self):
        p=31
        self.assertEqual({phase_code(s,a,p) for s in range(p) for a in range(p)}.__len__(),p*p)
        for s in range(p):
            for a in range(p):
                self.assertEqual(decode_phase(s,a,0,p),(s+a)%p)

    def test_one_label_offset_holds_on_untouched_keys(self):
        p=31
        keys=[(s,a) for s in range(p) for a in range(p)]
        random.Random(20261102).shuffle(keys)
        train, heldout=keys[:480],keys[480:]
        delta=11
        s,a=train[0]
        learned=label_offset(s,a,(s+a+delta)%p,p)
        self.assertEqual(learned,delta)
        self.assertEqual(len(heldout),481)
        self.assertTrue(all(decode_phase(x,y,learned,p)==(x+y+delta)%p for x,y in heldout))
        self.assertFalse(any((x,y) in set(train) for x,y in heldout))

    def test_affine_population_bound(self):
        p=31
        self.assertTrue(math.isclose(affine_onehot_risk(p),30/31,abs_tol=1e-15))
        # Directly verify the best constant additive prediction's one-hot loss.
        risk=sum((1-1/p)**2+(p-1)*(1/p)**2 for _ in range(p*p))/(p*p)
        self.assertAlmostEqual(risk,affine_onehot_risk(p))

    def test_affine_strict_margin_impossibility(self):
        # A class-score difference f(s)+g(a) has the SAME sum over every
        # target diagonal a=t-s. Thus it cannot be strictly positive over
        # every t diagonal and strictly negative over another t' diagonal.
        p=7
        f=[float(i*i) for i in range(p)]
        g=[float((-i)%p) for i in range(p)]
        diagonal_sums=[sum(f[s]+g[(t-s)%p] for s in range(p)) for t in range(p)]
        self.assertTrue(all(x==diagonal_sums[0] for x in diagonal_sums))

if __name__=='__main__': unittest.main()

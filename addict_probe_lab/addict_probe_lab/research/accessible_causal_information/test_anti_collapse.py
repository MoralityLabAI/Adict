"""Analytic character constraints and deterministic anti-collapse diagnostics."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from research.accessible_causal_information.anti_collapse import (
    components, ideal_phases, phase_alignment, phase_offset_decode,
    rotations, shift_indices, spectrum_diagnostic, train_only_character_basis_decode,
)
from research.accessible_causal_information.pilot import states


class AntiCollapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shift_action_is_exact_on_two_characters(self):
        for p in (5,7,31):
            z=ideal_phases(p)
            a,b=shift_indices(p)
            d=components(z,p,a,b)
            self.assertLess(d['equivariance'].item(),1e-11)
            self.assertLess(d['norm'].item(),1e-11)
            self.assertLess(d['covariance'].item(),1e-10)
            self.assertAlmostEqual(d['mean_phase_energy'].item(),1.0,places=5)
            self.assertTrue(torch.allclose(d['covariance_matrix'],0.5*torch.eye(4),atol=1e-5))

    def test_zero_code_is_exactly_equivariant_and_collapsed(self):
        p=7
        z=torch.zeros((p*p,32))
        a,b=shift_indices(p)
        d=components(z,p,a,b)
        self.assertEqual(d['equivariance'].item(),0)
        self.assertAlmostEqual(d['norm'].item(),1.0)
        self.assertAlmostEqual(d['covariance'].item(),1.0)
        self.assertEqual(d['mean_phase_energy'].item(),0)

    def test_fixed_nonzero_phases_cannot_be_shift_equivariant(self):
        p=7
        z=torch.zeros((p*p,32))
        z[:,0]=1.0
        z[:,2]=1.0
        a,b=shift_indices(p)
        d=components(z,p,a,b)
        self.assertGreater(d['equivariance'].item(),0.01)
        self.assertLess(d['norm'].item(),1e-10)
        self.assertAlmostEqual(d['covariance'].item(),1.0)

    def test_shift_indices_are_bijections(self):
        for p in (3,7,31):
            a,b=shift_indices(p)
            self.assertEqual(sorted(a.tolist()),list(range(p*p)))
            self.assertEqual(sorted(b.tolist()),list(range(p*p)))
            ia,ib=rotations(ideal_phases(p),p)
            self.assertEqual(ia.shape,(p*p,4))
            self.assertEqual(ib.shape,(p*p,4))

    def test_phase_product_and_alignment_on_ideal_characters(self):
        for p in (5,31):
            _,y=states(p)
            z=ideal_phases(p)
            train=torch.arange(p*p//2)
            test=torch.arange(p*p//2,p*p)
            d=phase_offset_decode(z,y['shear'],train,test,p)
            self.assertAlmostEqual(d['heldout_accuracy'],1.0,places=10)
            a=phase_alignment(z,y,train,test,p)
            for v in a['axes']:
                self.assertAlmostEqual(v['heldout_cosine'],1.0,places=5)
                self.assertAlmostEqual(v['heldout_norm'],1.0,places=5)

    def test_arbitrary_phase_origin_can_be_calibrated(self):
        p=31
        _,y=states(p)
        z=ideal_phases(p).double()
        c1,c2=math.cos(.61),math.sin(.61)
        z0=z[:,:2].clone()
        z[:,:2]=torch.stack((c1*z0[:,0]-c2*z0[:,1],c2*z0[:,0]+c1*z0[:,1]),1)
        train=torch.arange(480)
        test=torch.arange(480,961)
        d=phase_offset_decode(z,y['shear'],train,test,p)
        self.assertAlmostEqual(d['heldout_accuracy'],1.0,places=8)

    def test_corrupt_phase_layout_is_detectable(self):
        p=7
        z=ideal_phases(p)
        a,b=shift_indices(p)
        broken=z.clone()
        broken[:,2:4]=z[:,:2]
        d=components(broken,p,a,b)
        self.assertGreater(d['equivariance'].item(),.01)
        # It still has unit norm in both blocks, so that term alone is weak.
        self.assertLess(d['norm'].item(),1e-10)

    def test_wrong_character_passes_norm_and_whitening(self):
        p=31
        pairs,_=states(p)
        a,b=pairs[:,0],pairs[:,1]
        th=4*math.pi/p # frequency 2 instead of prescribed frequency 1
        s=a+b
        z=torch.stack((torch.cos(th*s),torch.sin(th*s),
                       torch.cos(th*a),torch.sin(th*a)),1)
        idxa,idxb=shift_indices(p)
        d=components(z,p,idxa,idxb)
        self.assertLess(d['norm'].item(),1e-10)
        self.assertLess(d['covariance'].item(),1e-10)
        self.assertAlmostEqual(d['equivariance'].item(),6*math.sin(math.pi/p)**2,places=5)
        modes=spectrum_diagnostic(z,p)
        self.assertEqual([(v['strongest_modes'][0]['k'],v['strongest_modes'][0]['l']) for v in modes],[(2,2),(2,0)])
        self.assertTrue(all(v['expected_power_fraction'] <1e-8 for v in modes))

    def test_train_only_basis_decode_of_operand_characters(self):
        p=31
        pairs,labels=states(p)
        a,b=pairs[:,0],pairs[:,1]
        # Two characters form a different group basis: a and -b.
        theta=2*math.pi/p
        z=torch.stack((torch.cos(theta*a),torch.sin(theta*a),
                       torch.cos(-theta*b),torch.sin(-theta*b)),1)
        permutation=torch.randperm(len(pairs), generator=torch.Generator().manual_seed(20261102))
        tr,te=permutation[:480],permutation[480:]
        result=train_only_character_basis_decode(z,pairs,labels['shear'],tr,te,p)
        self.assertEqual(result['modes'],[(1,0),(0,p-1)])
        self.assertTrue(result['recoverable_basis'])
        self.assertEqual(result['exponents'],[2,p-1])
        self.assertEqual(result['signed_exponents'],[2,-1])
        self.assertEqual(result['heldout_accuracy'],1.0)
        # The original fixed product decodes a+b instead of 2a+b.
        self.assertLess(phase_offset_decode(z,labels['shear'],tr,te,p)['heldout_accuracy'],0.1)

    def test_signed_group_exponent_avoids_phase_noise_amplification(self):
        p=31
        pairs,labels=states(p)
        a,b=pairs[:,0],pairs[:,1]
        theta=2*math.pi/p
        # Small varying angle noise on the negative-b character. The
        # algebraic p-1 exponent amplifies noise 30-fold; -1 does not.
        noise=0.015*torch.cos((3*a+5*b).double())
        z=torch.stack((torch.cos(theta*a),torch.sin(theta*a),
                       torch.cos(-theta*b+noise),torch.sin(-theta*b+noise)),1)
        permutation=torch.randperm(len(pairs), generator=torch.Generator().manual_seed(20261102))
        r=train_only_character_basis_decode(z,pairs,labels['shear'],permutation[:480],permutation[480:],p)
        self.assertEqual(r['signed_exponents'],[2,-1])
        self.assertGreater(r['heldout_accuracy'],.98)

    def test_manifest_frozen_before_results(self):
        m=json.loads(Path(__file__).with_name('anti_collapse_manifest.json').read_text())
        self.assertEqual(m['seeds'],[303,307])
        self.assertEqual(m['equivariance_weight'],1.0)
        self.assertEqual(m['norm_weight'],1.0)
        self.assertEqual(m['covariance_weight'],1.0)
        self.assertEqual(m['steps'],1500)
        self.assertEqual(m['conditions'],['task_only','equivariance','eq_norm','eq_norm_cov','task_and_a'])

if __name__=='__main__':unittest.main()

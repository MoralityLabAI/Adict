"""Dependency-free regression tests for the four-axis audit and gate replay."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from research.accessible_causal_information.four_axis_audit import (
        EvidenceError, audit, finite_code_risk, _seed_results,
    )
except ImportError:  # Enables independent inspection of this standalone artifact.
    from four_axis_audit import EvidenceError, audit, finite_code_risk, _seed_results


def fixture():
    manifest = {
        "schema":"adict-closeout-c1/v0.1", "seeds":[401,409,419],
        "conditions":["task_only","eq_norm_cov"],
        "primary_target":"shear_2_1", "targets":{"shear_2_1":[2,1],"shear_1_2":[1,2]},
        "interchange":{"pairs":256,"random_controls":16},
        "g1_gates": {"present_task_accuracy_min":0.99,
        "small_exponent_decoder_accuracy_min":0.95,"heldout_group_residual_max":0.01,
        "rotation_accuracy_change_max":0.02,"interchange_accuracy_min":0.8,
        "interchange_lift_over_best_random_min":0.2}}
    results = []
    per_seed = []
    for seed in manifest["seeds"]:
        for condition in manifest["conditions"]:
            rates = {"shear_2_1":{"abstained":False,"accuracy":.956},
                     "shear_1_2":{"abstained":False,"accuracy":.994}}
            if seed == 419:
                rates = {"shear_2_1":{"abstained":True,"accuracy":None},
                         "shear_1_2":{"abstained":True,"accuracy":None}}
            results.append({"seed":seed,"condition":condition,
                "present_task_accuracy":1,
                "blind":{"targets":rates, "frequencies":[[0,1],[1,0]],
                         "heldout_group_residual":0.0176 if seed==419 else .0002},
                "rotation_accuracy_change":{"shear_2_1":None if seed==419 else 0.0,
                                            "shear_1_2":None if seed==419 else 0.0},
                "native_interchange":{"pairs":256,"rank":2 if seed==419 else 4,
                      "discovered":{"accuracy":0},
                      "random_norm_matched":[{"accuracy":0} for _ in range(16)],
                      "best_random_norm_matched_accuracy":0,
                      "lift_over_best_random":0,
                      "full_code_donor":{"accuracy":1},
                      "complement_replacement":{"accuracy":1},
                      "no_edit":{"accuracy":0}}})
        checks={"present_task":True,"small_exponent_decoder":seed!=419,
                "heldout_group_residual":seed!=419,"rotation_change":seed!=419,
                "interchange_accuracy":False,"interchange_lift":False}
        per_seed.append({"seed":seed,**checks,"all":False})
    c1={"schema":manifest["schema"],"manifest":manifest,"results":results,
        "verdict":{"G1_pass":False,"native_gate_pass":False,"per_seed":per_seed},
        "bf02_artifact_lock":{"all_checkpoints_ok":True}}
    original={"schema":"learned-intervention-transfer/v0.1","n_states":961,
        "records":[{"quantized_oracle":[{"state_code_bits":96,"attained_codes":961,
                   "risk":{"shear":"0"}}]} for _ in range(6)]}
    bf02={"schema":"adict-blind-subspace/v0.1", "manifest":{"confirmation_seed":311},
          "results":[{"seed":311,"condition":"eq_norm_cov","role":"confirmation",
                    "blind":{"accuracy":.9833679833679834,
                             "group_law_residual":.00017416743592565687,
                             "frequencies":[[0,1],[1,0]]}}],
          "verdict":{"blind_accuracy":.9833679833679834,
                     "group_law_residual":.00017416743592565687}}
    return manifest,c1,original,bf02


class ExactFiniteCodeTests(unittest.TestCase):
    def test_fiber_information_loss(self):
        self.assertEqual(finite_code_risk([0,0,1,1],[0,1,1,1]),Fraction(1,4))
        self.assertEqual(finite_code_risk([0,1,2,3],[0,1,1,1]),0)
        self.assertEqual(finite_code_risk([0,0,0,0],[0,1,1,1]),Fraction(3,8))

    def test_modular_sum_code_and_shear(self):
        p=31
        codes=[]; now=[]; future=[]
        for a in range(p):
            for b in range(p):
                codes.append((a+b)%p)
                now.append((a+b)%p)
                future.append((2*a+b)%p)
        self.assertEqual(finite_code_risk(codes,now),Fraction())
        self.assertEqual(finite_code_risk(codes,future),Fraction(30,31))

    def test_invalid_inputs(self):
        with self.assertRaises(EvidenceError): finite_code_risk([],[])
        with self.assertRaises(EvidenceError): finite_code_risk([0],[0,1])
        with self.assertRaises(EvidenceError): finite_code_risk([{}],[0])

    def test_nested_codes_monotone_fixed_target(self):
        labels=[0,1,2,0,2,1]
        risk0=finite_code_risk([(i,) for i in range(6)],labels)
        risk1=finite_code_risk([i//2 for i in range(6)],labels)
        risk2=finite_code_risk([0]*6,labels)
        self.assertLessEqual(risk0,risk1)
        self.assertLessEqual(risk1,risk2)


class EvidenceReplayTests(unittest.TestCase):
    def setUp(self):
        self.manifest,self.c1,self.original,self.bf02=fixture()

    def _root(self,root):
        for name,data in [("c1_manifest.json",self.manifest),("c1_results.json",self.c1),
                          ("original_results.json",self.original),("blind_subspace_results.json",self.bf02)]:
            (root/name).write_text(json.dumps(data),encoding="utf-8")

    def test_gate_fails_despite_accessibility(self):
        seeds=_seed_results(self.c1,self.manifest)
        self.assertEqual(len(seeds),3)
        self.assertTrue(seeds[0]["gates"]["small_exponent_decoder"])
        self.assertFalse(seeds[0]["gates"]["interchange_accuracy"])
        self.assertTrue(seeds[2]["accessibility"]["primary_abstained"])
        self.assertFalse(any(s["passed_all"] for s in seeds))

    def test_summary_replay_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self._root(root)
            report=audit(root)
            self.assertEqual(report["status"],"closed_g1_failed")
            self.assertEqual(report["axes"]["storage"]["oracle_records_with_zero_shear_risk_at_96_bits"],6)
            self.assertEqual(len(report["provenance"]["sha256"]),4)
            self.assertFalse(report["provenance"]["independent_binary_checkpoint_verification"])
            self.assertEqual([x["discovered_successes"] for x in report["axes"]["native_causal_use"]["c1_seeds"]],[0,0,0])

    def test_mismatched_manifest_refused(self):
        bad=deepcopy(self.manifest);bad["g1_gates"]["interchange_accuracy_min"] = .01
        with self.assertRaisesRegex(EvidenceError,"manifest differing"):
            _seed_results(self.c1,bad)

    def test_false_saved_verdict_refused(self):
        self.c1["verdict"]["G1_pass"] = True
        with self.assertRaisesRegex(EvidenceError,"global C1 verdict"):
            _seed_results(self.c1,self.manifest)

    def test_missing_random_controls_refused(self):
        self.c1["results"][1]["native_interchange"]["random_norm_matched"].pop()
        with self.assertRaisesRegex(EvidenceError,"random controls"):
            _seed_results(self.c1,self.manifest)

    def test_incomplete_rotation_coverage_refused(self):
        row=next(x for x in self.c1["results"] if x["seed"]==401 and x["condition"]=="eq_norm_cov")
        del row["rotation_accuracy_change"]["shear_1_2"]
        with self.assertRaisesRegex(EvidenceError,"rotation change target coverage"):
            _seed_results(self.c1,self.manifest)

    def test_g1_abstention_counted_as_failure(self):
        seeds=_seed_results(self.c1,self.manifest)
        self.assertEqual(seeds[2]["accessibility"]["heldout_primary"],0)
        self.assertFalse(seeds[2]["gates"]["small_exponent_decoder"])

    def test_source_overstatement_refused(self):
        self.original["records"][0]["quantized_oracle"][0]["attained_codes"] = 960
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self._root(root)
            with self.assertRaisesRegex(EvidenceError,"96-bit"):
                audit(root)

    def test_cli_failure_is_a_result_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self._root(root)
            from pathlib import Path as P
            source=P(__file__).resolve().with_name('four_axis_audit.py')
            proc=subprocess.run([sys.executable,str(source),"--root",str(root),"--out",str(root/'out.json')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads((root/'out.json').read_text())["c1"]["all_pass"],False)
            strict=subprocess.run([sys.executable,str(source),"--root",str(root),"--require-pass"],capture_output=True,text=True)
            self.assertEqual(strict.returncode,2)


if __name__ == '__main__':
    unittest.main()
